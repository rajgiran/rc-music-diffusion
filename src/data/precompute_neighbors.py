#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precompute nearest neighbors (and per-query style vectors) for a fixed set of queries.

Two query modes:
  1) --query_manifest  JSONL with {"path": "<abs or rel audio path>", ["id": int]} per line
     (Audio is loaded, resampled to 48kHz, cropped/padded to --clip_seconds, and CLAP-embedded.)
  2) --query_emb + --query_mapping  Use precomputed query CLAP embeddings and mapping JSONL
     (mapping JSONL must have {"path": str} per row, aligned to embedding rows)

Indexes (banks):
  For each index you query (primary; optional aux), provide:
    --*_index_dir   (contains index.faiss + mapping.jsonl)
    --*_emb         (the .npy embeddings used to build that index)

Outputs (in --out_dir):
  - neighbors.jsonl   one line per query:
      {"q_idx","q_path","used_index","mean_sim","max_sim","neighbors":[{"rank","idx","path","sim"}]}
  - agg_emb.npy       [Nq, 512] per-query style vectors aggregated from top-K neighbors (headered .npy via open_memmap)
  - (optional) query_ids.npy if your query manifest has {"id": ...} entries

Typical usage (primary only):
  python precompute_neighbors.py \
    --query_manifest /scratch/.../union_queries.jsonl \
    --out_dir        /scratch/.../retrieval_cache/primary \
    --primary_index_dir /scratch/.../indexes/retrieval_primary \
    --primary_emb       /scratch/.../fsd50k/clap_embed_primary/clap.npy \
    --topk 50 --style_top_m 8 --style_agg softmax --softmax_tau 0.08 \
    --batch_size 32 --clip_seconds 10 --amp --device cuda

Optional aux fallback (if mean(sim) < --fallback_threshold on primary):
  ... add --aux_index_dir --aux_emb --fallback_threshold 0.30
"""

from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
import faiss

import torch
from transformers import ClapModel, ClapProcessor
from numpy.lib.format import open_memmap

# -------------------------- defaults --------------------------
SR = 48_000

# -------------------------- I/O utils -------------------------
def read_jsonl(p: Path) -> List[dict]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows

def resolve_path(path_str: str, root: Optional[Path]) -> Path:
    p = Path(path_str)
    if not p.is_absolute() and root is not None:
        return (root / p).resolve()
    return p.resolve()

# --------------------- audio / CLAP helpers --------------------
def fix_len(x: np.ndarray, T: int) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if x.shape[0] >= T:
        return x[:T]
    out = np.zeros(T, dtype=np.float32)
    out[: x.shape[0]] = x
    return out

def load_wave_48k_10s(p: Path, clip_seconds: float) -> np.ndarray:
    """Mono, 48k, exactly clip_seconds (crop/pad)."""
    x, sr = sf.read(p, dtype="float32", always_2d=False)
    if getattr(x, "ndim", 1) > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return fix_len(x, int(round(clip_seconds * SR)))

def load_clap(model_name="laion/clap-htsat-unfused", device: str = "cuda", amp: bool = True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    proc = ClapProcessor.from_pretrained(model_name)
    model = ClapModel.from_pretrained(model_name).to(device).eval()
    return model, proc, device, (amp and device.startswith("cuda"))

def clap_embed_batch(model, proc, device: str, amp: bool, waves: List[np.ndarray]) -> np.ndarray:
    """Return L2-normalized CLAP embeddings for a batch of mono 48k waves."""
    with torch.no_grad():
        inputs = proc(audios=waves, sampling_rate=SR, return_tensors="pt", padding=True).to(device)
        if amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                z = model.get_audio_features(**inputs)
        else:
            z = model.get_audio_features(**inputs)
        z = torch.nn.functional.normalize(z, dim=-1)
        return z.cpu().numpy().astype(np.float32, copy=False)

# ---------------------- index / search utils -------------------
def load_index(index_dir: Path) -> Tuple[faiss.Index, List[str]]:
    index_path = index_dir / "index.faiss"
    mapping_path = index_dir / "mapping.jsonl"
    if not index_path.exists() or not mapping_path.exists():
        raise SystemExit(f"Missing index components in {index_dir}: need index.faiss and mapping.jsonl")
    index = faiss.read_index(str(index_path))
    mapping = [json.loads(l)["path"] for l in mapping_path.open("r", encoding="utf-8") if l.strip()]
    return index, mapping

def load_index_embeddings(emb_path: Path) -> np.ndarray:
    if not emb_path.exists():
        raise SystemExit(f"Embeddings not found: {emb_path}")
    X = np.load(emb_path).astype("float32", copy=False)
    # L2 normalize for cosine/IP
    faiss.normalize_L2(X)
    return X

# ----------------------------- main ----------------------------
def main():
    ap = argparse.ArgumentParser()

    # Query sources (choose one mode)
    ap.add_argument("--query_manifest", type=Path, help="JSONL with {'path': ...} (audio will be embedded)")
    ap.add_argument("--query_root", type=Path, default=None, help="Root to resolve relative paths in manifest")
    ap.add_argument("--query_emb", type=Path, help="Precomputed query CLAP .npy")
    ap.add_argument("--query_mapping", type=Path, help="Mapping JSONL aligned to --query_emb")

    # Output
    ap.add_argument("--out_dir", type=Path, required=True)

    # Primary index (required)
    ap.add_argument("--primary_index_dir", type=Path, required=True, help="Folder with index.faiss + mapping.jsonl")
    ap.add_argument("--primary_emb", type=Path, required=True, help="Embeddings .npy used to build primary index")

    # Optional aux index (fallback when mean(sim) is low on primary)
    ap.add_argument("--aux_index_dir", type=Path, help="Folder with index.faiss + mapping.jsonl")
    ap.add_argument("--aux_emb", type=Path, help="Embeddings .npy used to build aux index")
    ap.add_argument("--fallback_threshold", type=float, default=0.30,
                    help="If mean(sim) on primary < this, use aux index for that query")

    # Retrieval / style settings
    ap.add_argument("--topk", type=int, default=50, help="K neighbors to fetch")
    ap.add_argument("--style_top_m", type=int, default=8, help="Neighbors used for style aggregation (<= topk, 0 disables)")
    ap.add_argument("--style_agg", type=str, default="softmax", choices=["mean", "softmax"], help="Aggregation type")
    ap.add_argument("--softmax_tau", type=float, default=0.08, help="Softmax temperature")

    # CLAP / audio embedding (only used in query_manifest mode)
    ap.add_argument("--clip_seconds", type=float, default=10.0, help="Query crop/pad length (sec)")
    ap.add_argument("--batch_size", type=int, default=32, help="CLAP batch size for query embedding")
    ap.add_argument("--amp", action="store_true", help="Use CUDA autocast fp16 for CLAP if available")
    ap.add_argument("--device", type=str, default=None, help="cuda / cpu")

    args = ap.parse_args()

    # Hygiene: threads + determinism-ish
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.backends.cudnn.benchmark = False  # avoid autotune nondeterminism

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load indexes + their embeddings + mapping
    P_index, P_paths = load_index(args.primary_index_dir)
    P_X = load_index_embeddings(args.primary_emb)

    use_aux = bool(args.aux_index_dir and args.aux_emb)
    if use_aux:
        A_index, A_paths = load_index(args.aux_index_dir)
        A_X = load_index_embeddings(args.aux_emb)
    else:
        A_index = A_paths = A_X = None

    # Clamp style_top_m to topk (safety)
    if args.style_top_m > args.topk:
        args.style_top_m = args.topk

    # Load queries
    # Mode 1: precomputed query embeddings
    if args.query_emb and args.query_mapping:
        Q = np.load(args.query_emb).astype("float32", copy=False)
        faiss.normalize_L2(Q)
        Q_paths = [json.loads(l)["path"] for l in open(args.query_mapping, "r", encoding="utf-8") if l.strip()]
        if Q.shape[0] != len(Q_paths):
            raise SystemExit("query_emb and query_mapping size mismatch.")
        def query_generator():
            for i in range(Q.shape[0]):
                yield i, Q_paths[i], Q[i:i+1]
        Nq = Q.shape[0]
        ids_from_manifest = None
        embed_mode = "precomputed"
    # Mode 2: manifest (re-embed)
    elif args.query_manifest:
        rows = read_jsonl(args.query_manifest)
        Nq = len(rows)
        ids_from_manifest = [int(r["id"]) for r in rows] if rows and "id" in rows[0] else None
        model, proc, device, amp = load_clap(device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
                                             amp=args.amp)
        bsz = int(args.batch_size)
        def query_generator():
            i = 0
            while i < Nq:
                j = min(i + bsz, Nq)
                batch_paths = [resolve_path(rows[k]["path"], args.query_root) for k in range(i, j)]
                waves = [load_wave_48k_10s(p, args.clip_seconds) for p in batch_paths]
                Z = clap_embed_batch(model, proc, device, amp, waves)  # [B,512], L2-normalized
                for bi in range(Z.shape[0]):
                    yield i + bi, str(batch_paths[bi]), Z[bi:bi+1]
                i = j
        embed_mode = "manifest"
    else:
        raise SystemExit("Provide either (--query_emb AND --query_mapping) or --query_manifest.")

    # Prepare outputs
    neigh_f = (out_dir / "neighbors.jsonl").open("w", encoding="utf-8")
    style_m = None
    if args.style_top_m > 0:
        style_m = open_memmap(out_dir / "agg_emb.npy", mode="w+", dtype=np.float32, shape=(Nq, 512))

    # Process queries
    for qi, qpath, z in tqdm(query_generator(), total=Nq, desc=f"Precompute[{embed_mode}]"):
        # z is [1,512], already normalized (precomputed or from CLAP helper)
        # 1) search primary
        D, I = P_index.search(z, args.topk)
        sims, idxs = D[0], I[0]
        mean_sim = float(np.mean(sims)) if sims.size else 0.0
        used = "primary"
        neigh_paths = P_paths
        bankX = P_X

        # 2) optional fallback to aux if weak match
        if use_aux and mean_sim < args.fallback_threshold:
            D2, I2 = A_index.search(z, args.topk)
            sims, idxs = D2[0], I2[0]
            used = "aux"
            neigh_paths = A_paths
            bankX = A_X
            mean_sim = float(np.mean(sims)) if sims.size else 0.0

        # 3) write neighbors
        rec = {
            "q_idx": int(qi),
            "q_path": str(qpath),
            "used_index": used,
            "mean_sim": mean_sim,
            "max_sim": float(np.max(sims)) if sims.size else 0.0,
            "neighbors": [
                {"rank": int(r+1),
                 "idx": int(int(idxs[r])),
                 "path": neigh_paths[int(idxs[r])] if 0 <= int(idxs[r]) < len(neigh_paths) else None,
                 "sim": float(sims[r])}
                for r in range(min(len(idxs), args.topk))
            ],
        }
        neigh_f.write(json.dumps(rec) + "\n")

        # 4) style vector (aggregate)
        if style_m is not None and len(idxs) > 0:
            M = min(args.style_top_m, len(idxs))
            V = bankX[idxs[:M]]                    # [M,512], already L2-normalized
            if args.style_agg == "mean":
                w = np.ones((M, 1), dtype=np.float32) / float(M)
            else:
                s = sims[:M].astype(np.float32, copy=False)
                s = s - (s.max() if s.size else 0.0)
                w = np.exp(s / float(args.softmax_tau)).reshape(-1, 1)
                w /= (w.sum() + 1e-8)
            v = (V * w).sum(axis=0)
            v /= (np.linalg.norm(v) + 1e-9)
            style_m[qi] = v.astype(np.float32, copy=False)

    neigh_f.close()
    if style_m is not None:
        # ensure memmap is flushed and header is valid
        del style_m

    # Optional ids (manifest mode only, and only if present)
    if embed_mode == "manifest" and ids_from_manifest is not None:
        np.save(out_dir / "query_ids.npy", np.array(ids_from_manifest, dtype=np.int64))

    print(f"[DONE] neighbors.jsonl → {out_dir/'neighbors.jsonl'}")
    if args.style_top_m > 0:
        print(f"[DONE] agg_emb.npy    → {out_dir/'agg_emb.npy'}")

if __name__ == "__main__":
    main()