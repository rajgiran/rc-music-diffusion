#!/usr/bin/env python3
# build_style_pair_cache.py
# Turn neighbors.jsonl (+ bank embeddings) into a cache of retrieval pairs and style mixtures.
#
# Inputs:
#   --neighbors_jsonl   : from precompute_neighbors.py (has used_index, neighbors: [{idx, sim, path}, ...])
#   --primary_emb       : .npy used to build primary faiss (shape [Np,512], float32)
#   --aux_emb           : (optional) .npy used for aux faiss (shape [Na,512])
#   --top_m             : neighbors to aggregate (default 8)
#   --softmax_tau       : temperature for softmax weights (default 0.08)
#   --out_dir           : where to write outputs
#   --save_neighbor_embs: write neighbor_emb_topM.npy [N, M, 512] (off by default)
#   --agg_ref           : (optional) path to agg_emb.npy to compare against (supports raw memmap)
#
# Outputs in out_dir:
#   style_topM.npy            [N, 512]     — softmax mixture of top_m
#   neighbor_idx_topM.npy     [N, M]       — int32 indices (into primary/aux emb tables, per used bank)
#   neighbor_w_topM.npy       [N, M]       — float32 weights (softmax)
#   pairs_flat.jsonl                        — flattened pairs per query: {query_id, rank, bank, idx, sim, weight, path}
#   (optional) neighbor_emb_topM.npy [N,M,512]  — only if --save_neighbor_embs
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from tqdm import tqdm

def _softmax(x: np.ndarray, tau: float) -> np.ndarray:
    """Numerically-stable softmax(x / tau)."""
    if x.size == 0:
        return np.zeros((0,), dtype=np.float32)
    z = (x.astype(np.float32) / max(tau, 1e-6))
    z -= np.max(z)
    ez = np.exp(z, dtype=np.float32)
    s = ez.sum()
    return ez / (s + 1e-8)

def _load_neighbors(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open() as f:
        for ln in f:
            if ln.strip():
                rows.append(json.loads(ln))
    return rows

def _normalize_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return (X / n).astype(np.float32)

def _robust_load_matrix(path: Path, N: int, D: int) -> np.ndarray:
    """
    Robustly load a [N,D] float32 matrix:
      1) try np.load (handles true .npy or .npz with header)
      2) fall back to raw memmap with known shape (our pipeline often writes via np.memmap)
    """
    try:
        arr = np.load(path, mmap_mode="r")
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == N and arr.shape[1] == D:
            return arr
        # if header exists but shape mismatched, still fall back to memmap
    except Exception:
        pass
    return np.memmap(path, mode="r", dtype=np.float32, shape=(N, D))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neighbors_jsonl", type=Path, required=True)
    ap.add_argument("--primary_emb", type=Path, required=True)
    ap.add_argument("--aux_emb", type=Path)  # optional
    ap.add_argument("--top_m", type=int, default=8)
    ap.add_argument("--softmax_tau", type=float, default=0.08)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--save_neighbor_embs", action="store_true")
    ap.add_argument("--agg_ref", type=Path, help="Optional agg_emb.npy to compare against (supports memmap)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # load neighbor records
    NBR = _load_neighbors(args.neighbors_jsonl)
    N = len(NBR)
    if N == 0:
        raise SystemExit("neighbors.jsonl appears empty.")

    # load banks (mmap: constant RAM)
    P = np.load(args.primary_emb, mmap_mode="r").astype(np.float32)
    D = int(P.shape[1])
    A: Optional[np.ndarray] = None
    if args.aux_emb:
        A = np.load(args.aux_emb, mmap_mode="r").astype(np.float32)

    # (Re)normalize to unit L2 (standard for cosine/IP retrieval)
    Pn = _normalize_rows(P)
    An = _normalize_rows(A) if A is not None else None

    top_m = int(args.top_m)
    if top_m <= 0:
        raise SystemExit("--top_m must be >= 1")

    # allocate outputs as memmaps (fast, low RAM)
    style = np.memmap(args.out_dir / "style_topM.npy", mode="w+", dtype=np.float32, shape=(N, D))
    idx_mat = np.memmap(args.out_dir / "neighbor_idx_topM.npy", mode="w+", dtype=np.int32,   shape=(N, top_m))
    w_mat   = np.memmap(args.out_dir / "neighbor_w_topM.npy",   mode="w+", dtype=np.float32, shape=(N, top_m))
    emb_mat = None
    if args.save_neighbor_embs:
        emb_mat = np.memmap(args.out_dir / "neighbor_emb_topM.npy", mode="w+", dtype=np.float32, shape=(N, top_m, D))

    flat = (args.out_dir / "pairs_flat.jsonl").open("w")

    mism_bank = 0
    for i, row in enumerate(tqdm(NBR, desc="Pairs")):
        used = row.get("used_index", "primary")  # "primary" or "aux"
        neighs = row.get("neighbors", [])
        # take up to top_m neighbors (might be fewer if upstream trimmed)
        neighs = neighs[:top_m]

        sims = np.array([n.get("sim", 0.0) for n in neighs], dtype=np.float32)
        idxs = np.array([int(n.get("idx", -1)) for n in neighs], dtype=np.int64)
        paths = [n.get("path") for n in neighs]

        # default fill (in case fewer than top_m)
        idx_mat[i, :] = -1
        w_mat[i, :]   = 0.0
        if emb_mat is not None:
            emb_mat[i, :, :] = 0.0

        if sims.size == 0:
            style[i, :] = 0.0
            # no flattened pairs to write
            continue

        # choose bank
        if used == "primary":
            bank = Pn
        else:
            if An is None:
                # neighbors say aux but aux embeddings weren’t provided; fall back to primary
                used = "primary"
                bank = Pn
                mism_bank += 1
            else:
                bank = An

        # weights
        w = _softmax(sims, tau=args.softmax_tau)
        # mixture using available neighbors
        Vecs = bank[idxs]  # [m, D] where m = len(idxs)
        mix = (Vecs * w[:, None]).sum(axis=0)
        mix /= (np.linalg.norm(mix) + 1e-12)

        # write arrays
        style[i, :] = mix.astype(np.float32)
        m = idxs.shape[0]
        idx_mat[i, :m] = idxs.astype(np.int32)
        w_mat[i, :m]   = w.astype(np.float32)
        if emb_mat is not None:
            emb_mat[i, :m, :] = Vecs.astype(np.float32)

        # write flattened pairs
        for rnk, (j, s, ww, pth) in enumerate(zip(idxs.tolist(), sims.tolist(), w.tolist(), paths), 1):
            flat.write(json.dumps({
                "query_id": i,
                "rank": rnk,
                "bank": used,
                "idx": int(j),
                "sim": float(s),
                "weight": float(ww),
                "path": pth
            }) + "\n")

    flat.close()
    # flush memmaps
    del style, idx_mat, w_mat
    if emb_mat is not None:
        del emb_mat

    if mism_bank:
        print(f"[WARN] {mism_bank} rows requested AUX but no aux_emb provided; fell back to PRIMARY.")

    # Optional sanity: compare with your precomputed agg_emb.npy (supports raw memmap)
    if args.agg_ref and Path(args.agg_ref).exists():
        # N = #queries, D = embedding dim (from primary bank)
        ref = _robust_load_matrix(args.agg_ref, N, D)
        new = _robust_load_matrix(args.out_dir / "style_topM.npy", N, D)

        # L2-normalize then cosine per row
        def _norm(X):
            n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            return (X / n).astype(np.float32)
        ref_n = _norm(ref)
        new_n = _norm(new)
        cos = (ref_n * new_n).sum(axis=1)

        mean = float(cos.mean())
        p10  = float(np.percentile(cos, 10))
        p90  = float(np.percentile(cos, 90))
        print(f"Compare to agg_ref: cos mean={mean:.4f}  p10={p10:.4f}  p90={p90:.4f}")
        if mean < 0.98:
            print("[NOTE] Different top_m/tau or bank choices can explain a small delta; "
                  "verify flags match your precompute_neighbors settings.")

    print(f"[DONE] style_topM.npy, neighbor_idx_topM.npy, neighbor_w_topM.npy, pairs_flat.jsonl in {args.out_dir}")

if __name__ == "__main__":
    main()