#!/usr/bin/env python3
# compare_query_vs_style.py
# Compare per-query CLAP embedding (from MAESTRO/URMP banks) with your aggregated style
# vector (agg_emb.npy). Handles memmap-backed arrays and missing dataset_idx.

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

def _load_clap_bank(clap_json: Path, clap_npy: Path) -> tuple[list[str], np.ndarray]:
    meta = json.loads(clap_json.read_text())
    paths: List[str] = meta["file_paths"]
    X = np.load(clap_npy, mmap_mode="r").astype(np.float32)
    assert len(paths) == X.shape[0], f"paths ({len(paths)}) != emb rows ({X.shape[0]})"
    # L2 normalize
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return paths, X

def _normalize_rows(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n

def _robust_load_matrix(path: Path, N: int, D: int) -> np.ndarray:
    """Try np.load first; fall back to raw memmap with known shape (float32)."""
    try:
        arr = np.load(path, mmap_mode="r")
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2 and arr.shape == (N, D):
            return arr
    except Exception:
        pass
    return np.memmap(path, mode="r", dtype=np.float32, shape=(N, D))

def _build_index(paths: List[str]) -> tuple[Dict[str, int], Dict[str, List[int]]]:
    """Exact path → idx, and basename → [idx,...] for fallback."""
    path2idx: Dict[str, int] = {}
    base2idxs: Dict[str, List[int]] = {}
    for i, p in enumerate(paths):
        path2idx[p] = i
        base = Path(p).name
        base2idxs.setdefault(base, []).append(i)
    return path2idx, base2idxs

def _infer_dataset(rec: dict) -> str:
    if "dataset" in rec and rec["dataset"]:
        return rec["dataset"]
    p = rec.get("path", "")
    if "/maestro/" in p: return "maestro"
    if "/urmp/" in p: return "urmp"
    return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union_manifest", required=True, type=Path)
    ap.add_argument("--agg_npy",        required=True, type=Path)
    ap.add_argument("--maestro_clap_json", required=True, type=Path)
    ap.add_argument("--maestro_clap_npy",   required=True, type=Path)
    ap.add_argument("--urmp_clap_json", required=True, type=Path)
    ap.add_argument("--urmp_clap_npy",   required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load banks
    m_paths, M = _load_clap_bank(args.maestro_clap_json, args.maestro_clap_npy)
    u_paths, U = _load_clap_bank(args.urmp_clap_json, args.urmp_clap_npy)
    D = int(M.shape[1])  # 512 for CLAP

    m_path2idx, m_base2idxs = _build_index(m_paths)
    u_path2idx, u_base2idxs = _build_index(u_paths)

    # Load union manifest
    UNI = [json.loads(l) for l in args.union_manifest.open() if l.strip()]
    N = len(UNI)

    # Load aggregated style vectors (robustly)
    S = _robust_load_matrix(args.agg_npy, N, D)
    S = _normalize_rows(S)

    cos_vals: List[float] = []
    misses = 0
    basename_ambig = 0

    for i, rec in enumerate(UNI):
        path = rec.get("path") or rec.get("query_path")
        if not path:
            misses += 1
            cos_vals.append(0.0)
            continue

        ds = _infer_dataset(rec)
        ds_idx = rec.get("dataset_idx", None)

        # Get query embedding q
        q = None
        if ds == "maestro":
            if isinstance(ds_idx, int) and 0 <= ds_idx < M.shape[0]:
                q = M[ds_idx]
            else:
                # exact path match
                if path in m_path2idx:
                    q = M[m_path2idx[path]]
                else:
                    # basename fallback (only if unique)
                    base = Path(path).name
                    cand = m_base2idxs.get(base, [])
                    if len(cand) == 1:
                        q = M[cand[0]]
                    elif len(cand) > 1:
                        basename_ambig += 1
        elif ds == "urmp":
            if isinstance(ds_idx, int) and 0 <= ds_idx < U.shape[0]:
                q = U[ds_idx]
            else:
                if path in u_path2idx:
                    q = U[u_path2idx[path]]
                else:
                    base = Path(path).name
                    cand = u_base2idxs.get(base, [])
                    if len(cand) == 1:
                        q = U[cand[0]]
                    elif len(cand) > 1:
                        basename_ambig += 1
        else:
            # unknown dataset
            q = None

        if q is None:
            misses += 1
            cos_vals.append(0.0)
            continue

        s = S[i]  # style vector row aligns to manifest order
        cos_vals.append(float((q * s).sum()))

    cos = np.asarray(cos_vals, dtype=np.float32)

    # Stats
    mean = float(cos.mean())
    p10  = float(np.percentile(cos, 10))
    p50  = float(np.percentile(cos, 50))
    p90  = float(np.percentile(cos, 90))

    # Save text + json
    (args.out_dir / "query_vs_style.txt").write_text(
        f"mean={mean:.4f}  p10={p10:.4f}  p50={p50:.4f}  p90={p90:.4f}\n"
        f"misses={misses}  basename_ambiguous={basename_ambig}  N={N}\n"
    )
    (args.out_dir / "query_vs_style.json").write_text(json.dumps({
        "mean": mean, "p10": p10, "p50": p50, "p90": p90,
        "misses": int(misses),
        "basename_ambiguous": int(basename_ambig),
        "N": int(N)
    }, indent=2))

    # Plot
    plt.figure(figsize=(6,4), dpi=110)
    plt.hist(cos, bins=50)
    plt.title("cos(query_CLAP, style_vector)")
    plt.xlabel("cosine similarity")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(args.out_dir / "query_vs_style_hist.png")
    plt.close()

    print(f"mean cos={mean:.4f}  (misses={misses}, basename_ambiguous={basename_ambig})")
    print("→ wrote:",
          args.out_dir / "query_vs_style_hist.png",
          args.out_dir / "query_vs_style.txt")

if __name__ == "__main__":
    main()