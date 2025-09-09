#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a FAISS index from one or more embedding banks.

Bank config entry supports two schemas (choose one):
A) Path-aligned mapping  [RECOMMENDED for your artifacts]
   {
     "name": "fsd50k_primary",
     "emb_npy": "/.../clap.npy",
     "mapping_jsonl": "/.../clap_mapping.jsonl"   # rows align 1:1 with emb_npy
   }

B) ID-based mapping
   {
     "name": "some_bank",
     "emb_npy": "/.../clap.npy",
     "ids_npy": "/.../ids.npy",
     "manifest_jsonl": "/.../manifest.jsonl"      # JSONL with {"id": int, "path": str}
   }

Outputs in --out_dir:
  - index.faiss
  - mapping.jsonl  (one row per vector; fields: idx, bank, bank_row, path, id?)
  - stats.json
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import faiss

def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows

def _ensure_unit_norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def _mapping_from_ids(ids_npy: Path, manifest_jsonl: Path) -> List[dict]:
    """Return per-row dicts with 'id' and 'path' resolved via manifest."""
    ids = np.load(ids_npy)
    id2path: Dict[int, str] = {}
    for rec in _read_jsonl(manifest_jsonl):
        if "id" in rec:
            id2path[int(rec["id"])] = rec["path"]
    out = []
    for i, rid in enumerate(ids.tolist()):
        rid = int(rid)
        out.append({"id": rid, "path": id2path.get(rid)})
    return out

def _mapping_from_paths(mapping_jsonl: Path) -> List[dict]:
    """Accepts rows with {"path": ...} or {"id": ..., "path": ...}."""
    rows = _read_jsonl(mapping_jsonl)
    out = []
    for r in rows:
        out.append({"id": int(r["id"]) if "id" in r else None, "path": r.get("path")})
    return out

def _stack_banks(cfg: dict, renorm: bool, metric: str) -> Tuple[np.ndarray, List[dict]]:
    all_X = []
    mapping_rows: List[dict] = []
    idx_cursor = 0

    for bank in cfg["banks"]:
        name = bank["name"]
        emb = np.load(bank["emb_npy"], mmap_mode="r").astype(np.float32)
        D = emb.shape[1]

        # choose mapping source
        if "mapping_jsonl" in bank:
            map_rows = _mapping_from_paths(Path(bank["mapping_jsonl"]))
            assert len(map_rows) == emb.shape[0], f"{name}: mapping rows != emb rows"
        elif "ids_npy" in bank and "manifest_jsonl" in bank:
            map_rows = _mapping_from_ids(Path(bank["ids_npy"]), Path(bank["manifest_jsonl"]))
            assert len(map_rows) == emb.shape[0], f"{name}: ids rows != emb rows"
        else:
            raise SystemExit(f"{name}: provide either mapping_jsonl OR (ids_npy + manifest_jsonl).")

        # normalize for cosine/IP or if forced
        X = emb
        if renorm or metric.lower() == "ip":
            X = _ensure_unit_norm(X)

        # accumulate
        all_X.append(X)
        for r in range(X.shape[0]):
            mr = {
                "idx": idx_cursor + r,
                "bank": name,
                "bank_row": r,
                "path": map_rows[r].get("path"),
            }
            if map_rows[r].get("id") is not None:
                mr["id"] = int(map_rows[r]["id"])
            mapping_rows.append(mr)

        print(f"[{name}] OK  rows={emb.shape[0]} dim={D}")
        idx_cursor += X.shape[0]

    Xcat = np.concatenate(all_X, axis=0).astype(np.float32) if all_X else np.zeros((0, 0), np.float32)
    return Xcat, mapping_rows

def _build_index(X: np.ndarray, index_type: str, metric: str,
                 ivf_nlist: int, pq_m: int, pq_bits: int) -> faiss.Index:
    d = X.shape[1]
    metric_kind = faiss.METRIC_INNER_PRODUCT if metric.lower() == "ip" else faiss.METRIC_L2

    if index_type == "flat":
        index = faiss.IndexFlatIP(d) if metric_kind == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)
        index.add(X)
        return index

    if index_type == "ivfpq":
        quantizer = faiss.IndexFlatIP(d) if metric_kind == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, ivf_nlist, pq_m, pq_bits)
        # train on subset
        train_sz = min(100_000, X.shape[0])
        faiss.random_seed(123)
        perm = np.random.permutation(X.shape[0])[:train_sz]
        index.train(X[perm])
        index.add(X)
        index.nprobe = max(1, min(16, ivf_nlist // 64))
        return index

    raise SystemExit("--index_type must be 'flat' or 'ivfpq'")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--index_type", type=str, default="flat", choices=["flat", "ivfpq"])
    ap.add_argument("--metric", type=str, default="ip", choices=["ip", "l2"])
    ap.add_argument("--renorm", action="store_true", help="Force L2 normalization before indexing")
    ap.add_argument("--ivf_nlist", type=int, default=4096)
    ap.add_argument("--pq_m", type=int, default=64)
    ap.add_argument("--pq_bits", type=int, default=8)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    X, mapping_rows = _stack_banks(cfg, renorm=args.renorm, metric=args.metric)
    if X.size == 0:
        raise SystemExit("No embeddings provided.")
    print(f"[COMBINED] rows={X.shape[0]} dim={X.shape[1]}")

    index = _build_index(X, args.index_type, args.metric, args.ivf_nlist, args.pq_m, args.pq_bits)

    # save index + mapping + stats
    faiss.write_index(index, str(args.out_dir / "index.faiss"))
    with (args.out_dir / "mapping.jsonl").open("w") as f:
        for r in mapping_rows:
            f.write(json.dumps(r) + "\n")

    per_bank = {}
    for r in mapping_rows:
        per_bank[r["bank"]] = per_bank.get(r["bank"], 0) + 1

    stats = {
        "rows": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "index_type": args.index_type,
        "metric": args.metric,
        "renorm": bool(args.renorm),
        "per_bank": per_bank,
    }
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[DONE] {args.out_dir/'index.faiss'}  mapping.jsonl  stats.json")

if __name__ == "__main__":
    main()