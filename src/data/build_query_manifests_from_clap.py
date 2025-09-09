#!/usr/bin/env python3
# build_query_manifests_from_clap.py
# Create per-dataset and union query manifests from clap.json files.
# Usage:
#   python build_query_manifests_from_clap.py \
#     --clap maestro=/scratch/rc01492/data/processed/maestro/clap_embed/clap.json \
#     --clap urmp=/scratch/rc01492/data/processed/urmp/clap_embed/clap.json \
#     --out_dir /scratch/rc01492/data/processed/retrieval_cache \
#     --dedup
#
# Output files:
#   /scratch/.../retrieval_cache/maestro_queries.jsonl
#   /scratch/.../retrieval_cache/urmp_queries.jsonl
#   /scratch/.../retrieval_cache/union_queries.jsonl

from __future__ import annotations
import argparse, json
from pathlib import Path

def load_file_paths(clap_json: Path) -> list[str]:
    meta = json.loads(clap_json.read_text())
    fps = meta.get("file_paths") or meta.get("paths") or []
    if not isinstance(fps, (list, tuple)) or not fps:
        raise SystemExit(f"{clap_json}: no 'file_paths' found")
    return list(map(str, fps))

def write_jsonl(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def parse_name_eq_path(s: str) -> tuple[str, Path]:
    if "=" not in s:
        raise argparse.ArgumentTypeError("Use the form name=/abs/path/to/clap.json")
    name, p = s.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Dataset name is empty")
    return name, Path(p)

def main():
    ap = argparse.ArgumentParser(description="Build query manifests from clap.json")
    ap.add_argument("--clap", action="append", required=True,
                    help="Repeat: name=/abs/path/to/clap.json (e.g., maestro=/.../clap.json)")
    ap.add_argument("--out_dir", type=Path, required=True, help="Where to write manifests")
    ap.add_argument("--dedup", action="store_true", help="Remove duplicate paths in the union (keeps first occurrence)")
    args = ap.parse_args()

    # Parse inputs
    datasets: list[tuple[str, Path]] = [parse_name_eq_path(x) for x in args.clap]

    # Build per-dataset manifests
    per_ds_rows: dict[str, list[dict]] = {}
    total = 0
    for ds_name, clap_json in datasets:
        paths = load_file_paths(clap_json)
        rows = [{"id": total + i, "path": p, "dataset": ds_name, "dataset_idx": i}
                for i, p in enumerate(paths)]
        per_ds_rows[ds_name] = rows
        write_jsonl(rows, args.out_dir / f"{ds_name}_queries.jsonl")
        print(f"[{ds_name}] wrote {len(rows):,} rows → {args.out_dir / f'{ds_name}_queries.jsonl'}")
        total += len(rows)

    # Build union (preserve dataset order; assign global ids in that same order)
    union_rows: list[dict] = []
    gid = 0
    for ds_name in [n for n,_ in datasets]:
        for r in per_ds_rows[ds_name]:
            r_union = dict(r)  # copy
            r_union["id"] = gid
            gid += 1
            union_rows.append(r_union)

    if args.dedup:
        seen = set()
        deduped = []
        for r in union_rows:
            p = r["path"]
            if p in seen:  # skip duplicates, keep first
                continue
            seen.add(p)
            deduped.append(r)
        print(f"[union] dedup: {len(union_rows):,} → {len(deduped):,}")
        union_rows = deduped

    out_union = args.out_dir / "union_queries.jsonl"
    write_jsonl(union_rows, out_union)
    print(f"[union] wrote {len(union_rows):,} rows → {out_union}")

if __name__ == "__main__":
    main()