#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prune_with_audits.py

Prune a curated JSONL manifest using QC outputs from audit_fast.py and audit_plus.py.

Inputs
------
- --manifest        JSONL with one object per line: {"id": <int>, "path": "/abs/or/rel.wav"}
- --fast_json       audit_fast.json produced on the *view* folder
- --plus_json       audit_plus.json produced on the *suspects_view* (optional)
- --suspects_txt    suspects.txt from audit_fast (optional; if provided, we will also drop suspects
                    that failed to appear in audit_plus.json because of read/link errors)

Output
------
- --out             pruned manifest JSONL (same schema as input)
- Also writes alongside: <out>.stats.json and (optional) <out>.dropped.jsonl when --write_drops

Decisions
---------
- Mark as SILENT if (fast["silent"] == True) OR (fast["rms"] <= --rms_silent).
- Mark as NOISY if (plus["noisy"] == True) OR (plus["silent"] == True)
                  OR (plus["floor_db"] >= --floor_db_min)
                  OR (plus has both hf_ratio & spectral_flatness_mean and they are >= thresholds).
- If --suspects_txt is given, *suspects that did not show up in plus* are dropped as unreadable.

Usage
-----
python prune_with_audits.py \
  --manifest   /scratch/.../musiccaps_instrumental_relaxed.jsonl \
  --fast_json  /scratch/.../qc_views/instrumental_relaxed/audit_fast.json \
  --plus_json  /scratch/.../qc_views/instrumental_relaxed/audit_plus_relaxed.json \
  --suspects_txt /scratch/.../qc_views/instrumental_relaxed/suspects.txt \
  --out        /scratch/.../musiccaps_instrumental_relaxed_clean.jsonl

Thresholds can be adjusted; defaults match prior guidance.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, Iterable, Set, List
from tqdm import tqdm

# -------- helpers --------

def _basename(p: str) -> str:
    """Return base filename; strip a leading "id__" prefix used in view links."""
    name = Path(p).name
    if "__" in name:
        # strip first id__ prefix only
        parts = name.split("__", 1)
        if len(parts) == 2 and parts[0].isdigit():
            return parts[1]
    return name


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_fast_bad(fast_json: Path, rms_silent: float) -> Set[str]:
    """Return basenames (original, without id__ prefix) considered silent from fast audit."""
    data = load_json(fast_json)
    bad = set()
    for row in data:
        fn = _basename(row.get("file", ""))
        rms = float(row.get("rms", -999.0))
        silent_flag = bool(row.get("silent", False))
        if silent_flag or (rms <= rms_silent):
            bad.add(fn)
    return bad


def load_plus_bad(plus_json: Path, hf_ratio: float, flatness: float, floor_db_min: float) -> Set[str]:
    """Return basenames marked noisy/silent by deep audit thresholds."""
    if not plus_json or not plus_json.exists():
        return set()
    data = load_json(plus_json)
    bad = set()
    for row in data:
        fn = _basename(row.get("file", ""))
        silent_flag = bool(row.get("silent", False))
        noisy_flag = bool(row.get("noisy", False))
        floor_db = float(row.get("floor_db", -300.0))
        hf = row.get("hf_ratio")
        flat = row.get("spectral_flatness_mean")
        rule_plus = False
        if (hf is not None) and (flat is not None):
            try:
                hf = float(hf); flat = float(flat)
                rule_plus = (hf >= hf_ratio and flat >= flatness)
            except Exception:
                rule_plus = False
        if silent_flag or noisy_flag or (floor_db >= floor_db_min) or rule_plus:
            bad.add(fn)
    return bad


def load_missing_deep(suspects_txt: Path, plus_json: Path) -> Set[str]:
    """Return basenames that are in suspects.txt but absent from plus JSON (likely unreadable).
    NOTE: suspects.txt lines are id-prefixed (e.g., 0000123__27.wav), so we must
    strip the prefix to compare fairly with deep-audit entries.
    """
    if not suspects_txt or not Path(suspects_txt).exists() or not plus_json or not plus_json.exists():
        return set()
    # strip CRLF and the id__ prefix
    sus = [_basename(l.strip().replace("", "")) for l in open(suspects_txt, "r", encoding="utf-8") if l.strip()]
    deep = [_basename(r.get("file", "")) for r in load_json(plus_json)]
    return set(sus) - set(deep)


# -------- main --------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--fast_json", required=True, type=Path)
    ap.add_argument("--plus_json", type=Path, default=None)
    ap.add_argument("--suspects_txt", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)

    # thresholds
    ap.add_argument("--rms_silent", type=float, default=-60.0)
    ap.add_argument("--hf_ratio", type=float, default=0.35)
    ap.add_argument("--flatness", type=float, default=0.40)
    ap.add_argument("--floor_db_min", type=float, default=-40.0)

    # extras
    ap.add_argument("--write_drops", action="store_true", help="Write <out>.dropped.jsonl with reasons")
    ap.add_argument("--verify_audio", action="store_true", help="Warn if manifest paths are missing on disk")

    args = ap.parse_args()

    # build bad sets
    base_bad = load_fast_bad(args.fast_json, args.rms_silent)
    plus_bad = load_plus_bad(args.plus_json, args.hf_ratio, args.flatness, args.floor_db_min)
    miss_deep = load_missing_deep(args.suspects_txt, args.plus_json)

    all_bad = set(base_bad) | set(plus_bad) | set(miss_deep)

    kept = []
    dropped = []

    total = 0
    with args.manifest.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Prune manifest"):
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            bn = Path(rec.get("path", "")).name
            if bn in all_bad:
                if args.write_drops:
                    reason = []
                    if bn in base_bad: reason.append("fast_silent")
                    if bn in plus_bad: reason.append("plus_noisy_or_floor_or_silent")
                    if bn in miss_deep: reason.append("missing_deep")
                    rec2 = {**rec, "drop_reason": ",".join(reason)}
                    dropped.append(rec2)
                continue
            if args.verify_audio:
                p = Path(rec.get("path", ""))
                if not p.is_absolute():
                    # if relative, just warn (we don't know audio_root here)
                    pass
                else:
                    if not p.exists():
                        print(f"[WARN] missing on disk: {p}")
            kept.append(rec)

    # write outputs
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    stats = {
        "input_manifest": str(args.manifest),
        "fast_json": str(args.fast_json),
        "plus_json": str(args.plus_json) if args.plus_json else None,
        "suspects_txt": str(args.suspects_txt) if args.suspects_txt else None,
        "thresholds": {
            "rms_silent": args.rms_silent,
            "hf_ratio": args.hf_ratio,
            "flatness": args.flatness,
            "floor_db_min": args.floor_db_min,
        },
        "total_rows": total,
        "dropped": len(all_bad),
        "kept": len(kept),
        "dropped_breakdown": {
            "fast_silent": len(base_bad),
            "plus_noisy_or_floor_or_silent": len(plus_bad),
            "missing_deep": len(miss_deep),
        },
        "out": str(args.out.resolve()),
    }
    stats_path = args.out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2))

    if args.write_drops and dropped:
        drops_path = args.out.with_suffix(".dropped.jsonl")
        with drops_path.open("w", encoding="utf-8") as f:
            for r in dropped:
                f.write(json.dumps(r) + "\n")

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()