#!/usr/bin/env python3
# Robust FSD50K curator with flexible CSV→audio matching
from __future__ import annotations
import argparse, csv, json, os, re
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm

AUDIO_EXTS = [".wav", ".flac", ".mp3", ".ogg", ".m4a"]

def parse_labels(cell: str | None) -> List[str]:
    if not cell: return []
    cell = cell.strip().strip('"').strip("'")
    parts = re.split(r"[;,]", cell)
    return [p.strip().lower() for p in parts if p.strip()]

def ok_len(path: Path, min_s: float) -> bool:
    if min_s <= 0: return True
    try:
        import soundfile as sf
        info = sf.info(str(path))
        dur = float(info.frames) / max(1.0, float(info.samplerate or 0))
        return dur >= min_s
    except Exception:
        # if we can't read header quickly, don't block
        return True

def index_audio_recursive(root: Path):
    """
    Build two maps:
      - by_name: '391277.wav' -> [Path,...]
      - by_stem: '391277'     -> [Path,...]  (only if stem is unique)
    """
    by_name: Dict[str, List[Path]] = defaultdict(list)
    by_stem: Dict[str, List[Path]] = defaultdict(list)
    total = 0
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            total += 1
            by_name[p.name].append(p.resolve())
            estem = p.stem
            # only use numeric-looking stems for stem index; safer
            if estem.isdigit():
                by_stem[estem].append(p.resolve())
    return by_name, by_stem, total

def choose_path(cands: List[Path], split_hint: str | None) -> Path | None:
    if not cands: return None
    if split_hint:
        split_hint = split_hint.lower()
        for c in cands:
            if split_hint in str(c).lower():
                return c
    return cands[0]

def resolve_fname(fname: str, by_name, by_stem, split_hint: str | None) -> Path | None:
    """
    Try multiple variants:
      1) raw fname
      2) basename(fname)
      3) basename with common audio extensions if missing an extension
      4) numeric stem (e.g., '391277')
    """
    raw = fname.strip()
    base = Path(raw).name  # drops any split folder like 'FSD50K.dev_audio/'
    # 1–2: direct basename match
    if base in by_name:
        return choose_path(by_name[base], split_hint)
    # 3: if no extension, try common extensions
    if Path(base).suffix == "":
        for ext in AUDIO_EXTS:
            cand = base + ext
            if cand in by_name:
                return choose_path(by_name[cand], split_hint)
    # 4: numeric stem match
    stem = Path(base).stem
    if stem in by_stem and len(by_stem[stem]) > 0:
        return choose_path(by_stem[stem], split_hint)
    return None

def curate(args):
    # load config
    cfg = json.loads(Path(args.config).read_text())
    POS, VOC, NEG_S, NEG_R = map(set, (cfg["pos"], cfg["vocal"], cfg["neg_strict"], cfg["neg_relaxed"]))

    def is_music_ok(tags, neg): return any(t in POS for t in tags) and not any(t in neg for t in tags)
    def has_vocals(tags):       return any(t in VOC for t in tags)

    # index audio once
    audio_root = Path(args.audio_root)
    print(f"Indexing audio under {audio_root} (recursive)…")
    by_name, by_stem, total_files = index_audio_recursive(audio_root)
    print(f"Indexed {total_files:,} files; {len(by_name):,} unique basenames; {len(by_stem):,} numeric stems")

    # read CSV rows
    rows = []
    for csv_path in args.csv:
        split = "dev" if "dev" in csv_path.lower() else ("eval" if "eval" in csv_path.lower() else None)
        with open(csv_path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                fname = row.get("fname") or row.get("filename") or row.get("clip_name")
                labs  = parse_labels(row.get("labels") or row.get("label"))
                if not fname: 
                    continue
                rows.append((fname, labs, split))

    # buckets
    buckets = {
        "instrumental_strict": [], "with_vocals_strict": [],
        "instrumental_relaxed": [], "with_vocals_relaxed": []
    }
    id_counters = {k: 0 for k in buckets}

    total = len(rows)
    missing = 0
    resolved = 0
    first_20_misses = []

    # curate
    for fname, tags, split in tqdm(rows, desc="Curate"):
        path = resolve_fname(fname, by_name, by_stem, split)
        if path is None:
            missing += 1
            if len(first_20_misses) < 20:
                first_20_misses.append(fname)
            continue

        resolved += 1
        # strict
        if is_music_ok(tags, NEG_S) and ok_len(path, args.min_secs_strict):
            key = "with_vocals_strict" if has_vocals(tags) else "instrumental_strict"
            buckets[key].append({"id": id_counters[key], "path": str(path)})
            id_counters[key] += 1
        # relaxed
        if is_music_ok(tags, NEG_R) and ok_len(path, args.min_secs_relaxed):
            key = "with_vocals_relaxed" if has_vocals(tags) else "instrumental_relaxed"
            buckets[key].append({"id": id_counters[key], "path": str(path)})
            id_counters[key] += 1

    # write outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, lst in buckets.items():
        out = out_dir / f"{name}.jsonl"
        with out.open("w") as f:
            for r in lst: f.write(json.dumps(r) + "\n")
        print(f"{name:22s}: {len(lst):6d} -> {out}")

    stats = {
        "total_csv_rows": total,
        "resolved_files": resolved,
        "missing_files": missing,
        "kept": {k: len(v) for k, v in buckets.items()},
        "mins": {"strict": args.min_secs_strict, "relaxed": args.min_secs_relaxed},
        "audio_root": str(audio_root),
    }
    (out_dir/"fsd50k_curation_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    if first_20_misses:
        print("\nExample CSV names that didn’t resolve (first 20):")
        for s in first_20_misses:
            print("  -", s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True)
    ap.add_argument("--audio_root", type=Path, required=True, help="Parent dir that contains audio (scan recursively).")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--min_secs_strict", type=float, default=6.0)
    ap.add_argument("--min_secs_relaxed", type=float, default=5.0)
    args = ap.parse_args()
    curate(args)

if __name__ == "__main__":
    main()