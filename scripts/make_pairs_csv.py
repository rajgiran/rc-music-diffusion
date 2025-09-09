#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build pairs.csv (audio_path,text) from MAESTRO, URMP (AuMix WAVs), and FSD50K.

Now with:
- Explicit counts of indexed audio files per dataset.
- Clear diagnostics for found roots and example paths.
- Progress bars for long loops.

CLI stays the same.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv
import random

import pandas as pd
from tqdm.auto import tqdm

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

def is_audio(p: Path) -> bool:
    return p.suffix.lower() in AUDIO_EXTS

def write_pairs(rows: List[Tuple[str, str]], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio_path", "text"])
        for a, t in rows:
            w.writerow([a, t])

def take_first_n(rows, n: Optional[int]):
    if n is None: return list(rows)
    return list(rows)[: max(0, int(n))]

# -----------------------------
# MAESTRO
# -----------------------------
def _normalize_rel(s: str) -> str:
    return s.strip().lstrip("/\\").replace("\\", "/").lower()

def _index_audio_tree(root: Path) -> tuple[Dict[str, Path], Dict[str, List[Path]], List[Path]]:
    """
    rel_map: normalized relative path -> absolute
    base_map: basename -> [absolute paths] (collision-aware)
    all_files: list of absolute audio files (for diagnostics)
    """
    rel_map: Dict[str, Path] = {}
    base_map: Dict[str, List[Path]] = {}
    all_audio: List[Path] = []

    root = root.resolve()
    for ap in root.rglob("*"):
        if ap.is_file() and is_audio(ap):
            all_audio.append(ap)
            try:
                rel = ap.relative_to(root).as_posix().lower()
            except Exception:
                rel = ap.name.lower()
            rel_map[rel] = ap
            base_map.setdefault(ap.name.lower(), []).append(ap)
    return rel_map, base_map, all_audio

def harvest_maestro(maestro_root: Path, max_items: Optional[int] = None) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    if not maestro_root or not maestro_root.exists():
        print("[MAESTRO] root not found; skipping.")
        return rows

    csvs = sorted(maestro_root.rglob("maestro-*.csv"))
    if not csvs:
        print("[MAESTRO] No maestro-*.csv under:", maestro_root)
        return rows

    rel_map, base_map, all_audio = _index_audio_tree(maestro_root)
    print(f"[MAESTRO] indexed {len(all_audio)} audio files under {maestro_root}")
    if all_audio:
        print("  e.g.:", *(str(p.relative_to(maestro_root)) for p in all_audio[:3]), sep="\n       ")

    collected_total = 0
    misses: List[str] = []

    for csv_path in csvs:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[MAESTRO] failed to read {csv_path.name}: {e}")
            continue

        audio_col = next((c for c in ["audio_filename", "audio_path", "path", "filename"] if c in df.columns), None)
        title_col = next((c for c in ["canonical_title", "title", "piece_title"] if c in df.columns), None)
        comp_col  = next((c for c in ["canonical_composer", "composer"] if c in df.columns), None)

        if audio_col is None:
            print(f"[MAESTRO] missing audio filename column in {csv_path.name}. Columns: {list(df.columns)}")
            continue

        iter_df = df if max_items is None else df.head(max_items)
        collected_here = 0
        for _, r in tqdm(iter_df.iterrows(), total=len(iter_df), desc=f"[MAESTRO] {csv_path.name}", leave=False):
            rel = _normalize_rel(str(r[audio_col]))
            apath = rel_map.get(rel)
            if apath is None:
                # try with a prefixed folder name variant
                for pref in ["maestro-v3.0.0/", "maestro-v2.0.0/", ""]:
                    apath = rel_map.get(_normalize_rel(pref + rel))
                    if apath: break
            if apath is None:
                # fallback by basename, prefer same-year hits if present
                base = Path(rel).name.lower()
                cands = base_map.get(base)
                if cands:
                    yr = rel.split("/", 1)[0]
                    best = None
                    if yr.isdigit():
                        for c in cands:
                            if f"/{yr}/" in c.as_posix():
                                best = c; break
                    apath = best or cands[0]
            if apath is None:
                if len(misses) < 8:
                    misses.append(f"{csv_path.name}: miss '{r[audio_col]}'")
                continue

            title = str(r.get(title_col, "")).strip() if title_col else ""
            comp  = str(r.get(comp_col, "")).strip()  if comp_col  else ""
            if title and comp:
                text = f"solo piano performance: {title} by {comp}"
            elif title:
                text = f"solo piano performance: {title}"
            else:
                text = "solo piano performance"

            rows.append((str(apath), text))
            collected_here += 1

        print(f"[MAESTRO] {csv_path.name}: collected {collected_here}")
        collected_total += collected_here

    print(f"[MAESTRO] total collected {collected_total}")
    if collected_total == 0:
        if not all_audio:
            print("[MAESTRO] WARNING: no audio files found under the root. "
                  "Are the WAVs downloaded/extracted?")
        if misses:
            print("[MAESTRO] example misses (first few):")
            for m in misses[:5]:
                print("  ·", m)

    return take_first_n(rows, max_items)

# -----------------------------
# URMP (AuMix WAVs)
# -----------------------------
URMP_INSTR_MAP = {
    "vn": "violin", "va": "viola", "vc": "cello", "db": "double bass",
    "fl": "flute", "cl": "clarinet", "ob": "oboe", "bn": "bassoon",
    "tpt": "trumpet", "hn": "horn", "tbn": "trombone", "tba": "tuba",
    "sax": "saxophone", "gt": "guitar", "hp": "harp", "pn": "piano",
    "perc": "percussion", "timp": "timpani",
}

def urmp_prompt_from_name(name: str) -> str:
    base = Path(name).stem
    parts = base.split("_")
    codes = []
    for p in reversed(parts):
        if p.isdigit(): break
        if p.lower() in {"aumix", "mix", "video", "vid"}: break
        codes.append(p.lower())
    insts = [URMP_INSTR_MAP.get(c, c) for c in codes[::-1] if c]
    if not insts:
        return "classical chamber ensemble recording"
    if len(insts) == 1:
        return f"solo {insts[0]} performance in a chamber hall"
    if len(insts) == 2:
        return f"{insts[0]} and {insts[1]} duet in a chamber hall"
    return f"{', '.join(insts)} ensemble performance in a chamber hall"

def harvest_urmp(urmp_aumix_dir: Path, max_items: Optional[int] = None) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    if not urmp_aumix_dir or not urmp_aumix_dir.exists():
        print("[URMP] root not found; skipping.")
        return rows
    wavs = sorted([p.resolve() for p in urmp_aumix_dir.glob("*.wav") if p.is_file()])
    print(f"[URMP] found {len(wavs)} AuMix WAVs in {urmp_aumix_dir}")
    for p in tqdm(wavs, desc="[URMP] collect AuMix", leave=False):
        rows.append((str(p), urmp_prompt_from_name(p.name)))
    print(f"[URMP] collected {len(rows)}")
    return take_first_n(rows, max_items)

# -----------------------------
# FSD50K
# -----------------------------
def load_fsd_vocabulary(vocab_csv: Path) -> Dict[str, str]:
    txt = vocab_csv.read_text(encoding="utf-8", errors="ignore")
    head = txt.splitlines()[:3]
    has_header = any("display_name" in ln or ",mid" in ln or ";mid" in ln for ln in head)
    if has_header:
        df = pd.read_csv(vocab_csv)
        cols = {c.lower(): c for c in df.columns}
        dn = cols.get("display_name") or "display_name"
        md = cols.get("mid") or "mid"
        out = {}
        for _, r in df.iterrows():
            mid = str(r[md]).strip()
            name = str(r[dn]).strip().replace("_", " ")
            if mid:
                out[mid] = name
        return out
    else:
        df = pd.read_csv(vocab_csv, header=None, names=["idx", "display_name", "mid"])
        out = {}
        for _, r in df.iterrows():
            mid = str(r["mid"]).strip()
            name = str(r["display_name"]).strip().replace("_", " ")
            if mid:
                out[mid] = name
        return out

def find_fsd_meta(fsd_root: Path) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    fsd_root = fsd_root.resolve()
    vocab_csv = None
    for nm in ["vocabulary.csv", "ground_truth_vocabulary.csv", "class_labels_indices.csv"]:
        hits = list(fsd_root.rglob(nm))
        if hits:
            vocab_csv = hits[0]; break
    dev_csv = eval_csv = None
    for c in fsd_root.rglob("*.csv"):
        n = c.name.lower()
        if n == "dev.csv":  dev_csv = c
        if n == "eval.csv": eval_csv = c
    if dev_csv is None:
        for c in fsd_root.rglob("*dev*ground*truth*.csv"):
            dev_csv = c; break
    if eval_csv is None:
        for c in fsd_root.rglob("*eval*ground*truth*.csv"):
            eval_csv = c; break
    return vocab_csv, dev_csv, eval_csv

def find_audio_roots(fsd_root: Path, names: List[str]) -> List[Path]:
    roots: List[Path] = []
    for nm in names:
        roots += [d for d in fsd_root.rglob(nm) if d.is_dir()]
    return sorted(set(roots))

def _index_basename_map(roots: List[Path]) -> Dict[str, List[Path]]:
    m: Dict[str, List[Path]] = {}
    total = 0
    for rt in roots:
        pool = [p for p in rt.rglob("*") if p.is_file() and is_audio(p)]
        print(f"[FSD50K] indexing {len(pool)} files under {rt}")
        total += len(pool)
        for p in pool:
            m.setdefault(p.name.lower(), []).append(p.resolve())
    print(f"[FSD50K] total indexed audio files: {total}")
    return m

def harvest_fsd50k(fsd_root: Path, max_items: Optional[int] = None) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    if not fsd_root or not fsd_root.exists():
        print("[FSD50K] root not found; skipping.")
        return rows

    vocab_csv, dev_csv, eval_csv = find_fsd_meta(fsd_root)
    if dev_csv is None and eval_csv is None:
        print("[FSD50K] ground truth CSVs not found; skipping.")
        return rows
    vocab = load_fsd_vocabulary(vocab_csv) if vocab_csv else {}

    dev_roots = find_audio_roots(fsd_root, ["dev_audio", "FSD50K.dev_audio"]) or []
    eval_roots = find_audio_roots(fsd_root, ["eval_audio", "FSD50K.eval_audio"]) or []
    print(f"[FSD50K] dev roots:  {', '.join(str(p) for p in dev_roots) or '(none)'}")
    print(f"[FSD50K] eval roots: {', '.join(str(p) for p in eval_roots) or '(none)'}")

    base_map_dev = _index_basename_map(dev_roots) if dev_roots else {}
    base_map_eval = _index_basename_map(eval_roots) if eval_roots else {}

    target = max_items if max_items is not None else 10**9
    remain = target

    def collect_from_split(split_csv: Path, base_map: Dict[str, List[Path]], budget: int, tag: str) -> List[Tuple[str, str]]:
        if budget <= 0 or not split_csv:
            return []
        try:
            df = pd.read_csv(split_csv)
        except Exception as e:
            print(f"[FSD50K] failed to read {split_csv.name}: {e}")
            return []
        fname_col = next((c for c in ["fname", "filename", "file_name"] if c in df.columns), None)
        mids_col = "mids" if "mids" in df.columns else None
        labels_col = next((c for c in ["labels", "label", "classes"] if c in df.columns), None)
        if fname_col is None:
            print(f"[FSD50K] no filename column in {split_csv.name}. Columns: {list(df.columns)}")
            return []
        out: List[Tuple[str, str]] = []
        for _, r in tqdm(df.iterrows(), total=len(df), desc=f"[FSD50K] {tag}", leave=False):
            if len(out) >= budget: break
            fname = Path(str(r[fname_col])).name.lower()
            cands = base_map.get(fname)
            if not cands:
                continue
            apath = cands[0]
            names: List[str] = []
            if mids_col and pd.notna(r.get(mids_col, "")):
                mids = str(r[mids_col]).split(",")
                names = [vocab.get(m.strip(), "") for m in mids if m.strip()]
                names = [n for n in names if n]
            if not names and labels_col and pd.notna(r.get(labels_col, "")):
                names = [str(r[labels_col]).replace("_", " ")]
            text = "environmental sound" if not names else "environmental sound: " + ", ".join(names[:3])
            out.append((str(apath), text))
        return out

    if dev_csv and remain > 0:
        got = collect_from_split(dev_csv, base_map_dev, remain, "dev")
        rows += got; remain -= len(got)
        print(f"[FSD50K] dev: +{len(got)} (remain {remain})")
    if eval_csv and remain > 0:
        got = collect_from_split(eval_csv, base_map_eval, remain, "eval")
        rows += got; remain -= len(got)
        print(f"[FSD50K] eval: +{len(got)} (remain {remain})")

    print(f"[FSD50K] collected {len(rows)}")
    return rows

# -----------------------------
# Orchestrator (CLI)
# -----------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser("Build audio/text training pairs")
    ap.add_argument("--maestro_root", type=Path, help="MAESTRO root (contains maestro-*.csv and year folders)")
    ap.add_argument("--urmp_root", type=Path, help="URMP AuMix WAVs folder")
    ap.add_argument("--fsd50k_root", type=Path, help="FSD50K root (with ground_truth + dev_audio/eval_audio)")
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--max_maestro", type=int, default=None)
    ap.add_argument("--max_urmp",   type=int, default=None)
    ap.add_argument("--max_fsd",    type=int, default=None)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    rows: List[Tuple[str, str]] = []
    if args.maestro_root:
        rows += harvest_maestro(args.maestro_root, args.max_maestro)
    if args.urmp_root:
        rows += harvest_urmp(args.urmp_root, args.max_urmp)
    if args.fsd50k_root:
        rows += harvest_fsd50k(args.fsd50k_root, args.max_fsd)

    # De-dupe & ensure existence
    uniq: Dict[str, str] = {}
    for a, t in rows:
        p = Path(a)
        if p.exists() and is_audio(p):
            uniq[str(p.resolve())] = t

    final_rows = list(uniq.items())
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(final_rows)

    print(f"[TOTAL] writing {len(final_rows)} rows → {args.out_csv}")
    write_pairs(final_rows, args.out_csv)

if __name__ == "__main__":
    main()