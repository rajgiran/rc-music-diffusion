#!/usr/bin/env python3
"""
Chunk audio listed in a JSONL manifest into fixed-length windows.

- Reads a CLEAN manifest JSONL where each line has at least {"path": "<rel/or/abs audio file>"}.
- Resolves relative paths against --audio_root.
- Outputs per-source folders with chunk files, e.g. out_dir/<stem>/<00000>.flac
- Supports overlap via --hop_secs (default: --secs, i.e., no overlap).
- Resamples to --sr, crops/pads last window according to --min_tail_ratio.
- Writes:
    out_dir/chunk_mapping.jsonl  ({"path": "<abs path to chunk>", "src_path": "<abs src>", "src_stem": "<stem>", "i": <index>})
    out_dir/chunk_stats.json     (summary JSON: counts, hours, errors)
"""

from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm


def read_jsonl(path: Path):
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)


def _resolve_audio_path(path_str: str, audio_root: Optional[Path]) -> Path:
    p = Path(path_str)
    if not p.is_absolute() and audio_root is not None:
        p = (audio_root / p).resolve()
    return p


def _load_mono(path: Path, sr: int) -> Tuple[np.ndarray, int]:
    """Load as float32 mono at (possibly different) native sr; resample to `sr` if needed."""
    x, srr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)  # mono
    if srr != sr:
        x = librosa.resample(x, orig_sr=srr, target_sr=sr)
        srr = sr
    return x.astype(np.float32, copy=False), srr


def _chunk_one(
    rec: Dict[str, Any],
    audio_root: Optional[Path],
    out_root: Path,
    sr: int,
    secs: float,
    hop_secs: float,
    min_tail_ratio: float,
    audio_format: str,
    subdir_by_stem: bool,
) -> Tuple[int, int]:
    """
    Returns (num_chunks_kept, err_flag)
    """
    src_path = _resolve_audio_path(rec["path"], audio_root)
    if not src_path.exists():
        return 0, 1

    try:
        y, _ = _load_mono(src_path, sr)
    except Exception:
        return 0, 1

    step = int(round(secs * sr))
    hop = int(round(hop_secs * sr))
    if hop <= 0:
        hop = step

    if len(y) < int(step * min_tail_ratio):
        return 0, 0  # too short

    stem = src_path.stem
    tgt_dir = (out_root / stem) if subdir_by_stem else out_root
    tgt_dir.mkdir(parents=True, exist_ok=True)

    i = 0
    kept = 0
    start = 0
    while start < len(y):
        end = start + step
        if end > len(y):
            # last window: keep only if long enough
            if (len(y) - start) < int(step * min_tail_ratio):
                break
            # pad the tail
            chunk = np.zeros(step, dtype=np.float32)
            chunk[: len(y) - start] = y[start:]
        else:
            chunk = y[start:end]

        out = tgt_dir / (f"{i:05d}.{audio_format}" if subdir_by_stem else f"{stem}__{i:05d}.{audio_format}")
        sf.write(out, chunk, sr, format=audio_format.upper())  # 'FLAC' or 'WAV'
        kept += 1
        i += 1
        start += hop

    return kept, 0


def main():
    ap = argparse.ArgumentParser(description="Chunk audio from a JSONL manifest into fixed-length windows.")
    ap.add_argument("--manifest", type=Path, required=True, help="Clean JSONL manifest with {'path': str} per line")
    ap.add_argument("--audio_root", type=Path, default=None, help="Root to resolve relative paths in manifest")
    ap.add_argument("--out_dir", type=Path, required=True, help="Destination directory for chunked audio")
    ap.add_argument("--sr", type=int, default=48000, help="Target sample rate (default: 48000)")
    ap.add_argument("--secs", type=float, default=10.0, help="Window length in seconds (default: 10)")
    ap.add_argument("--hop_secs", type=float, default=None, help="Hop between windows; default: same as --secs (no overlap)")
    ap.add_argument("--min_tail_ratio", type=float, default=0.50, help="Keep last window if >= ratio of full window (default: 0.50)")
    ap.add_argument("--audio_format", type=str, default="flac", choices=["flac", "wav"], help="Output format")
    ap.add_argument("--workers", type=int, default=16, help="Parallel processes")
    ap.add_argument("--max_files", type=int, default=None, help="Limit number of manifest rows (debug)")
    ap.add_argument("--flat", action="store_true", help="Do not create per-stem subfolders; write flat files with stem prefix")
    args = ap.parse_args()

    hop_secs = args.hop_secs if args.hop_secs is not None else args.secs

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(read_jsonl(args.manifest))
    if args.max_files is not None:
        rows = rows[: args.max_files]

    total = len(rows)
    if total == 0:
        raise SystemExit("Manifest is empty; nothing to do.")

    # process
    kept_total = 0
    err_total = 0
    mapping_path = args.out_dir / "chunk_mapping.jsonl"
    stats_path = args.out_dir / "chunk_stats.json"

    # we’ll append to mapping as we go, to avoid holding a giant list in RAM
    with mapping_path.open("w") as map_f:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {
                ex.submit(
                    _chunk_one,
                    rec,
                    args.audio_root,
                    args.out_dir,
                    int(args.sr),
                    float(args.secs),
                    float(hop_secs),
                    float(args.min_tail_ratio),
                    args.audio_format.lower(),
                    not args.flat,
                ): rec
                for rec in rows
            }
            for fut in tqdm(as_completed(futs), total=total, desc=f"Chunking {args.secs}s (hop {hop_secs}s)"):
                rec = futs[fut]
                try:
                    kept, err = fut.result()
                except Exception:
                    kept, err = 0, 1
                kept_total += kept
                err_total += err

                # write mapping entries for this source
                if kept > 0:
                    src_path = _resolve_audio_path(rec["path"], args.audio_root)
                    stem = src_path.stem
                    if args.flat:
                        for i in range(kept):
                            p = args.out_dir / f"{stem}__{i:05d}.{args.audio_format}"
                            map_f.write(json.dumps({"path": str(p), "src_path": str(src_path), "src_stem": stem, "i": i}) + "\n")
                    else:
                        for i in range(kept):
                            p = args.out_dir / stem / f"{i:05d}.{args.audio_format}"
                            map_f.write(json.dumps({"path": str(p), "src_path": str(src_path), "src_stem": stem, "i": i}) + "\n")

    # summarize
    # estimate hours from written files (each ~secs long, except tails; close enough)
    hours = kept_total * args.secs / 3600.0
    stats = dict(
        manifest=str(args.manifest),
        audio_root=str(args.audio_root) if args.audio_root else None,
        out_dir=str(args.out_dir),
        sr=int(args.sr),
        secs=float(args.secs),
        hop_secs=float(hop_secs),
        min_tail_ratio=float(args.min_tail_ratio),
        audio_format=args.audio_format.lower(),
        total_sources=int(total),
        sources_with_errors=int(err_total),
        total_chunks=int(kept_total),
        approx_hours=float(hours),
    )
    stats_path.write_text(json.dumps(stats, indent=2))
    print("\nSaved:")
    print("  •", mapping_path)
    print("  •", stats_path)


if __name__ == "__main__":
    # keep BLAS threads tame when forking
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()