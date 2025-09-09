#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_fast_pass.py  —  quick, parallel pre-screen for large audio dirs (with progress bar)

Computes per-file:
  - rms_db, peak_db, zcr

Outputs:
  - audit_fast.json  (list of dicts with metrics)
  - suspects.txt     (files that warrant deeper audit_plus.py)

Heuristics (tunable via CLI):
  - silent if rms_db <= --rms_silent
  - suspect if:
       (rms_db > --rms_noise and zcr >= --zcr_noise)          # loud + noisy
    or (rms_db between [-40,-12] and zcr >= --zcr_mid)         # mid-level hiss
    or (peak_db >= -1.0 and rms_db <= -30)                     # near-clipped spikes but low average

Usage:
  python audit_fast_pass.py --audio_dir /path/to/flacs --workers 24
"""
import argparse, json, math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import soundfile as sf

def _get_tqdm(total, desc):
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, smoothing=0.05, miniters=1, leave=True)
    except Exception:
        class _NoTQDM:
            def update(self, n=1): pass
            def close(self): pass
        return _NoTQDM()

def rms_db(x):
    x = x.astype(np.float32)
    if x.size == 0: return -240.0
    r = float(np.sqrt(np.mean(x**2)))
    if r <= 1e-12: return -240.0
    return 20.0 * math.log10(r + 1e-12)

def peak_db(x):
    p = float(np.max(np.abs(x))) if x.size else 0.0
    if p <= 1e-12: return -240.0
    return 20.0 * math.log10(p + 1e-12)

def zero_cross_rate(x):
    x = x.astype(np.float32)
    return (np.abs(np.diff(np.signbit(x)))).mean()

def analyze(path):
    try:
        x, sr = sf.read(path, always_2d=False)
        if getattr(x, "ndim", 1) > 1:
            x = x.mean(axis=1)
        rdb = rms_db(x)
        pdb = peak_db(x)
        zcr = zero_cross_rate(x)
        return {"file": str(path), "rms": rdb, "peak": pdb, "zcr": zcr}
    except Exception as e:
        return {"file": str(path), "error": str(e)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True, type=Path)
    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--suspects_out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    # thresholds
    ap.add_argument("--rms_silent", type=float, default=-60.0)
    ap.add_argument("--rms_noise",  type=float, default=-12.0)
    ap.add_argument("--zcr_noise",  type=float, default=0.18)
    ap.add_argument("--zcr_mid",    type=float, default=0.20)
    args = ap.parse_args()

    audio_dir = args.audio_dir
    files = sorted(list(audio_dir.glob("*.flac")) + list(audio_dir.glob("*.wav")))
    if not files:
        raise SystemExit(f"No audio files found under {audio_dir}")

    out = []
    suspects = []

    pbar = _get_tqdm(total=len(files), desc="Fast audit")
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(analyze, f): f for f in files}
            for fut in as_completed(futures):
                row = fut.result()
                out.append(row)
                pbar.update(1)
    finally:
        pbar.close()

    # Decide suspects
    for row in out:
        if "error" in row:
            suspects.append(row["file"])
            continue
        rms = row["rms"]; zcr = row["zcr"]; peak = row["peak"]
        if rms <= args.rms_silent:
            continue  # silent → skip; not a "suspect" for hiss, but you likely exclude anyway
        loud_noisy = (rms > args.rms_noise and zcr >= args.zcr_noise)
        mid_hiss   = (-40.0 <= rms <= -12.0 and zcr >= args.zcr_mid)
        spike_low  = (peak >= -1.0 and rms <= -30.0)
        if loud_noisy or mid_hiss or spike_low:
            suspects.append(row["file"])

    # Write results
    out_json = args.out_json or (audio_dir / "audit_fast.json")
    suspects_out = args.suspects_out or (audio_dir / "suspects.txt")
    out_json.write_text(json.dumps(out, indent=2))
    suspects_out.write_text("\n".join(sorted(suspects)))
    print(f"Wrote {out_json} with {len(out)} rows")
    print(f"Wrote {suspects_out} with {len(suspects)} suspects")

if __name__ == "__main__":
    main()