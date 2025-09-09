#!/usr/bin/env python3
"""
audit_plus.py — richer audio QC for MIDI renders (with progress bar)

Metrics per file:
  - rms_db, peak_db, zcr
  - spectral_flatness_mean, spectral_centroid_mean
  - hf_ratio (energy > 8 kHz / total)
  - floor_db (10th-percentile frame RMS as noise-floor proxy)

Classification (tunable via CLI):
  - silent if rms_db <= --rms_silent
  - noisy  if (hf_ratio >= --hf_ratio and spectral_flatness_mean >= --flatness)
             or (floor_db >= --floor_db_min)
"""
import argparse, json, math
from pathlib import Path
import numpy as np, soundfile as sf

def _get_tqdm(total, desc):
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, smoothing=0.05, miniters=1, leave=True)
    except Exception:
        class _NoTQDM:
            def update(self, n=1): pass
            def close(self): pass
        return _NoTQDM()

def frame_signal(x, frame_len=2048, hop=512):
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    n = 1 + max(0, (len(x) - frame_len) // hop)
    if n <= 0:
        n = 1
    idx = np.tile(np.arange(frame_len), (n,1)) + np.tile(np.arange(0, n*hop, hop), (frame_len,1)).T
    return x[idx]

def stft_mag(x, sr, frame_len=2048, hop=512):
    frames = frame_signal(x, frame_len, hop).astype(np.float32)
    frames *= np.hanning(frame_len).astype(np.float32)[None, :]
    spec = np.fft.rfft(frames, n=frame_len, axis=1)
    return np.abs(spec).astype(np.float32), np.fft.rfftfreq(frame_len, d=1.0/sr)

def spectral_centroid(mag, freqs):
    num = (mag * freqs[None, :]).sum(axis=1)
    den = (mag.sum(axis=1) + 1e-12)
    return num / den

def spectral_flatness(mag):
    gm = np.exp(np.mean(np.log(mag + 1e-12), axis=1))
    am = np.mean(mag + 1e-12, axis=1)
    return gm / am

def zcr(x): return (np.abs(np.diff(np.signbit(x.astype(np.float32))))).mean()

def rms_db(x):
    r = float(np.sqrt(np.mean(x.astype(np.float32)**2)))
    return -240.0 if r <= 1e-12 else 20.0*np.log10(r + 1e-12)

def peak_db(x):
    p = float(np.max(np.abs(x))) if x.size else 0.0
    return -240.0 if p <= 1e-12 else 20.0*np.log10(p + 1e-12)

def analyze_one(path, frame_len, hop, rms_silent, hf_ratio_thr, flatness_thr, floor_db_min):
    x, sr = sf.read(path, always_2d=False)
    if getattr(x, "ndim", 1) > 1: x = x.mean(axis=1)
    rdb, pdb, z = rms_db(x), peak_db(x), zcr(x)
    mag, freqs = stft_mag(x, sr, frame_len=frame_len, hop=hop)
    sc = spectral_centroid(mag, freqs).mean()
    sfm = spectral_flatness(mag).mean()
    hf = float((mag[:, freqs>=8000.0].sum()+1e-12)/(mag.sum()+1e-12))
    frame_rms_db = 20.0*np.log10(np.sqrt((mag**2).mean(axis=1))+1e-12)
    floor_db = float(np.percentile(frame_rms_db, 10.0))
    return {
        "file": str(path), "rms": float(rdb), "peak": float(pdb), "zcr": float(z),
        "spectral_centroid_mean": float(sc),
        "spectral_flatness_mean": float(sfm),
        "hf_ratio": float(hf), "floor_db": float(floor_db),
        "silent": bool(rdb <= rms_silent),
        "noisy":  bool((hf >= hf_ratio_thr and sfm >= flatness_thr) or (floor_db >= floor_db_min)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_dir", help="dir containing suspect FLAC/WAV files")
    ap.add_argument("--out", default="audit_plus.json")
    ap.add_argument("--rms_silent", type=float, default=-60.0)
    ap.add_argument("--hf_ratio", type=float, default=0.35)
    ap.add_argument("--flatness", type=float, default=0.40)
    ap.add_argument("--floor_db_min", type=float, default=-40.0)
    ap.add_argument("--frame_len", type=int, default=2048, help="STFT frame length")
    ap.add_argument("--hop", type=int, default=512, help="STFT hop size")
    args = ap.parse_args()

    d = Path(args.audio_dir)
    files = sorted(list(d.glob("*.flac")) + list(d.glob("*.wav")))
    if not files:
        raise SystemExit(f"No audio files found under {d}")

    pbar = _get_tqdm(total=len(files), desc="Deep audit")
    out = []
    try:
        for f in files:
            try:
                out.append(analyze_one(
                    f, args.frame_len, args.hop,
                    args.rms_silent, args.hf_ratio, args.flatness, args.floor_db_min
                ))
            except Exception as e:
                out.append({"file": str(f), "error": str(e)})
            pbar.update(1)
    finally:
        pbar.close()

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out} with {len(out)} entries.")

if __name__ == "__main__":
    main()