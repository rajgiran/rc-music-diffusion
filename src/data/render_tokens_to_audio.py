#!/usr/bin/env python3
"""
Render tokenized MIDI to audio files for CLAP embedding extraction.

Adds (kept from your working version + new extras):
  1) Suppresses known PrettyMIDI warnings.
  2) Auto-fix for corrupt MIDIs with absurd largest tick (via mido re-write).
  3) Dense-chunk detection -> route to Fluidsynth CLI with higher polyphony/buffers.
  4) Process-shared rate limiting for CLI renders (Semaphore).
  5) MIDI sanitization for dense material (clip overlaps; drop ultra-short).
  6) Optional velocity floor pre-gate; post-render RMS dBFS gate.
  7) Fixed-length crop/pad to --target_seconds.
  8) Save outputs as FLAC to reduce disk usage (lossless).
  9) Retry mode: re-render failed chunks from **rendering_metadata.json** with ultra-safe CLI settings
     (use --retry_from_meta and optional --retry_ultra_safe).
 10) Robust loader for chunks.pkl vs. raw_tokens.pkl (no section removed; only additive).
 11) --skip_existing to resume safely without re-writing existing FLACs.

Dependencies:
  - pretty_midi, soundfile, numpy, miditok, tqdm
  - Optional for rescue path: mido
  - Optional for dense rendering path: fluidsynth CLI in PATH
  - Optional for resampling: librosa
"""

from __future__ import annotations
import argparse
import pickle
import json
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from multiprocessing.synchronize import Semaphore
import soundfile as sf
import pretty_midi
from miditok import REMI, TokSequence
import logging
import tempfile
import os
import subprocess
import warnings
import math

# ─────────── constants ───────────
DEFAULT_SF = Path("/scratch/rc01492/data/assets/soundfonts/FluidR3_GM.sf2")

# Suppress noisy runtime warnings from pretty_midi
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pretty_midi")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# Global variables for multiprocessing
_global_tokenizer: Optional[REMI] = None
_global_soundfont: Optional[Path] = None
_global_sample_rate: Optional[int] = None
_global_target_len: Optional[int] = None
_global_dense_poly_thresh: Optional[int] = None

# CLI / fluidsynth tuning globals
_global_cli_polyphony: Optional[int] = None
_global_cli_periods: Optional[int] = None
_global_cli_period_size: Optional[int] = None
_global_cli_timeout: Optional[int] = None
_global_cli_max_retries: Optional[int] = None
_global_cli_ultra_polyphony: Optional[int] = None
_global_cli_ultra_periods: Optional[int] = None
_global_cli_ultra_period_size: Optional[int] = None
_global_disable_cli: Optional[bool] = None
_global_disable_cli_reverb: Optional[bool] = None
_global_disable_cli_chorus: Optional[bool] = None
_global_cli_gain: Optional[float] = None

# Rate limiting (process-shared) for CLI renders
_global_cli_sem: Optional[Semaphore] = None

# Gating / sanitization
_global_velocity_floor: Optional[int] = None
_global_skip_quiet_db: Optional[float] = None
_global_init_cc_levels: Optional[bool] = None


# ────────────────────────── helpers ──────────────────────────
def _fix_len(audio: np.ndarray, target_len: int | None) -> np.ndarray:
    if target_len is None:
        return audio.astype(np.float32)
    n = audio.shape[0]
    if n == target_len:
        return audio.astype(np.float32)
    if n > target_len:
        return audio[:target_len].astype(np.float32)
    out = np.zeros(int(target_len), dtype=np.float32)
    out[:n] = audio.astype(np.float32)
    return out


def _safe_load_pretty_midi(mid_path: Path) -> Optional[pretty_midi.PrettyMIDI]:
    try:
        return pretty_midi.PrettyMIDI(str(mid_path))
    except Exception as e:
        msg = str(e)
        if "largest tick" in msg or "negative time" in msg or "exceeds" in msg:
            try:
                import mido
                mid = mido.MidiFile(str(mid_path))
                tpq_old = max(1, int(getattr(mid, "ticks_per_beat", 480)))
                tpq_new = 480 if tpq_old > 480 else tpq_old
                if tpq_new != tpq_old:
                    scale = tpq_new / float(tpq_old)
                    for track in mid.tracks:
                        for msg2 in track:
                            if hasattr(msg2, "time"):
                                msg2.time = int(round(msg2.time * scale))
                    mid.ticks_per_beat = tpq_new
                    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tf:
                        rescue_path = Path(tf.name)
                    mid.save(str(rescue_path))
                    try:
                        pm = pretty_midi.PrettyMIDI(str(rescue_path))
                        os.unlink(rescue_path)
                        return pm
                    except Exception:
                        os.unlink(rescue_path)
                        return None
                else:
                    return None
            except Exception:
                return None
        else:
            return None


def _max_concurrent_notes(pm: pretty_midi.PrettyMIDI) -> int:
    events = []
    for inst in pm.instruments:
        for n in inst.notes:
            events.append((n.start, 1))
            events.append((n.end, -1))
    if not events:
        return 0
    events.sort(key=lambda x: (x[0], -x[1]))
    cur = 0
    peak = 0
    for _, delta in events:
        cur += delta
        if cur > peak:
            peak = cur
    return peak


def _sanitize_dense_midi(pm: pretty_midi.PrettyMIDI) -> None:
    min_dur = 0.010
    eps = 0.010
    for inst in pm.instruments:
        by_pitch = {}
        for n in inst.notes:
            by_pitch.setdefault(n.pitch, []).append(n)
        new_notes = []
        for _, arr in by_pitch.items():
            arr.sort(key=lambda n: n.start)
            for i, n in enumerate(arr):
                if i + 1 < len(arr):
                    nxt = arr[i + 1]
                    if n.end > nxt.start - eps:
                        n.end = max(n.start + min_dur, nxt.start - eps)
                if (n.end - n.start) >= min_dur:
                    new_notes.append(n)
        inst.notes = new_notes


def _estimate_avg_velocity(pm: pretty_midi.PrettyMIDI) -> float:
    vals = [n.velocity for inst in pm.instruments for n in inst.notes]
    return float(np.mean(vals)) if vals else 0.0


def _rms_dbfs(x: np.ndarray, eps: float = 1e-9) -> float:
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))
    if rms <= eps:
        return -120.0
    return 20.0 * math.log10(rms + 1e-12)


def _render_with_prettymidi(pm: pretty_midi.PrettyMIDI, sr: int, sf2: Path) -> Optional[np.ndarray]:
    try:
        if sf2.exists():
            audio = pm.fluidsynth(fs=sr, sf2_path=str(sf2))
        else:
            audio = pm.synthesize(fs=sr)
        if audio.size == 0:
            return None
        mx = float(np.max(np.abs(audio)))
        if mx == 0.0:
            return None
        return (audio / mx * 0.95).astype(np.float32)
    except Exception as e:
        logger.warning(f"pretty_midi render failed: {str(e).splitlines()[0]}")
        return None


def _render_with_fluidsynth_cli(mid_path: Path, sr: int, sf2: Path,
                                polyphony: int, periods: int, period_size: int,
                                disable_reverb: bool, disable_chorus: bool, gain: float,
                                timeout: int, max_retries: int) -> Optional[np.ndarray]:
    tries = max(1, int(max_retries))
    for attempt in range(tries):
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
                out_wav = Path(wf.name)
            cmd = [
                "fluidsynth",
                "-ni",
                str(sf2),
                str(mid_path),
                "-F", str(out_wav),
                "-r", str(sr),
                "-o", f"synth.polyphony={polyphony}",
                "-o", f"audio.periods={periods}",
                "-o", f"audio.period-size={period_size}",
                "-o", f"synth.gain={gain}",
            ]
            if disable_reverb:
                cmd += ["-o", "synth.reverb.active=0"]
            if disable_chorus:
                cmd += ["-o", "synth.chorus.active=0"]

            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=timeout)
            audio, sr_read = sf.read(out_wav)
            os.unlink(out_wav)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr_read != sr:
                try:
                    import librosa
                except ImportError:
                    raise RuntimeError("librosa is required for resampling but is not installed")
                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr_read, target_sr=sr)
            mx = float(np.max(np.abs(audio))) if audio.size else 0.0
            if audio.size == 0 or mx == 0.0:
                return None
            return (audio / mx * 0.95).astype(np.float32)
        except subprocess.TimeoutExpired:
            logger.warning("fluidsynth CLI timed out (attempt %d/%d)", attempt + 1, tries)
            continue
        except Exception as e:
            logger.warning(f"fluidsynth CLI render failed (attempt {attempt+1}/{tries}): {str(e).splitlines()[0]}")
            continue
    return None


# ───────────────────── multiprocessing init ─────────────────────
def init_worker(vocab_path: Path,
                soundfont_path: Path,
                sample_rate: int,
                target_seconds: float | None,
                dense_poly_thresh: int,
                cli_polyphony: int,
                cli_periods: int,
                cli_period_size: int,
                cli_timeout: int,
                cli_rate_limit: int,
                cli_ultra_polyphony: int,
                cli_ultra_periods: int,
                cli_ultra_period_size: int,
                cli_max_retries: int,
                disable_cli: bool,
                disable_cli_reverb: bool,
                disable_cli_chorus: bool,
                cli_gain: float,
                velocity_floor: Optional[int],
                skip_quiet_db: Optional[float],
                init_cc_levels: bool,
                cli_sem: Semaphore):
    """Initialize tokenizer and globals in each worker."""
    global _global_tokenizer, _global_soundfont, _global_sample_rate, _global_target_len
    global _global_dense_poly_thresh
    global _global_cli_polyphony, _global_cli_periods, _global_cli_period_size
    global _global_cli_timeout, _global_cli_max_retries
    global _global_cli_ultra_polyphony, _global_cli_ultra_periods, _global_cli_ultra_period_size
    global _global_cli_sem
    global _global_disable_cli, _global_disable_cli_reverb, _global_disable_cli_chorus, _global_cli_gain
    global _global_velocity_floor, _global_skip_quiet_db, _global_init_cc_levels

    _global_tokenizer = REMI(params=vocab_path)
    _global_soundfont = soundfont_path
    _global_sample_rate = int(sample_rate)
    _global_target_len = int(round(target_seconds * _global_sample_rate)) if target_seconds else None

    _global_dense_poly_thresh = int(dense_poly_thresh)

    _global_cli_polyphony = int(cli_polyphony)
    _global_cli_periods = int(cli_periods)
    _global_cli_period_size = int(cli_period_size)
    _global_cli_timeout = int(cli_timeout)
    _global_cli_max_retries = int(cli_max_retries)
    _global_cli_ultra_polyphony = int(cli_ultra_polyphony)
    _global_cli_ultra_periods = int(cli_ultra_periods)
    _global_cli_ultra_period_size = int(cli_ultra_period_size)
    _global_disable_cli = bool(disable_cli)
    _global_disable_cli_reverb = bool(disable_cli_reverb)
    _global_disable_cli_chorus = bool(disable_cli_chorus)
    _global_cli_gain = float(cli_gain)

    _global_velocity_floor = velocity_floor if velocity_floor is None else int(velocity_floor)
    _global_skip_quiet_db = skip_quiet_db if skip_quiet_db is None else float(skip_quiet_db)
    _global_init_cc_levels = bool(init_cc_levels)

    _global_cli_sem = cli_sem  # process-shared semaphore


# ─────────────────────────── main worker ───────────────────────────
def process_chunk_worker(args: Tuple[int, List[int], Path]) -> Tuple[int, bool, int]:
    """Process a single chunk in worker process."""
    chunk_idx, tokens, output_path = args

    if not tokens:
        return chunk_idx, False, 0

    tmp_path = None
    try:
        # Decode tokens → MIDI
        score = _global_tokenizer.decode([TokSequence(ids=tokens)])
        if isinstance(score, list):
            score = score[0]

        # Optional: set initial CC levels
        if _global_init_cc_levels and hasattr(score, "instruments"):
            for inst in score.instruments:
                inst.control_changes.append(pretty_midi.ControlChange(7, 100, time=0.0))
                inst.control_changes.append(pretty_midi.ControlChange(11, 100, time=0.0))

        # Write temp MIDI
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        score.dump_midi(tmp_path)

        # Load with PrettyMIDI (rescue if needed)
        pm = _safe_load_pretty_midi(tmp_path)
        if pm is None:
            try:
                pm = pretty_midi.PrettyMIDI(str(tmp_path))
            except Exception:
                return chunk_idx, False, len(tokens)

        # If no notes, skip
        total_notes = sum(len(inst.notes) for inst in pm.instruments)
        if total_notes == 0:
            return chunk_idx, False, len(tokens)

        # Velocity floor
        if _global_velocity_floor is not None and _global_velocity_floor > 0:
            avg_vel = _estimate_avg_velocity(pm)
            if avg_vel < _global_velocity_floor:
                return chunk_idx, False, len(tokens)

        # Dense detection + sanitization
        peak_poly_before = _max_concurrent_notes(pm)
        dense = peak_poly_before >= _global_dense_poly_thresh
        if dense:
            _sanitize_dense_midi(pm)

        # Recompute density post-sanitize
        peak_poly = _max_concurrent_notes(pm)
        ultra_dense = peak_poly >= max(_global_dense_poly_thresh * 2,
                                       _global_cli_ultra_polyphony // 3)

        # Render
        audio = None
        if not dense and not _global_disable_cli:
            audio = _render_with_prettymidi(pm, _global_sample_rate, _global_soundfont or DEFAULT_SF)
            if audio is None:
                if _global_cli_sem is not None:
                    with _global_cli_sem:
                        audio = _render_with_fluidsynth_cli(
                            mid_path=tmp_path, sr=_global_sample_rate,
                            sf2=_global_soundfont or DEFAULT_SF,
                            polyphony=_global_cli_polyphony,
                            periods=_global_cli_periods,
                            period_size=_global_cli_period_size,
                            disable_reverb=_global_disable_cli_reverb,
                            disable_chorus=_global_disable_cli_chorus,
                            gain=_global_cli_gain,
                            timeout=_global_cli_timeout,
                            max_retries=_global_cli_max_retries,
                        )
                else:
                    audio = _render_with_fluidsynth_cli(
                        mid_path=tmp_path, sr=_global_sample_rate,
                        sf2=_global_soundfont or DEFAULT_SF,
                        polyphony=_global_cli_polyphony,
                        periods=_global_cli_periods,
                        period_size=_global_cli_period_size,
                        disable_reverb=_global_disable_cli_reverb,
                        disable_chorus=_global_disable_cli_chorus,
                        gain=_global_cli_gain,
                        timeout=_global_cli_timeout,
                        max_retries=_global_cli_max_retries,
                    )
        else:
            if not _global_disable_cli:
                poly = _global_cli_ultra_polyphony if ultra_dense else _global_cli_polyphony
                per  = _global_cli_ultra_periods    if ultra_dense else _global_cli_periods
                psz  = _global_cli_ultra_period_size if ultra_dense else _global_cli_period_size
                if _global_cli_sem is not None:
                    with _global_cli_sem:
                        audio = _render_with_fluidsynth_cli(
                            mid_path=tmp_path, sr=_global_sample_rate,
                            sf2=_global_soundfont or DEFAULT_SF,
                            polyphony=poly, periods=per, period_size=psz,
                            disable_reverb=_global_disable_cli_reverb,
                            disable_chorus=_global_disable_cli_chorus,
                            gain=_global_cli_gain, timeout=_global_cli_timeout,
                            max_retries=_global_cli_max_retries,
                        )
                else:
                    audio = _render_with_fluidsynth_cli(
                        mid_path=tmp_path, sr=_global_sample_rate,
                        sf2=_global_soundfont or DEFAULT_SF,
                        polyphony=poly, periods=per, period_size=psz,
                        disable_reverb=_global_disable_cli_reverb,
                        disable_chorus=_global_disable_cli_chorus,
                        gain=_global_cli_gain, timeout=_global_cli_timeout,
                        max_retries=_global_cli_max_retries,
                    )
            if audio is None:
                audio = _render_with_prettymidi(pm, _global_sample_rate, _global_soundfont or DEFAULT_SF)

        if audio is None:
            return chunk_idx, False, len(tokens)

        # Skip near silent
        if _global_skip_quiet_db is not None:
            db = _rms_dbfs(audio)
            if db < _global_skip_quiet_db:
                return chunk_idx, False, len(tokens)

        # Fixed length
        audio = _fix_len(audio, _global_target_len)

        # Save as FLAC
        sf.write(output_path, audio, _global_sample_rate, format="FLAC")
        return chunk_idx, True, len(tokens)

    except Exception as e:
        logger.error(f"Failed to process chunk {chunk_idx}: {e}")
        return chunk_idx, False, len(tokens)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ───────────────────────── dataset runner ─────────────────────────
def _load_token_sequences(tokens_path: Path, use_raw_tokens: bool) -> List[List[int]]:
    """Robustly load tokens from chunks.pkl or raw_tokens.pkl.
    Keeps backward compatibility; does NOT remove any existing sections.
    """
    with open(tokens_path, "rb") as f:
        data = pickle.load(f)

    # Typical keys we might see
    possible_keys = [
        "tokens",      # list[list[int]] of chunks
        "chunks",      # alias
        "raw_tokens",  # list[list[int]] or list[int]
    ]

    key_found = None
    for k in possible_keys:
        if k in data:
            key_found = k
            break

    if key_found is None:
        # if the pickled file itself is a list of lists
        if isinstance(data, list):
            if len(data) and isinstance(data[0], list):
                return data
        raise KeyError(f"No expected token keys found in {tokens_path}")

    seqs = data[key_found]

    # If user explicitly asked for raw tokens, but raw is a single long list, wrap it
    if use_raw_tokens and isinstance(seqs, list) and seqs and isinstance(seqs[0], int):
        return [seqs]  # single sequence

    # Ensure we output list[list[int]]
    if isinstance(seqs, list) and seqs and isinstance(seqs[0], list):
        return seqs

    # Fallback: empty or unknown shape
    if isinstance(seqs, list) and not seqs:
        return []

    raise TypeError(f"Unsupported token structure in {tokens_path}: type={type(seqs)}")


def render_dataset(
    tokens_path: Path,
    metadata_path: Path,
    vocab_path: Path,
    soundfont_path: Path,
    output_dir: Path,
    max_chunks: int = None,
    num_workers: int = 8,
    chunk_mode: bool = True,
    sample_rate: int = 48000,
    target_seconds: float | None = None,
    dense_poly_thresh: int = 128,
    cli_rate_limit: int = 1,
    cli_polyphony: int = 512,
    cli_periods: int = 16,
    cli_period_size: int = 4096,
    cli_ultra_polyphony: int = 768,
    cli_ultra_periods: int = 24,
    cli_ultra_period_size: int = 4096,
    cli_timeout: int = 120,
    cli_max_retries: int = 2,
    disable_cli: bool = False,
    disable_cli_reverb: bool = True,
    disable_cli_chorus: bool = True,
    cli_gain: float = 0.5,
    velocity_floor: Optional[int] = 8,
    skip_quiet_db: Optional[float] = -45.0,
    init_cc_levels: bool = True,
    retry_indices: Optional[list] = None,
    skip_existing: bool = False,
):
    """Render chunks from dataset (or a provided retry list)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # (kept original logging, now using robust loader)
    logger.info(f"Loading tokens from {tokens_path}")
    all_chunks = _load_token_sequences(tokens_path, use_raw_tokens=not chunk_mode)

    logger.info(f"Loaded {len(all_chunks)} sequences")

    file_mapping = {}
    if metadata_path and metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                file_mapping = metadata.get("file_mapping", {})
        except Exception:
            pass

    # Build task list
    if retry_indices is not None:
        logger.info(f"Retry mode: rendering {len(retry_indices)} failed chunks only")
        raw_tasks = [(idx, all_chunks[idx], output_dir / f"chunk_{idx:06d}.flac") for idx in retry_indices if idx < len(all_chunks)]
    else:
        if max_chunks is not None:
            all_chunks = all_chunks[:max_chunks]
            logger.info(f"Processing first {max_chunks} chunks")
        raw_tasks = [(idx, chunk, output_dir / f"chunk_{idx:06d}.flac") for idx, chunk in enumerate(all_chunks)]

    # Optionally skip existing
    if skip_existing:
        tasks = [(i, t, p) for (i, t, p) in raw_tasks if not p.exists()]
        skipped = len(raw_tasks) - len(tasks)
        if skipped:
            logger.info(f"Skip-existing active: {skipped} files already present")
    else:
        tasks = raw_tasks

    cli_sem = mp.Semaphore(max(1, int(cli_rate_limit)))

    logger.info(
        f"Rendering {len(tasks)} chunks with {num_workers} workers at {sample_rate} Hz"
        + (f", target {target_seconds:.2f}s" if target_seconds else "")
    )

    successful = 0
    failed = []
    chunk_lengths = []

    with Pool(
        num_workers,
        initializer=init_worker,
        initargs=(
            vocab_path, soundfont_path, sample_rate, target_seconds, dense_poly_thresh,
            cli_polyphony, cli_periods, cli_period_size, cli_timeout, cli_rate_limit,
            cli_ultra_polyphony, cli_ultra_periods, cli_ultra_period_size, cli_max_retries,
            disable_cli, disable_cli_reverb, disable_cli_chorus, cli_gain,
            velocity_floor, skip_quiet_db, init_cc_levels, cli_sem
        ),
    ) as pool:
        with tqdm(total=len(tasks), desc="Rendering") as pbar:
            for idx, success, length in pool.imap_unordered(process_chunk_worker, tasks, chunksize=10):
                if success:
                    successful += 1
                    chunk_lengths.append(length)
                else:
                    failed.append(idx)
                pbar.update(1)

    render_metadata = {
        "source_tokens": str(tokens_path),
        "total_chunks": len(tasks),
        "rendered_chunks": successful,
        "failed_chunks": len(failed),
        "failed_indices": failed,
        "avg_chunk_length": float(np.mean(chunk_lengths)) if chunk_lengths else 0.0,
        "file_mapping": file_mapping,
        "sample_rate": sample_rate,
        "target_seconds": target_seconds,
        "dense_poly_thresh": dense_poly_thresh,
        "encoding": "FLAC",
    }

    meta_out = output_dir / "rendering_metadata.json"
    with open(meta_out, "w") as f:
        json.dump(render_metadata, f, indent=2)

    logger.info(f"Rendering complete: {successful}/{len(tasks)} successful")
    logger.info(f"Metadata saved to {meta_out}")

    return successful, failed


# ──────────────────────────────── CLI ────────────────────────────────
def _apply_ultra_safe_overrides(args: argparse.Namespace) -> None:
    """In-place override of args for an ultra-safe retry rendering.
    Keeps original arguments otherwise; does NOT remove user sections.
    """
    # Ensure CLI path is used (not disabled), bump timeouts/retries/buffers
    args.disable_cli = False
    # Very high safety margins
    if args.cli_polyphony < 1024:
        args.cli_polyphony = 1024
    if args.cli_ultra_polyphony < 1536:
        args.cli_ultra_polyphony = 1536
    if args.cli_periods < 32:
        args.cli_periods = 32
    if args.cli_ultra_periods < 48:
        args.cli_ultra_periods = 48
    if args.cli_period_size < 4096:
        args.cli_period_size = 4096
    if args.cli_ultra_period_size < 8192:
        args.cli_ultra_period_size = 8192
    if args.cli_timeout < 300:
        args.cli_timeout = 300
    if args.cli_max_retries < 4:
        args.cli_max_retries = 4
    # Disable reverb/chorus for determinism/stability
    args.disable_cli_reverb = True
    args.disable_cli_chorus = True
    # Keep modest gain
    if args.cli_gain > 0.7:
        args.cli_gain = 0.7


def main():
    p = argparse.ArgumentParser(description="Render tokenized MIDI to audio")
    p.add_argument("--tokens_path", type=Path, required=True,
                   help="Path to chunks.pkl or raw_tokens.pkl")
    p.add_argument("--metadata_path", type=Path,
                   help="Path to metadata.json (optional)")
    p.add_argument("--vocab_path", type=Path, required=True,
                   help="Path to vocabulary JSON")
    p.add_argument("--soundfont", type=Path, default=DEFAULT_SF,
                   help=f"GM SoundFont (default: {DEFAULT_SF})")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Output directory for audio files")
    p.add_argument("--max_chunks", type=int, default=None,
                   help="Maximum number of chunks to process")
    p.add_argument("--workers", type=int, default=min(8, cpu_count()),
                   help="Number of parallel workers")
    p.add_argument("--use_raw_tokens", action="store_true",
                   help="Use raw_tokens.pkl instead of chunks.pkl (robust loader also auto-detects)")
    p.add_argument("--sample_rate", type=int, default=48000,
                   help="Sample rate for audio rendering (default: 48000)")
    p.add_argument("--target_seconds", type=float, default=8.0,
                   help="Crop/pad each render to exactly this duration (default: 8.0)")
    # Dense-chunk & CLI rendering controls
    p.add_argument("--dense_poly_thresh", type=int, default=64,
                   help="If max concurrent notes ≥ this, use CLI fluidsynth")
    p.add_argument("--cli_rate_limit", type=int, default=1,
                   help="Max concurrent CLI fluidsynth renders across workers")
    p.add_argument("--cli_polyphony", type=int, default=512,
                   help="Fluidsynth synth.polyphony for dense/CLI renders")
    p.add_argument("--cli_periods", type=int, default=16,
                   help="Fluidsynth audio.periods")
    p.add_argument("--cli_period_size", type=int, default=4096,
                   help="Fluidsynth audio.period-size")
    p.add_argument("--cli_ultra_polyphony", type=int, default=768,
                   help="Fluidsynth polyphony for ultra-dense renders")
    p.add_argument("--cli_ultra_periods", type=int, default=24,
                   help="Fluidsynth periods for ultra-dense")
    p.add_argument("--cli_ultra_period_size", type=int, default=4096,
                   help="Fluidsynth period-size for ultra-dense")
    p.add_argument("--cli_timeout", type=int, default=120,
                   help="Fluidsynth CLI timeout per render (seconds)")
    p.add_argument("--cli_max_retries", type=int, default=2,
                   help="Retries for CLI render on failure/timeout")
    p.add_argument("--disable_cli", action="store_true",
                   help="Force fully in-process rendering (no CLI)")
    p.add_argument("--disable_cli_reverb", action="store_true",
                   help="Turn off FluidSynth reverb")
    p.add_argument("--disable_cli_chorus", action="store_true",
                   help="Turn off FluidSynth chorus")
    p.add_argument("--cli_gain", type=float, default=0.5,
                   help="FluidSynth global gain (0..10)")
    # Gates / init
    p.add_argument("--velocity_floor", type=int, default=8,
                   help="Skip chunks whose avg note velocity < this")
    p.add_argument("--skip_quiet_db", type=float, default=-45.0,
                   help="Drop renders whose RMS level is below this dBFS")
    p.add_argument("--init_cc_levels", action="store_true",
                   help="Inject default CC7/CC11 at t=0")
    # Resume / retry helpers
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip rendering chunks whose FLAC already exists in output_dir")
    p.add_argument("--retry_from_meta", type=Path,
                   help="Path to a prior rendering_metadata.json to re-render its failed_indices")
    p.add_argument("--retry_ultra_safe", action="store_true",
                   help="When used with --retry_from_meta, apply ultra-safe CLI overrides")

    args = p.parse_args()

    # Check soundfont
    if not args.soundfont.exists():
        logger.warning(f"Soundfont not found at {args.soundfont}, will use sine wave synthesis")

    # Auto-detect metadata path
    if args.metadata_path is None:
        args.metadata_path = args.tokens_path.parent / "metadata.json"

    retry_indices = None
    if args.retry_from_meta is not None:
        try:
            with open(args.retry_from_meta, "r") as f:
                prev_meta = json.load(f)
            retry_indices = prev_meta.get("failed_indices") or []
            logger.info(f"Loaded {len(retry_indices)} failed indices from {args.retry_from_meta}")
        except Exception as e:
            logger.error(f"Failed to load retry metadata from {args.retry_from_meta}: {e}")
            retry_indices = []
        if args.retry_ultra_safe:
            _apply_ultra_safe_overrides(args)
            logger.info("Applied ultra-safe overrides for retry mode")

    render_dataset(
        tokens_path=args.tokens_path,
        metadata_path=args.metadata_path,
        vocab_path=args.vocab_path,
        soundfont_path=args.soundfont,
        output_dir=args.output_dir,
        max_chunks=args.max_chunks,
        num_workers=args.workers,
        chunk_mode=not args.use_raw_tokens,
        sample_rate=args.sample_rate,
        target_seconds=args.target_seconds,
        dense_poly_thresh=args.dense_poly_thresh,
        cli_rate_limit=args.cli_rate_limit,
        cli_polyphony=args.cli_polyphony,
        cli_periods=args.cli_periods,
        cli_period_size=args.cli_period_size,
        cli_ultra_polyphony=args.cli_ultra_polyphony,
        cli_ultra_periods=args.cli_ultra_periods,
        cli_ultra_period_size=args.cli_ultra_period_size,
        cli_timeout=args.cli_timeout,
        cli_max_retries=args.cli_max_retries,
        disable_cli=args.disable_cli,
        disable_cli_reverb=args.disable_cli_reverb,
        disable_cli_chorus=args.disable_cli_chorus,
        cli_gain=args.cli_gain,
        velocity_floor=args.velocity_floor,
        skip_quiet_db=args.skip_quiet_db,
        init_cc_levels=args.init_cc_levels,
        retry_indices=retry_indices,
        skip_existing=args.skip_existing,
    )

if __name__ == "__main__":
    if os.name == "nt":
        mp.set_start_method("spawn")
    main()