#!/usr/bin/env python3
"""
Improved MIDI tokenizer with metadata tracking and better error handling.
Preserves all original functionality while adding research-friendly features.
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import sys
import os
import logging
import logging.handlers
import warnings
import signal
from hashlib import md5
from itertools import chain
from pathlib import Path
from typing import List, Sequence, Tuple, Dict, Any
from multiprocessing import Pool, cpu_count
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from miditok import REMI, TokenizerConfig, TokSequence
from tqdm import tqdm

# ─────────── constants ───────────
MAX_LEN, MIN_LEN = 2048, 256
NUM_VELOC, NUM_TEMPO = 127, 128
DEFAULT_SF = Path("/scratch/rc01492/assets/soundfonts/FluidR3_GM.sf2")

# ─────────── helpers ───────────
def setup_logging(log_dir: Path, dataset_name: str = "tokenise") -> logging.Logger:
    """Enhanced logging setup with dataset-specific log files."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(f"tokenise_{dataset_name}")
    log.setLevel(logging.INFO)
    
    # Clear existing handlers
    log.handlers.clear()
    
    # Console handler with custom formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler with detailed formatter
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / f"tokenise_{dataset_name}.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    
    log.addHandler(console_handler)
    log.addHandler(file_handler)
    
    return log


def digest(p: Path) -> str:
    """Get file hash for reproducibility."""
    return md5(p.read_bytes()).hexdigest()[:8]


def load_or_create_tokeniser(vocab_path: Path, log: logging.Logger) -> REMI:
    """Load existing or create new tokenizer."""
    if vocab_path.exists():
        tok = REMI(params=vocab_path)
        log.info("✓ Using existing vocabulary: %d tokens (hash: %s)",
                 tok.vocab_size, digest(vocab_path))
        return tok
    
    cfg = TokenizerConfig(
        tokenizer="REMI",
        num_velocities=NUM_VELOC,
        special_tokens=["PAD", "BOS", "EOS", "MASK"],
        num_tempos=NUM_TEMPO,
        use_tempos=True, 
        use_time_signatures=True,
        use_programs=False,  # Set to True for multi-instrument datasets
        use_chords=True, 
        use_rests=True,
        use_sustain_pedals=False,
        additional_params={
            "max_bar_embedding": 1024,
            "use_bar_end_tokens": False,
            "add_trailing_bars": False,
        },
    )
    tok = REMI(cfg)
    tok.save(vocab_path)
    log.info("✓ Created new vocabulary: %d tokens → %s", tok.vocab_size, vocab_path.name)
    return tok


def bar_chunks(seq: Sequence[int], bar_id: int) -> List[List[int]]:
    """Split sequence into chunks at bar boundaries."""
    out, i = [], 0
    while i < len(seq):
        j = min(i + MAX_LEN, len(seq))
        cut = next((k + 1 for k in range(j - 1, i, -1) if seq[k] == bar_id), None)
        if cut is None or cut - i < MIN_LEN:
            cut = j
        out.append(list(seq[i:cut]))
        i = cut
    return out


def robust_decode(tok: REMI, ids: List[int]) -> Any:
    """Safely decode token IDs back to MIDI."""
    seq = [TokSequence(ids=ids)]
    out = tok.decode(seq)
    return out[0] if isinstance(out, list) else out


def is_valid_midi(path: Path) -> bool:
    """Check if MIDI file is valid and readable."""
    import pretty_midi
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="pretty_midi")
    try:
        pretty_midi.PrettyMIDI(str(path))
        return True
    except Exception:
        return False


# ─────────── improved worker ───────────
def _tokenise_one_improved(args: Tuple[Path, str, str, Path]) -> Dict[str, Any]:
    """Enhanced tokenization worker that returns metadata."""
    midi_path, tok_json, bar_token, base_dir = args
    
    # Get relative path for better tracking
    try:
        rel_path = midi_path.relative_to(base_dir)
    except:
        rel_path = midi_path
    
    result = {
        "file": str(rel_path),
        "success": False,
        "tokens": [],
        "error": None,
        "hash": None,
    }
    
    try:
        # Get file hash
        result["hash"] = md5(midi_path.read_bytes()).hexdigest()
        
        # Tokenize (same as original)
        tok = REMI(params=Path(tok_json))
        seqs = [s.ids for s in tok(midi_path)]
        
        result["tokens"] = seqs
        result["success"] = True
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


# ─────────── diagnostics (preserved exactly) ───────────
def create_visuals(chunks: List[List[int]], tok: REMI, out_dir: Path, log: logging.Logger):
    """Create diagnostic visualizations - unchanged from original."""
    log.info("✓ Creating diagnostic visualizations...")
    id2tok = {v: k for k, v in tok.vocab.items()}
    all_ids = list(chain(*chunks))
    counts = collections.Counter(all_ids)
    lengths = np.array([len(c) for c in chunks])

    cats: dict[str, list[int]] = collections.defaultdict(list)
    for tid, c in counts.items():
        cats[id2tok.get(tid, "Other").split("_")[0]].append(c)

    fig, ax = plt.subplots(2, 3, figsize=(18, 10), dpi=110)

    ax[0, 0].hist(lengths, bins=60, color="#5dade2", edgecolor="k")
    mu = lengths.mean()
    ax[0, 0].axvline(mu, c="r", ls="--", label=f"μ ≈ {mu:.0f}")
    ax[0, 0].set(title="Chunk length", xlabel="tokens", ylabel="freq")
    ax[0, 0].legend(loc="upper right")

    pie_vals = np.array(list(cats.values()), dtype=object)
    pie_labels = np.array(list(cats.keys()))
    vals = np.array([sum(v) for v in pie_vals], dtype=float)
    big = vals / vals.sum() > .03
    explode = .06 * big[big]
    ax[0, 1].pie(vals[big], labels=pie_labels[big], autopct="%1.1f%%",
                 explode=explode, pctdistance=.8, startangle=90)
    ax[0, 1].set_title("Token‑type share")
    small_labels = [f"{l}: {v/vals.sum()*100:.1f} %" for l, v, b in zip(pie_labels, vals, big) if not b]
    if small_labels:
        ax[0, 1].legend(small_labels, loc="center left",
                        bbox_to_anchor=(1, .5), fontsize=8, title="< 3 % each")

    top = counts.most_common(20)
    names = [id2tok.get(t, f"id_{t}") for t, _ in top]
    vals = [v for _, v in top]
    ax[0, 2].barh(range(len(vals))[::-1], vals[::-1], color="#ec7063")
    ax[0, 2].set_yticks(range(len(vals))[::-1])
    ax[0, 2].set_yticklabels(names[::-1], fontsize=7)
    ax[0, 2].margins(y=0.15)
    ax[0, 2].set(title="Top‑20 tokens", xlabel="count")

    use = np.array(list(counts.values()))
    side = int(np.ceil(np.sqrt(use.size)))
    heat = np.pad(use, (0, side**2 - use.size)).reshape(side, side)
    im = ax[1, 0].imshow(heat + 1, cmap="YlOrRd",
                         norm=LogNorm(vmin=1, vmax=heat.max()+1))
    ax[1, 0].set_title(f"Used vocab {use.size}/{tok.vocab_size}")
    cbar = fig.colorbar(im, ax=ax[1, 0], fraction=.046, pad=.04, format="%d")
    cbar.set_ticks([1, 10, 1e2, 1e3, 1e4, 1e5])

    sorted_vals = np.sort(use)[::-1]
    cum = np.cumsum(sorted_vals) / sorted_vals.sum() * 100
    ax[1, 1].plot(np.arange(1, len(cum)+1), cum)
    ax[1, 1].axhline(80, c="r", ls="--")
    ax[1, 1].axhline(95, c="orange", ls="--")
    ax[1, 1].grid(alpha=.3)
    ax[1, 1].set(title="Cumulative distribution",
                 xlabel="token rank", ylabel="cumulative %")

    data = [v for k, v in cats.items() if k != "Other"]
    labels = [f"{k} (n={len(v)})" for k, v in cats.items() if k != "Other"]
    ax[1, 2].boxplot(data, vert=False, patch_artist=True, showfliers=False)
    ax[1, 2].set_yticklabels(labels, fontsize=8)
    ax[1, 2].set_xscale("log")
    ax[1, 2].set_title("Counts by category")

    fig.tight_layout()
    out_png = out_dir / "analysis.png"
    fig.savefig(out_png)
    plt.close(fig)
    log.info("✓ Analysis plots saved → %s", out_png.name)

    stats = dict(
        total_chunks=int(len(chunks)),
        total_tokens=int(len(all_ids)),
        unique_tokens=int(len(counts)),
        vocab_size=int(tok.vocab_size),
        coverage_pct=float(len(counts)/tok.vocab_size*100),
        avg_chunk_len=float(lengths.mean()),
        median_chunk_len=float(np.median(lengths))
    )
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    log.info("✓ Statistics saved → %s", "stats.json")


# ─────────── preview audio (preserved exactly) ───────────
def preview_audio(chunks: List[List[int]], tok: REMI, out_dir: Path, sf2: Path, log: logging.Logger):
    """Generate audio preview - unchanged from original."""
    idx, ch = next(((i, c) for i, c in enumerate(chunks) if len(c) >= MIN_LEN), (-1, None))
    if ch is None:
        log.warning("⚠ Preview skipped – no chunk ≥ %s tokens", MIN_LEN)
        return
    log.info("✓ Generating preview from chunk %d (%s tokens)", idx, f"{len(ch):,}")

    ids = ch + ([tok["EOS_None"]] if "EOS_None" in tok.vocab and ch[-1] != tok["EOS_None"] else [])
    score = robust_decode(tok, ids)

    mid_out = out_dir / "preview.mid"
    wav_out = out_dir / "preview.wav"
    score.dump_midi(mid_out)
    log.info("  • MIDI saved → %s", mid_out.name)

    try:
        import pretty_midi
        import soundfile as sf
        pm = pretty_midi.PrettyMIDI(str(mid_out))
        audio = None
        if sf2.exists():
            try:
                audio = pm.fluidsynth(sf2_path=str(sf2), fs=48_000)
                log.info("  • Audio rendered with FluidSynth")
            except Exception as e:
                log.warning("  ⚠ FluidSynth failed: %s", str(e).split('\n')[0])
        if audio is None:
            try:
                from midi2audio import FluidSynth
                FluidSynth(sound_font=str(sf2) if sf2.exists() else None)\
                    .midi_to_audio(mid_out, wav_out)
                log.info("  • Audio rendered with midi2audio")
                return
            except Exception as e:
                log.warning("  ⚠ midi2audio failed, using fallback synthesizer")
                audio = pm.synthesize(fs=48_000)
                log.info("  • Audio rendered with pretty_midi synthesizer")

        audio = audio / np.abs(audio).max() * .89
        sf.write(wav_out, audio, 48_000)
        log.info("✓ Audio preview saved → %s (%.1fs)", wav_out.name, audio.shape[0]/48_000)
    except ImportError:
        log.info("⚠ Audio libraries missing – preview skipped")


# ─────────── improved runner ───────────
def run(args):
    """Enhanced main runner with metadata tracking."""
    # Setup logging with dataset name
    dataset_name = args.dataset_name if hasattr(args, 'dataset_name') else "dataset"
    log = setup_logging(args.log_dir, dataset_name)
    
    # Log start
    start_time = datetime.now()
    log.info("╔" + "═"*78 + "╗")
    log.info("║" + f" MIDI TOKENIZATION - {dataset_name.upper()} ".center(78) + "║")
    log.info("╚" + "═"*78 + "╝")
    log.info("")
    log.info("Configuration:")
    log.info("  • MIDI directory  : %s", args.midi_dir)
    log.info("  • Output directory: %s", args.proc_dir)
    log.info("  • Vocabulary      : %s", args.vocab_path)
    log.info("  • Workers         : %d", args.workers)
    log.info("")
    
    args.proc_dir.mkdir(parents=True, exist_ok=True)

    tok = load_or_create_tokeniser(args.vocab_path, log)
    bar_id = tok["Bar_None"]

    # ── 1. discover files ───────────────────────────────────────
    log.info("─" * 40)
    log.info("PHASE 1: File Discovery")
    log.info("─" * 40)
    
    all_midis = sorted(args.midi_dir.rglob("*.mid*"))
    if not all_midis:
        log.error("❌ No MIDI files found in %s", args.midi_dir)
        sys.exit(1)
    log.info("✓ Found %s MIDI files", f"{len(all_midis):,}")

    # ── 2. sanity‑filter ────────────────────────────────────────
    log.info("")
    log.info("─" * 40)
    log.info("PHASE 2: File Validation")
    log.info("─" * 40)
    
    pre_workers = min(args.workers, 32)
    log.info("✓ Validating files with %d workers...", pre_workers)
    
    def _init_worker():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    mask = []
    with Pool(pre_workers, initializer=_init_worker) as pool:
        with tqdm(total=len(all_midis), desc="Validating", 
                  bar_format="{l_bar}{bar:40}{r_bar}", leave=True) as pbar:
            for ok in pool.imap_unordered(is_valid_midi, all_midis, chunksize=64):
                mask.append(ok)
                pbar.update(1)

    midi_files = [p for p, ok in zip(all_midis, mask) if ok]
    bad_files = [p for p, ok in zip(all_midis, mask) if not ok]
    
    log.info("")
    log.info("Validation Results:")
    log.info("  • Valid files  : %s (%.1f%%)", 
             f"{len(midi_files):,}", len(midi_files)/len(all_midis)*100)
    log.info("  • Invalid files: %s (%.1f%%)", 
             f"{len(bad_files):,}", len(bad_files)/len(all_midis)*100)
    
    if not midi_files:
        log.error("❌ All MIDI files failed validation!")
        sys.exit(1)
        
    if bad_files:
        skip_path = args.proc_dir / "skipped_corrupt_midis.txt"
        skip_path.write_text("\n".join(str(p) for p in bad_files) + "\n")
        log.info("  • Invalid file list saved to: %s", skip_path.name)

    # ── 3. tokenisation pool (improved) ────────────────────────────────────
    log.info("")
    log.info("─" * 40)
    log.info("PHASE 3: Tokenization")
    log.info("─" * 40)
    log.info("✓ Processing %s files with %d workers...", f"{len(midi_files):,}", args.workers)
    
    worker_args = [(p, str(args.vocab_path), "Bar_None", args.midi_dir) for p in midi_files]
    
    seqs, fails = [], []
    file_metadata = {}
    
    with Pool(processes=args.workers) as pool:
        with tqdm(total=len(worker_args), desc="Tokenizing",
                  bar_format="{l_bar}{bar:40}{r_bar}", leave=True) as pbar:
            for result in pool.imap_unordered(_tokenise_one_improved, worker_args):
                if result["success"]:
                    # Track where tokens came from
                    start_idx = len(seqs)
                    seqs.extend(result["tokens"])
                    end_idx = len(seqs)
                    
                    file_metadata[result["file"]] = {
                        "token_range": [start_idx, end_idx],
                        "num_sequences": len(result["tokens"]),
                        "hash": result["hash"],
                    }
                else:
                    fails.append((result["file"], result["error"]))
                pbar.update(1)

    log.info("")
    if fails:
        log.info("Tokenization Errors: %d files failed", len(fails))
        
        # Group errors by type
        error_types = {}
        for file, error in fails:
            # Extract error type (first part of error message)
            error_type = error.split(":")[0] if ":" in error else error
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(file)
        
        # Display grouped errors
        for error_type, files in list(error_types.items())[:5]:  # Show top 5 error types
            log.info("  • %s (%d files)", error_type, len(files))
            for file in files[:2]:  # Show first 2 files per error type
                log.info("    - %s", file)
            if len(files) > 2:
                log.info("    ... and %d more", len(files) - 2)
        
        if len(error_types) > 5:
            log.info("  ... and %d more error types", len(error_types) - 5)
        
        # Save full failure list
        fail_path = args.proc_dir / "failed_files.json"
        with open(fail_path, "w") as f:
            json.dump(fails, f, indent=2)
        log.info("  • Full error list saved to: %s", fail_path.name)

    # Save tokens with metadata
    log.info("")
    log.info("─" * 40)
    log.info("PHASE 4: Saving Results")
    log.info("─" * 40)
    
    raw_pkl = args.proc_dir / "raw_tokens.pkl"
    pickle.dump({"tokens": seqs}, raw_pkl.open("wb"))
    log.info("✓ Raw tokens saved → %s", raw_pkl.name)

    # Save file mapping
    metadata = {
        "dataset": dataset_name,
        "tokenization_date": datetime.now().isoformat(),
        "vocab_path": str(args.vocab_path),
        "vocab_size": tok.vocab_size,
        "total_files": len(midi_files),
        "successful_files": len(file_metadata),
        "failed_files": len(fails),
        "total_sequences": len(seqs),
        "file_mapping": file_metadata,
    }
    
    metadata_path = args.proc_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("✓ Metadata saved → %s", metadata_path.name)

    # Create chunks
    chunks = [c for s in seqs for c in bar_chunks(s, bar_id)]
    (args.proc_dir / "chunks.pkl").write_bytes(pickle.dumps({"tokens": chunks}))
    log.info("✓ Chunks saved → %s (%s chunks)", "chunks.pkl", f"{len(chunks):,}")

    # Update metadata with chunk info
    metadata["total_chunks"] = len(chunks)
    metadata["avg_chunks_per_file"] = len(chunks) / len(file_metadata) if file_metadata else 0
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if not args.dry_run:
        log.info("")
        log.info("─" * 40)
        log.info("PHASE 5: Analysis & Preview")
        log.info("─" * 40)
        create_visuals(chunks, tok, args.proc_dir, log)
        preview_audio(chunks, tok, args.proc_dir, args.soundfont, log)

    # Final summary
    duration = (datetime.now() - start_time).total_seconds()
    log.info("")
    log.info("╔" + "═"*78 + "╗")
    log.info("║" + " TOKENIZATION COMPLETE ".center(78) + "║")
    log.info("╠" + "═"*78 + "╣")
    log.info("║ Duration       : %-60s║" % f"{duration:.1f} seconds")
    log.info("║ Files processed: %-60s║" % f"{len(file_metadata):,} / {len(midi_files):,}")
    log.info("║ Success rate   : %-60s║" % f"{len(file_metadata)/len(midi_files)*100:.1f}%")
    log.info("║ Total sequences: %-60s║" % f"{len(seqs):,}")
    log.info("║ Total chunks   : %-60s║" % f"{len(chunks):,}")
    log.info("║ Avg chunks/file: %-60s║" % f"{len(chunks)/len(file_metadata):.1f}" if file_metadata else "0")
    log.info("║ Output location: %-60s║" % str(args.proc_dir))
    log.info("╚" + "═"*78 + "╝")


# ─────────── CLI (enhanced) ───────────
def cli():
    p = argparse.ArgumentParser(
        description="Tokenise MIDI datasets with metadata tracking",
        formatter_class=lambda prog: argparse.HelpFormatter(
            prog, max_help_position=34, width=90))
    
    p.add_argument("--midi_dir", type=Path, required=True,
                   help="Folder containing *.mid / *.midi files")
    p.add_argument("--proc_dir", type=Path, required=True,
                   help="Output directory for pickles, plots, preview")
    p.add_argument("--vocab_path", type=Path, required=True,
                   help="Tokenizer JSON (created if missing)")
    p.add_argument("--dataset_name", type=str, default="dataset",
                   help="Dataset name for tracking (e.g., maestro, lakh)")
    p.add_argument("--soundfont", type=Path, default=DEFAULT_SF,
                   help=f"GM SoundFont (default: {DEFAULT_SF})")
    p.add_argument("--log_dir", type=Path, required=True,
                   help="Folder for rotating logs")
    p.add_argument("--workers", type=int, default=min(8, cpu_count()),
                   help="Tokenisation worker processes (default: min(8, CPU cores))")
    p.add_argument("--dry_run", action="store_true",
                   help="Skip plots and audio preview")
    
    return p.parse_args()


if __name__ == "__main__":
    if os.name == "nt":
        import multiprocessing as mp
        mp.set_start_method("spawn")
    run(cli())