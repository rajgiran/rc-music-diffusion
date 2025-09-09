#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract CLAP embeddings from rendered audio files.

Updates (industry-ready):
- **Manifest-first**: accept a JSONL manifest (one {"id", "path"} per line) or a plain paths file.
- **Bounded streaming pipeline**: capped prefetch keeps RAM flat even on huge corpora.
- **Thread/Process I/O**: `--io_backend {thread,process}` (process uses safe **spawn** with CUDA).
- **Memmap output**: `--memmap_out` writes directly to a NumPy `.npy` via `open_memmap` (constant RAM). Produces a single compact file.
- **Progress bars**: tqdm for loader and overall.
- **AMP on CUDA**: autocast FP16 compute while keeping model weights FP32 (stable BatchNorm).
- **Deterministic order**: final embeddings follow manifest order (failed files are skipped).
- **ID alignment**: optional `<output>_ids.npy` (only successful rows, in row order).
- **Back-compat**: folder scan mode preserved.

Artifacts written next to --output_path:
- <output>.npy                      — float32 embeddings [N_success, D]
- <output>.json                    — metadata (file_paths, failed_files, model, etc.)
- <output>_ids.npy   (optional)    — int64 IDs aligned to rows (when manifest has "id" and --write_ids_npy)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional, List, Tuple, Iterable, Dict
import logging
import os
import multiprocessing as mp
from collections import deque

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

import torch
from transformers import ClapModel, ClapProcessor
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future, wait, FIRST_COMPLETED
from numpy.lib.format import open_memmap

# ─────────────────────────── logging ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("clap_extract")

SR = 48_000

# ───────────────────────── util ─────────────────────────
def _fix_len(audio: np.ndarray, sr: int, seconds: float | None) -> np.ndarray:
    """Crop/pad to exactly seconds if provided."""
    if seconds is None:
        return audio.astype(np.float32)
    T = int(round(seconds * sr))
    n = audio.shape[0]
    if n == T:
        return audio.astype(np.float32)
    if n > T:
        return audio[:T].astype(np.float32)
    out = np.zeros(T, dtype=np.float32)
    out[:n] = audio.astype(np.float32)
    return out

# Worker for parallel I/O (thread/process-safe)
# returns (idx, path_str, wave_or_None, error_or_None)
def _load_and_preprocess(args: Tuple[int, str, float | None]) -> Tuple[int, str, Optional[np.ndarray], Optional[str]]:
    idx, path_str, clip_seconds = args
    try:
        p = Path(path_str)
        # read as float32 to reduce RAM
        wave, sr = sf.read(p, dtype='float32')
        if getattr(wave, "ndim", 1) > 1:
            wave = wave.mean(axis=1)
        if sr != SR:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=SR)
        wave = _fix_len(wave, SR, clip_seconds)
        return idx, path_str, wave.astype(np.float32, copy=False), None
    except Exception as e:
        return idx, path_str, None, f"{type(e).__name__}: {e}"

# ──────────────────────── extractor class ───────────────────────
class CLAPEmbeddingExtractor:
    """GPU-batched CLAP embedding extractor."""

    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        device: Optional[str] = None,
        clip_seconds: Optional[float] = None,
        amp: bool = True,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.clip_seconds = clip_seconds
        self.amp = amp and (device.startswith("cuda") and torch.cuda.is_available())

        logger.info(f"Loading CLAP model: {model_name}")
        # Keep weights in FP32 for BatchNorm stability; use autocast for compute.
        self.model = ClapModel.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model.eval()
        logger.info("CLAP model loaded – ready")

    # forward 1 batch
    def _forward_batch(
        self,
        batch_audio: List[np.ndarray],
        embeddings_out: List[np.ndarray],
    ) -> np.ndarray:
        inputs = self.processor(
            audios=batch_audio,
            sampling_rate=SR,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            if self.amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feats = self.model.get_audio_features(**inputs)
            else:
                feats = self.model.get_audio_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.cpu().numpy().astype(np.float32, copy=False)
        embeddings_out.extend(feats)
        return feats

    # bounded streaming over files list
    def process_file_list(
        self,
        files: List[Path],
        output_path: Path,
        batch_size: int = 32,
        ids: Optional[List[int]] = None,
        io_workers: int = 8,
        io_backend: str = "thread",
        prefetch_batches: int = 2,
        memmap_out: bool = False,
    ) -> tuple[int, int, List[int]]:
        """Stream through `files` with bounded prefetch and GPU batching.
        Returns (num_success, num_failed, ids_success).
        """
        N = len(files)
        D = int(getattr(self.model.config, "projection_dim", 512))
        max_outstanding = max(1, int(prefetch_batches) * int(batch_size))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # outputs / bookkeeping
        ids_success: List[int] = []
        file_paths_success: List[str] = []
        failed_files: List[str] = []

        # Either collect in memory (small sets) or write to memmap staging directly
        if memmap_out:
            # stage to a compact file directly with open_memmap sized to successes is not possible a priori,
            # so we stage to a temporary memmap by index and compact at the end.
            # ensure output directory exists before creating memmaps
            output_path.parent.mkdir(parents=True, exist_ok=True)
            stage_path = output_path.with_name(output_path.stem + "_staging.npy")
            stage = open_memmap(stage_path, mode='w+', dtype=np.float32, shape=(N, D))
            success_mask = np.zeros(N, dtype=bool)
        else:
            results_by_idx: Dict[int, np.ndarray] = {}

        # choose executor
        if io_backend == "process":
            ctx = mp.get_context("spawn")  # safe with CUDA
            executor = ProcessPoolExecutor(max_workers=max(1, io_workers), mp_context=ctx)
        elif io_backend == "thread":
            executor = ThreadPoolExecutor(max_workers=max(1, io_workers))
        else:
            raise SystemExit("--io_backend must be 'thread' or 'process'")

        # submit/consume with bounded outstanding
        futures: Dict[int, Future] = {}
        pending_i = 0
        completed = 0
        pbar = tqdm(total=N, desc=f"Load+prep[{io_backend}]", smoothing=0.05)

        # helper to submit one
        def submit_one(i: int):
            futures[i] = executor.submit(_load_and_preprocess, (i, str(files[i]), self.clip_seconds))

        # prime pool up to the cap
        while pending_i < N and len(futures) < max_outstanding:
            submit_one(pending_i)
            pending_i += 1

        # batch buffers
        batch_audio: List[np.ndarray] = []
        batch_indices: List[int] = []

        while futures:
            done, _ = wait(list(futures.values()), return_when=FIRST_COMPLETED, timeout=1.0)
            if not done:
                continue  # no progress yet; loop again
            for fut in done:
                # find idx for this future
                idx = next(k for k, v in list(futures.items()) if v is fut)
                try:
                    _, path_str, wav, err = fut.result()
                except Exception as e:
                    path_str, wav, err = str(files[idx]), None, f"FutureError: {e}"
                del futures[idx]
                pbar.update(1)
                completed += 1

                # keep the cap by submitting more
                while pending_i < N and len(futures) < max_outstanding:
                    submit_one(pending_i)
                    pending_i += 1

                if wav is None:
                    failed_files.append(path_str)
                else:
                    batch_audio.append(wav)
                    batch_indices.append(idx)
                    if len(batch_audio) == batch_size:
                        feats = self._forward_batch(batch_audio, embeddings_out=[],)
                        if memmap_out:
                            stage[batch_indices, :] = feats
                            success_mask[batch_indices] = True
                            if ids is not None:
                                for bi in batch_indices:
                                    ids_success.append(int(ids[bi]))
                                    file_paths_success.append(str(files[bi]))
                            else:
                                for bi in batch_indices:
                                    file_paths_success.append(str(files[bi]))
                        else:
                            for bi, fv in zip(batch_indices, feats):
                                results_by_idx[bi] = fv
                            if ids is not None:
                                for bi in batch_indices:
                                    ids_success.append(int(ids[bi]))
                                    file_paths_success.append(str(files[bi]))
                            else:
                                for bi in batch_indices:
                                    file_paths_success.append(str(files[bi]))
                        batch_audio.clear(); batch_indices.clear()

        # flush final partial batch if any
        if batch_audio:
            feats = self._forward_batch(batch_audio, embeddings_out=[],)
            if memmap_out:
                stage[batch_indices, :] = feats
                success_mask[batch_indices] = True
                if ids is not None:
                    for bi in batch_indices:
                        ids_success.append(int(ids[bi]))
                        file_paths_success.append(str(files[bi]))
                else:
                    for bi in batch_indices:
                        file_paths_success.append(str(files[bi]))
            else:
                for bi, fv in zip(batch_indices, feats):
                    results_by_idx[bi] = fv
                if ids is not None:
                    for bi in batch_indices:
                        ids_success.append(int(ids[bi]))
                        file_paths_success.append(str(files[bi]))
                else:
                    for bi in batch_indices:
                        file_paths_success.append(str(files[bi]))

        pbar.close()
        executor.shutdown(wait=True, cancel_futures=False)

        # finalize outputs
        if memmap_out:
            # compact to final file in manifest order (skip failures)
            success_indices = np.flatnonzero(success_mask)
            num_success = int(success_indices.size)
            final_mm = open_memmap(output_path, mode='w+', dtype=np.float32, shape=(num_success, D))
            # copy in chunks to avoid RAM spikes
            CH = 8192
            dst = 0
            for off in range(0, num_success, CH):
                idx_chunk = success_indices[off:off+CH]
                final_mm[dst:dst+len(idx_chunk)] = stage[idx_chunk]
                dst += len(idx_chunk)
            del final_mm
            # cleanup staging file
            try:
                os.remove(stage_path)  # type: ignore[attr-defined]
            except Exception:
                pass
        else:
            # assemble in-manifest-order compact array
            success_indices = sorted(results_by_idx.keys())
            num_success = len(success_indices)
            arr = np.empty((num_success, D), dtype=np.float32)
            for j, bi in enumerate(success_indices):
                arr[j] = results_by_idx[bi]
            np.save(output_path, arr)

        # write metadata and ids
        if ids is not None and ids_success:
            np.save(output_path.with_name(output_path.stem + "_ids.npy"), np.array(ids_success, dtype=np.int64))

        meta = {
            "source": "manifest_or_paths",
            "num_files": N,
            "num_embeddings": int(num_success),
            "num_failed": int(N - num_success),
            "embedding_dim": D,
            "model_name": self.model.name_or_path,
            "file_paths": file_paths_success,
            "failed_files": failed_files,
            "batch_size": int(batch_size),
            "clip_seconds": self.clip_seconds,
            "sampling_rate": SR,
            "amp": bool(self.amp),
            "io_backend": io_backend,
            "io_workers": int(io_workers),
            "prefetch_batches": int(prefetch_batches),
        }
        with open(output_path.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Embeddings extracted: {num_success}  | dim = {D}")
        logger.info(f"Saved embeddings → {output_path}")
        logger.info(f"Metadata → {output_path.with_suffix('.json')}")

        return num_success, (N - num_success), ids_success

    # Back-compat: directory scan
    def process_directory(
        self,
        audio_dir: Path,
        output_path: Path,
        batch_size: int = 32,
        max_files: int | None = None,
        extensions: tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg", ".m4a"),
        recursive: bool = True,
        io_workers: int = 8,
        io_backend: str = "thread",
        prefetch_batches: int = 2,
        memmap_out: bool = False,
    ) -> tuple[int, int, List[int]]:
        if recursive:
            audio_files = [p for p in audio_dir.rglob("*") if p.suffix.lower() in extensions]
        else:
            audio_files = [p for p in audio_dir.iterdir() if p.suffix.lower() in extensions]
        audio_files = sorted(audio_files)
        logger.info(f"Found {len(audio_files)} audio files (recursive={recursive}, exts={list(extensions)})")
        if max_files is not None:
            audio_files = audio_files[: max_files]
            logger.info(f"Processing first {len(audio_files)} files")
        return self.process_file_list(
            audio_files, output_path, batch_size=batch_size, ids=None,
            io_workers=io_workers, io_backend=io_backend,
            prefetch_batches=prefetch_batches, memmap_out=memmap_out,
        )

# ─────────────────────────────── CLI ──────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Extract CLAP embeddings from audio (manifest or folder)")
    # Inputs
    ap.add_argument("--audio_dir", type=Path, help="Directory with audio files")
    ap.add_argument("--manifest_jsonl", type=Path, help="JSONL with {'id': int, 'path': str} per line")
    ap.add_argument("--paths_file", type=Path, help="Plain text: one absolute path per line")

    # Output
    ap.add_argument("--output_path", type=Path, required=True, help="Destination .npy for embeddings")

    # Model/inference
    ap.add_argument("--model", type=str, default="laion/clap-htsat-unfused", help="CLAP HF repo")
    ap.add_argument("--device", type=str, default=None, help="cuda / cpu")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size for GPU forward")
    ap.add_argument("--clip_seconds", type=float, default=10, help="Crop/pad length before CLAP")
    ap.add_argument("--amp", action="store_true", help="Enable autocast FP16 on CUDA")

    # Dir scan settings
    ap.add_argument("--max_files", type=int, default=None, help="Debug: limit number of files (dir scan or paths_file)")
    ap.add_argument("--no_recursive", action="store_true", help="Disable recursive file search (dir mode)")
    ap.add_argument("--extensions", type=str, default=".wav,.mp3,.flac,.ogg,.m4a", help="Comma-separated extensions for dir mode")

    # I/O
    ap.add_argument("--io_workers", type=int, default=8, help="Parallel workers for audio I/O and resampling")
    ap.add_argument("--io_backend", type=str, default="thread", choices=["thread","process"], help="I/O parallelism backend")
    ap.add_argument("--prefetch_batches", type=int, default=2, help="Max batches to prefetch into RAM ahead of GPU")
    ap.add_argument("--memmap_out", action="store_true", help="Write directly to final .npy via open_memmap (constant RAM)")

    # Extras
    ap.add_argument("--write_ids_npy", action="store_true", help="If manifest has 'id', also write <output>_ids.npy aligned to rows")

    args = ap.parse_args()

    if not args.audio_dir and not args.manifest_jsonl and not args.paths_file:
        raise SystemExit("Provide either --manifest_jsonl, --paths_file, or --audio_dir.")

    # set small BLAS threads to avoid oversubscription when using parallel I/O
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")

    extractor = CLAPEmbeddingExtractor(
        model_name=args.model,
        device=args.device,
        clip_seconds=args.clip_seconds,
        amp=args.amp,
    )

    # Manifest JSONL mode (preferred)
    if args.manifest_jsonl:
        files: List[Path] = []
        ids: List[int] = []
        with open(args.manifest_jsonl, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                files.append(Path(rec["path"]))
                ids.append(int(rec.get("id", -1)))
        if args.max_files is not None:
            files = files[: args.max_files]
            ids = ids[: args.max_files]
        succ, fail, ids_succ = extractor.process_file_list(
            files, args.output_path, batch_size=args.batch_size, ids=ids,
            io_workers=args.io_workers, io_backend=args.io_backend,
            prefetch_batches=args.prefetch_batches, memmap_out=args.memmap_out,
        )
        if args.write_ids_npy and ids_succ:
            np.save(args.output_path.with_name(args.output_path.stem + "_ids.npy"), np.array(ids_succ, dtype=np.int64))
        logger.info(f"Done.  ✅ {succ} succeeded   ❌ {fail} failed")
        return

    # Plain paths file mode
    if args.paths_file:
        paths = [Path(p.strip()) for p in open(args.paths_file) if p.strip()]
        if args.max_files is not None:
            paths = paths[: args.max_files]
        succ, fail, ids_succ = extractor.process_file_list(
            paths, args.output_path, batch_size=args.batch_size, ids=None,
            io_workers=args.io_workers, io_backend=args.io_backend,
            prefetch_batches=args.prefetch_batches, memmap_out=args.memmap_out,
        )
        logger.info(f"Done.  ✅ {succ} succeeded   ❌ {fail} failed")
        return

    # Directory scan mode (back-compat)
    if args.audio_dir:
        extensions = tuple([e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}" for e in (args.extensions or "").split(",") if e.strip()])
        succ, fail, ids_succ = extractor.process_directory(
            audio_dir=args.audio_dir,
            output_path=args.output_path,
            batch_size=args.batch_size,
            max_files=args.max_files,
            extensions=extensions,
            recursive=not args.no_recursive,
            io_workers=args.io_workers,
            io_backend=args.io_backend,
            prefetch_batches=args.prefetch_batches,
            memmap_out=args.memmap_out,
        )
        logger.info(f"Done.  ✅ {succ} succeeded   ❌ {fail} failed")

if __name__ == "__main__":
    main()