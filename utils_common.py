# utils_common.py
from __future__ import annotations
import logging, os, subprocess, json, shlex, time, warnings
from pathlib import Path
import numpy as np
import soundfile as sf

def setup_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def setup_logging(log_dir: Path, name: str):
    setup_dirs(log_dir)
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_dir / f"{name}.log")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    if not log.handlers:
        log.addHandler(fh)
        log.addHandler(ch)
    return log

def run(cmd: str, log: logging.Logger, check=True, env=None):
    log.info(f"$ {cmd}")
    t0 = time.time()
    proc = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
    log.info(proc.stdout)
    dt = time.time() - t0
    log.info(f"[done in {dt:.2f}s]")
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return proc

def save_flac(path: Path, audio: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, format="FLAC")