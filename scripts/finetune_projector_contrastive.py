#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ========== Environment hygiene BEFORE importing HF ==========
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# ========== Stdlib ==========
import math
import json
import time
import random
import argparse
import warnings
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

# ========== Third-party ==========
import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import ClapProcessor, ClapModel
from tqdm import tqdm

# Optuna (optional)
try:
    import optuna
    from optuna.exceptions import ExperimentalWarning
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
    OPTUNA_OK = True
except Exception:
    OPTUNA_OK = False


# -----------------------------
# Logging Setup
# -----------------------------
def setup_logging(log_dir: Path, name: str = "train"):
    """Setup logging with file and console handlers"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    # Root logger basic config
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    # Quiet down Matplotlib's categorical-units chatter (keep real warnings)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.category").setLevel(logging.WARNING)

    return logging.getLogger(name)


# -----------------------------
# Repro
# -----------------------------
def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Data
# -----------------------------
class PairsDS(Dataset):
    """Dataset for audio-text pairs from CSV"""

    def __init__(self, csv_path: Path, max_items=None, target_sr=48_000,
                 seconds=10.0, augment=False):
        df = pd.read_csv(csv_path)

        if "audio_path" not in df.columns or "text" not in df.columns:
            raise ValueError("CSV must contain columns: audio_path, text")

        rows = []
        for _, r in df.iterrows():
            ap = str(r["audio_path"]).strip()
            tx = str(r["text"]).strip()
            if ap and tx and Path(ap).exists():
                rows.append((ap, tx))

        if max_items is not None:
            rows = rows[: int(max_items)]

        self.rows = rows
        self.target_sr = int(target_sr)
        self.samples = int(seconds * target_sr)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.rows)

    def _load_wave(self, p: str) -> np.ndarray:
        """Load and preprocess audio waveform"""
        y, sr = librosa.load(p, sr=self.target_sr, mono=True)

        # Trim/pad to length
        if len(y) < self.samples:
            y = np.pad(y, (0, self.samples - len(y)))
        else:
            y = y[: self.samples]

        y = y.astype(np.float32)

        # Light augmentation (optional)
        if self.augment:
            # Random gain [-3dB, +3dB]
            g = 10 ** (np.random.uniform(-3, 3) / 20.0)
            y = np.clip(y * g, -1.0, 1.0).astype(np.float32)

        return y

    def __getitem__(self, idx):
        ap, tx = self.rows[idx]
        y = self._load_wave(ap)
        return y, tx


def collate_list(batch: List[Tuple[np.ndarray, str]]):
    """Collate function for DataLoader"""
    waves = [b[0] for b in batch]
    texts = [b[1] for b in batch]
    return waves, texts


# -----------------------------
# Projectors
# -----------------------------
class MLPProjector(nn.Module):
    """MLP projector with residual connection"""

    def __init__(self, dim=512, hidden=1024, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return self.ln(self.net(x) + x)


def build_projector(kind: str, dropout: float = 0.0):
    """Factory function for building projectors"""
    if kind == "mlp":
        logging.info(f"Using internal MLP projector (dropout={dropout:.2f})")
        return MLPProjector(dropout=dropout)
    elif kind == "external":
        try:
            from aldm_style_adapter import ClapAudioProjector
            logging.info("Using external aldm2_style_adapter.ClapAudioProjector")
            return ClapAudioProjector()
        except ImportError as e:
            raise ImportError(
                "External adapter not found. Ensure aldm2_style_adapter is in PYTHONPATH"
            ) from e
    else:
        raise ValueError(f"Unknown adapter type: {kind}")


# -----------------------------
# Checkpoint Management
# -----------------------------
class CheckpointManager:
    """Manage model checkpoints with loading/saving"""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model: nn.Module, path: Path, metadata: Optional[Dict] = None):
        """Save model checkpoint with optional metadata"""
        checkpoint = {
            "state_dict": model.state_dict(),
            "metadata": metadata or {},
        }
        torch.save(checkpoint, path)
        logging.info(f"Saved checkpoint to {path}")

    def load(self, model: nn.Module, path: Path, strict: bool = True) -> Dict:
        """Load model checkpoint (safe load if supported)"""
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Prefer safe load if available in this PyTorch version
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)  # PyTorch >= 2.4
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")  # older versions

        sd = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(sd, strict=strict)
        logging.info(f"Loaded checkpoint from {path}")
        return checkpoint.get("metadata", {})

    def load_if_exists(self, model: nn.Module, path: Path, strict: bool = True) -> bool:
        """Load checkpoint if it exists, return True if loaded"""
        if path.exists():
            try:
                self.load(model, path, strict=strict)
                return True
            except Exception as e:
                logging.warning(f"Failed to load checkpoint: {e}")
        return False


# -----------------------------
# Loss + Retrieval Metrics
# -----------------------------
def info_nce(audio_z: torch.Tensor, text_z: torch.Tensor, temperature: float = 0.07):
    """InfoNCE contrastive loss"""
    a = F.normalize(audio_z, dim=-1)
    t = F.normalize(text_z, dim=-1)
    logits = (a @ t.T) / temperature  # (B,B)
    labels = torch.arange(a.size(0), device=a.device)

    loss_a2t = F.cross_entropy(logits, labels)
    loss_t2a = F.cross_entropy(logits.T, labels)
    loss = 0.5 * (loss_a2t + loss_t2a)

    # R@1
    r1_t2a = (logits.argmax(dim=1) == labels).float().mean().item()
    r1_a2t = (logits.argmax(dim=0) == labels).float().mean().item()

    return loss, r1_t2a, r1_a2t, logits


def recalls_at_k(logits: torch.Tensor, ks=(1, 5, 10)) -> Dict[str, float]:
    """Calculate Recall@K metrics"""
    B = logits.size(0)
    labels = torch.arange(B, device=logits.device)
    out: Dict[str, float] = {}

    if B == 0:
        for k in ks:
            out[f"t2a@{k}"] = 0.0
            out[f"a2t@{k}"] = 0.0
        return out

    for k in ks:
        kk = min(k, B)

        # text → audio (rows)
        topk_rows = logits.topk(kk, dim=1).indices
        match_rows = (topk_rows == labels[:, None]).any(dim=1).float().mean().item()
        out[f"t2a@{k}"] = match_rows

        # audio → text (cols)
        topk_cols = logits.topk(kk, dim=0).indices
        match_cols = (topk_cols == labels[None, :]).any(dim=0).float().mean().item()
        out[f"a2t@{k}"] = match_cols

    return out


# -----------------------------
# LR Scheduler: Cosine with Warmup
# -----------------------------
class CosineWarmup:
    """Cosine learning rate scheduler with linear warmup"""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7):
        self.opt = optimizer
        self.warm = int(warmup_steps)
        self.total = int(total_steps)
        self.min_lr = float(min_lr)
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.t = 0

    def step(self):
        self.t += 1
        for i, pg in enumerate(self.opt.param_groups):
            base = self.base_lrs[i]
            if self.t <= self.warm:
                lr = base * self.t / max(self.warm, 1)
            else:
                tt = max(self.t - self.warm, 1)
                T = max(self.total - self.warm, 1)
                lr = self.min_lr + 0.5 * (base - self.min_lr) * (1 + math.cos(math.pi * tt / T))
            pg["lr"] = lr

    def get_lrs(self):
        return [pg["lr"] for pg in self.opt.param_groups]


# -----------------------------
# Train / Eval
# -----------------------------
@dataclass
class TrainConfig:
    """Training configuration"""
    epochs: int
    batch: int
    lr: float
    temperature: float
    clip_grad: float
    warmup_ratio: float
    min_lr: float
    weight_decay: float
    dropout: float  # only used if adapter=mlp


def run_epoch(model, projector, processor, device, dl,
              optimizer=None, scaler=None, temperature=0.07, clip_grad=1.0):
    """Run single epoch of training or validation"""
    training = optimizer is not None

    if training:
        model.eval()
        projector.train()
    else:
        model.eval()
        projector.eval()

    losses, r1_t2a_list, r1_a2t_list = [], [], []
    r5_t2a_list, r10_t2a_list, r5_a2t_list, r10_a2t_list = [], [], [], []

    pbar = tqdm(dl, leave=False)
    for waves, texts in pbar:
        # Process inputs with CLAP
        with torch.no_grad():
            proc = processor(
                audios=waves,
                text=texts,
                sampling_rate=processor.feature_extractor.sampling_rate,
                return_tensors="pt",
                padding=True,
            )
            for k in proc:
                proc[k] = proc[k].to(device, non_blocking=True)

            out = model(**proc)
            a_z = F.normalize(out.audio_embeds, dim=-1)
            t_z = F.normalize(out.text_embeds, dim=-1)

        # Apply projector
        a_proj = projector(a_z)

        if training:
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                loss, r1_t2a, r1_a2t, logits = info_nce(a_proj, t_z, temperature=temperature)

            scaler.scale(loss).backward()

            if clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(projector.parameters(), max_norm=clip_grad)

            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                loss, r1_t2a, r1_a2t, logits = info_nce(a_proj, t_z, temperature=temperature)

        # Compute additional metrics
        rks = recalls_at_k(logits, ks=(5, 10))

        # Update statistics
        losses.append(loss.item())
        r1_t2a_list.append(r1_t2a)
        r1_a2t_list.append(r1_a2t)
        r5_t2a_list.append(rks["t2a@5"])
        r10_t2a_list.append(rks["t2a@10"])
        r5_a2t_list.append(rks["a2t@5"])
        r10_a2t_list.append(rks["a2t@10"])

        pbar.set_description(f"loss={np.mean(losses):.4f}  R@1 t2a={np.mean(r1_t2a_list):.3f}")

    metrics = {
        "loss": float(np.mean(losses)),
        "t2a@1": float(np.mean(r1_t2a_list)),
        "a2t@1": float(np.mean(r1_a2t_list)),
        "t2a@5": float(np.mean(r5_t2a_list)),
        "t2a@10": float(np.mean(r10_t2a_list)),
        "a2t@5": float(np.mean(r5_a2t_list)),
        "a2t@10": float(np.mean(r10_a2t_list)),
    }

    return metrics


def fit_once(args, config: TrainConfig, work_dir: Path,
             ds_tr: Dataset, ds_va: Dataset,
             processor: ClapProcessor, model: ClapModel,
             projector: nn.Module, device: torch.device,
             load_best: bool = False) -> Dict:
    """Train model once with given configuration"""

    # Setup logging
    logger = setup_logging(work_dir, "train")

    # Checkpoint manager
    ckpt_manager = CheckpointManager(work_dir)

    # Try to load best checkpoint if requested
    best_ckpt = work_dir / "projector.best.pt"
    if load_best and best_ckpt.exists():
        try:
            metadata = ckpt_manager.load(projector, best_ckpt, strict=True)
            logger.info(f"Loaded best checkpoint with metadata: {metadata}")
        except Exception as e:
            logger.warning(f"Failed to load best checkpoint: {e}")

    # Data loaders
    pin_mem = (device.type == "cuda")
    dl_tr = DataLoader(
        ds_tr, batch_size=config.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_mem,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_list, drop_last=True,
    )
    dl_va = DataLoader(
        ds_va, batch_size=config.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_mem,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_list, drop_last=False,
    )

    # Optimizer and scheduler
    optim = torch.optim.AdamW(
        projector.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    steps_per_epoch = max(1, len(dl_tr))
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(config.warmup_ratio * total_steps)

    sched = CosineWarmup(
        optim,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=config.min_lr
    )

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # Training loop
    best_val = float("inf")
    best_va_metrics = None
    last_ckpt = work_dir / "projector.last.pt"
    bad_epochs = 0
    history = []

    # CSV logging
    csv_path = work_dir / "log.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("epoch,split,loss,t2a@1,a2t@1,t2a@5,a2t@5,t2a@10,a2t@10,lr\n")

    for ep in range(1, config.epochs + 1):
        # Train
        projector.train()
        model.eval()
        tr_metrics = run_epoch(
            model, projector, processor, device, dl_tr,
            optimizer=optim, scaler=scaler,
            temperature=config.temperature,
            clip_grad=config.clip_grad,
        )
        sched.step()

        # Validation
        projector.eval()
        model.eval()
        va_metrics = run_epoch(
            model, projector, processor, device, dl_va,
            optimizer=None, scaler=None,
            temperature=config.temperature,
            clip_grad=0.0,
        )

        # Log metrics
        lr_now = sched.get_lrs()[0]
        with csv_path.open("a", encoding="utf-8") as f:
            f.write(f"{ep},train,{tr_metrics['loss']:.6f},{tr_metrics['t2a@1']:.6f},"
                    f"{tr_metrics['a2t@1']:.6f},{tr_metrics['t2a@5']:.6f},"
                    f"{tr_metrics['a2t@5']:.6f},{tr_metrics['t2a@10']:.6f},"
                    f"{tr_metrics['a2t@10']:.6f},{lr_now:.8f}\n")
            f.write(f"{ep},val,{va_metrics['loss']:.6f},{va_metrics['t2a@1']:.6f},"
                    f"{va_metrics['a2t@1']:.6f},{va_metrics['t2a@5']:.6f},"
                    f"{va_metrics['a2t@5']:.6f},{va_metrics['t2a@10']:.6f},"
                    f"{va_metrics['a2t@10']:.6f},{lr_now:.8f}\n")

        history.append({
            "epoch": ep,
            "train": tr_metrics,
            "val": va_metrics,
            "lr": lr_now
        })

        logger.info(f"[Epoch {ep}] train {tr_metrics['loss']:.4f} | val {va_metrics['loss']:.4f} "
                    f"| R@1 t2a={va_metrics['t2a@1']:.3f} a2t={va_metrics['a2t@1']:.3f}")

        # Save last checkpoint
        ckpt_manager.save(projector, last_ckpt, {"epoch": ep, "metrics": va_metrics})

        # Early stopping check
        if va_metrics["loss"] < best_val - 1e-6:
            best_val = va_metrics["loss"]
            best_va_metrics = va_metrics
            bad_epochs = 0
            ckpt_manager.save(projector, best_ckpt, {"epoch": ep, "metrics": va_metrics})
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                logger.info(f"[EarlyStop] patience={args.patience} reached. Best val={best_val:.4f}")
                break

    # Plot training curves
    try:
        plot_curves(csv_path, work_dir / "curves.png")
    except Exception as e:
        logger.warning(f"Failed to plot curves: {e}")

    return {
        "best_val": float(best_val),
        "best_val_metrics": best_va_metrics,
        "history": history,
        "best_ckpt": str(best_ckpt),
        "last_ckpt": str(last_ckpt),
    }


def plot_curves(csv_path: Path, out_png: Path):
    """Plot training curves"""
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    df_tr = df[df["split"] == "train"]
    df_va = df[df["split"] == "val"]

    # Loss
    axes[0, 0].plot(df_tr["epoch"], df_tr["loss"], label="train")
    axes[0, 0].plot(df_va["epoch"], df_va["loss"], label="val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # R@1 (t2a)
    axes[0, 1].plot(df_tr["epoch"], df_tr["t2a@1"], label="t2a@1 train")
    axes[0, 1].plot(df_va["epoch"], df_va["t2a@1"], label="t2a@1 val")
    axes[0, 1].set_title("Recall@1 text→audio")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # R@5 (a2t)
    axes[1, 0].plot(df_tr["epoch"], df_tr["a2t@5"], label="a2t@5 train")
    axes[1, 0].plot(df_va["epoch"], df_va["a2t@5"], label="a2t@5 val")
    axes[1, 0].set_title("Recall@5 audio→text")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # LR
    axes[1, 1].plot(df_tr["epoch"], df_tr["lr"], label="lr")
    axes[1, 1].set_title("Learning rate")
    axes[1, 1].grid(True, alpha=0.3)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Optuna visualization (clean)
# -----------------------------
def _save_axlike(obj, out_png: Path, figsize=(14, 8)):
    """Save matplotlib figure/axes to file using constrained layout."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    fig = None

    # If it's already a Figure
    if hasattr(obj, "savefig"):
        fig = obj

    # If it's an Axes
    if fig is None and hasattr(obj, "figure"):
        fig = obj.figure

    # If it's an array of Axes
    if fig is None and isinstance(obj, (list, tuple, np.ndarray)):
        flat = np.ravel(obj)
        for it in flat:
            if hasattr(it, "figure"):
                fig = it.figure
                break

    if fig is None:
        fig = plt.gcf()

    # Improve readability
    try:
        fig.set_constrained_layout(True)  # avoid tight_layout warnings
        fig.set_size_inches(*figsize)
        for ax in fig.get_axes():
            ax.tick_params(axis="x", labelrotation=30)
            try:
                ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            except Exception:
                pass
            ax.grid(True, alpha=0.25)
    except Exception:
        pass

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_optuna_figs(study, out_dir: Path, skip_if_single_trial: bool = True, topk_params: int = 4):
    """Save Optuna diagnostic plots with cleaner layouts."""
    import warnings
    from optuna.visualization.matplotlib import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
        plot_slice,
        plot_contour,
        plot_edf,
    )
    try:
        from optuna.importance import get_param_importances
    except Exception:
        get_param_importances = None

    if skip_if_single_trial and len(study.trials) <= 1:
        logging.info("Skipping Optuna plots for single trial")
        return

    # Determine top-K params by importance (if available)
    top_params = None
    if get_param_importances is not None:
        try:
            imps = get_param_importances(study)
            top_params = list(imps.keys())[:topk_params]
        except Exception:
            top_params = None

    plots = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # 1) History
        plots.append(("optuna_history.png", plot_optimization_history(study)))
        # 2) Param importances
        try:
            plots.append(("optuna_param_importances.png", plot_param_importances(study)))
        except Exception as e:
            logging.debug(f"Skipped param_importances: {e}")
        # 3) Parallel coordinate (top-K)
        try:
            if top_params:
                plots.append(("optuna_parallel_coordinate.png", plot_parallel_coordinate(study, params=top_params)))
            else:
                plots.append(("optuna_parallel_coordinate.png", plot_parallel_coordinate(study)))
        except Exception as e:
            logging.debug(f"Skipped parallel_coordinate: {e}")
        # 4) Slice (top-K)
        try:
            if top_params:
                plots.append(("optuna_slice.png", plot_slice(study, params=top_params)))
            else:
                plots.append(("optuna_slice.png", plot_slice(study)))
        except Exception as e:
            logging.debug(f"Skipped slice: {e}")
        # 5) Contour (top-K)
        try:
            if top_params:
                plots.append(("optuna_contour.png", plot_contour(study, params=top_params)))
            else:
                plots.append(("optuna_contour.png", plot_contour(study)))
        except Exception as e:
            logging.debug(f"Skipped contour: {e}")
        # 6) EDF
        try:
            plots.append(("optuna_edf.png", plot_edf(study)))
        except Exception as e:
            logging.debug(f"Skipped edf: {e}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, obj in plots:
        try:
            _save_axlike(obj, out_dir / fname)
        except Exception as e:
            logging.debug(f"Failed to save {fname}: {e}")


# -----------------------------
# Hyperparameter space parsing
# -----------------------------
def parse_suggest_space(spec: str) -> Dict[str, dict]:
    """Parse hyperparameter search space specification"""
    out: Dict[str, dict] = {}

    if not spec or not spec.strip():
        return out

    parts = [p.strip() for p in spec.split(";") if p.strip()]

    for p in parts:
        if ":" not in p:
            continue

        key, rhs = p.split(":", 1)
        key = key.strip()
        rhs = rhs.strip()
        info = {}

        if rhs.startswith("{"):  # categorical
            rhs = rhs.strip("{}")
            choices = [x.strip() for x in rhs.split(",") if x.strip()]

            def cast(v):
                try:
                    if "." in v or "e" in v or "E" in v:
                        return float(v)
                    return int(v)
                except Exception:
                    return v

            info = {"kind": "categorical", "choices": [cast(x) for x in choices]}
        elif rhs.startswith("["):  # continuous range
            span = rhs
            log_flag = False

            if " log" in rhs:
                span, _ = rhs.split(" log", 1)
                log_flag = True

            span = span.strip("[]")
            a, b = [float(x.strip()) for x in span.split(",")]
            info = {"kind": "float", "low": a, "high": b, "log": log_flag}
        else:
            continue

        out[key] = info

    return out


# -----------------------------
# Optuna objective helpers
# -----------------------------
def compute_objective_to_minimize(objective: str, weights: Tuple[float, float],
                                  best_val: float, best_metrics: Optional[Dict[str, float]]) -> float:
    """Compute scalar objective to minimize"""
    if objective == "val_loss":
        return best_val

    if best_metrics is None:
        return best_val

    if objective == "t2a_r1":
        return 1.0 - float(best_metrics.get("t2a@1", 0.0))

    if objective == "a2t_r1":
        return 1.0 - float(best_metrics.get("a2t@1", 0.0))

    if objective == "combo_r1":
        a2t = float(best_metrics.get("a2t@1", 0.0))
        t2a = float(best_metrics.get("t2a@1", 0.0))
        w_a2t, w_t2a = weights
        score = w_a2t * a2t + w_t2a * t2a
        return 1.0 - score

    return best_val


def optuna_objective(trial, args, base_work, ds_tr, ds_va, processor, model, device):
    """Optuna objective function"""

    # Parse user suggest space
    space = parse_suggest_space(args.suggest_space)

    # Helpers (ensure each name is suggested once per trial)
    def suggest_float_once(name, default_low, default_high, log=False):
        if name in space and space[name]["kind"] == "float":
            s = space[name]
            return trial.suggest_float(name, s["low"], s["high"], log=bool(s.get("log", False)))
        return trial.suggest_float(name, default_low, default_high, log=log)

    def suggest_cat_once(name, default_val):
        if name in space and space[name]["kind"] == "categorical":
            return trial.suggest_categorical(name, space[name]["choices"])
        return default_val

    # Sample hyperparameters
    lr = suggest_float_once("lr", 5e-5, 5e-3, log=True)
    temperature = suggest_float_once("temp", 0.04, 0.25, log=False)
    warmup_ratio = suggest_float_once("warmup_ratio", 0.02, 0.2, log=False)
    clip_grad = suggest_float_once("clip_grad", 0.5, 5.0, log=False)
    min_lr = suggest_float_once("min_lr", 1e-7, 1e-5, log=True)
    weight_decay = 1e-5
    if "weight_decay" in space and space["weight_decay"]["kind"] == "float":
        s = space["weight_decay"]
        weight_decay = trial.suggest_float("weight_decay", s["low"], s["high"], log=bool(s.get("log", False)))

    batch = suggest_cat_once("batch", args.batch)
    dropout = 0.0
    if args.adapter == "mlp" and "dropout" in space and space["dropout"]["kind"] == "float":
        s = space["dropout"]
        dropout = trial.suggest_float("dropout", s["low"], s["high"], log=bool(s.get("log", False)))

    # Create fresh projector (no warm-start during HPO to avoid bias)
    projector = build_projector(args.adapter, dropout=(dropout if args.adapter == "mlp" else 0.0)).to(device)

    cfg = TrainConfig(
        epochs=args.epochs,
        batch=int(batch),
        lr=float(lr),
        temperature=float(temperature),
        clip_grad=float(clip_grad),
        warmup_ratio=float(warmup_ratio),
        min_lr=float(min_lr),
        weight_decay=float(weight_decay),
        dropout=float(dropout),
    )

    work_dir = base_work / f"trial_{trial.number:04d}"
    work_dir.mkdir(parents=True, exist_ok=True)

    result = fit_once(args, cfg, work_dir, ds_tr, ds_va, processor, model, projector, device)

    trial.set_user_attr("best_ckpt", result["best_ckpt"])
    trial.set_user_attr("history_json", json.dumps(result["history"]))

    # Compute objective
    w_a2t, w_t2a = args.objective_weights
    value = compute_objective_to_minimize(
        args.objective, (w_a2t, w_t2a), result["best_val"], result["best_val_metrics"]
    )

    return float(value)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)

    # Data arguments
    ap.add_argument("--target_sr", type=int, default=48_000)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--augment", action="store_true")

    # Training arguments
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.0)

    # Adapter
    ap.add_argument("--adapter", choices=["mlp", "external"], default="mlp")
    ap.add_argument("--load_best", action="store_true",
                    help="Load best checkpoint within the current work_dir if available (warm resume).")

    # NEW: Optional warm-start checkpoint path (final training only; not used during Optuna)
    ap.add_argument("--warm_start_ckpt", type=Path, default=None,
                    help="Path to a projector checkpoint to warm-start FINAL training (not used in HPO).")

    # Optuna
    ap.add_argument("--optuna-trials", type=int, default=0,
                    help=">0 to run Optuna HPO")
    ap.add_argument("--timeout-minutes", "--timeout_minutes", type=int, default=None)
    ap.add_argument("--objective", choices=["val_loss", "t2a_r1", "a2t_r1", "combo_r1"],
                    default="combo_r1")
    ap.add_argument("--objective-weights", "--objective_weights",
                    nargs=2, type=float, default=[0.6, 0.4],
                    help="Weights for combo_r1: A2T_WEIGHT T2A_WEIGHT")
    ap.add_argument("--suggest-space", "--suggest_space", type=str, default="",
                    help='e.g. "lr:[1e-4,3e-3] log; weight_decay:[1e-6,1e-3] log; temp:[0.03,0.2]; batch:{16,24,32}"')

    # Optuna resumeable storage
    ap.add_argument("--optuna-storage", type=str, default=None,
                    help='e.g., "sqlite:////abs/path/optuna.db" to enable resume across runs')
    ap.add_argument("--study-name", type=str, default=None,
                    help="Optuna study name (required when using --optuna-storage)")

    args = ap.parse_args()

    # Setup
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup logging
    logger = setup_logging(args.out_dir, "main")
    logger.info(f"Device: {device.type}")

    # Load dataset
    ds_all = PairsDS(
        args.csv,
        max_items=args.max_items,
        target_sr=args.target_sr,
        seconds=args.seconds,
        augment=args.augment
    )

    n = len(ds_all)
    logger.info(f"Loaded {n} pairs from {args.csv}")

    # Split dataset
    idxs = list(range(n))
    random.shuffle(idxs)
    n_val = max(1, int(args.val_frac * n))
    val_idx = set(idxs[:n_val])

    tr_rows, va_rows = [], []
    for i in range(n):
        (tr_rows if i not in val_idx else va_rows).append(ds_all.rows[i])

    # Create split datasets
    ds_tr = PairsDS.__new__(PairsDS)
    ds_tr.rows = tr_rows
    ds_tr.target_sr = ds_all.target_sr
    ds_tr.samples = ds_all.samples
    ds_tr.augment = ds_all.augment

    ds_va = PairsDS.__new__(PairsDS)
    ds_va.rows = va_rows
    ds_va.target_sr = ds_all.target_sr
    ds_va.samples = ds_all.samples
    ds_va.augment = False

    logger.info(f"Split: train={len(ds_tr)}, val={len(ds_va)} (val_frac={args.val_frac:.2f})")

    # Load CLAP model
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)

    # Freeze CLAP
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    # HPO or single run
    if args.optuna_trials > 0:
        if not OPTUNA_OK:
            raise ImportError("Optuna is not installed. Run: pip install optuna")

        # Optuna study directory
        study_dir = args.out_dir / "optuna"
        study_dir.mkdir(parents=True, exist_ok=True)

        # Sampler & pruner
        sampler = optuna.samplers.TPESampler(seed=args.seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0)

        # Storage-aware create/load
        if args.optuna_storage:
            if not args.study_name:
                raise SystemExit("When using --optuna-storage, you must also set --study-name")
            study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
                pruner=pruner,
                storage=args.optuna_storage,
                study_name=args.study_name,
                load_if_exists=True,
            )
            logger.info(f"[Optuna] storage={args.optuna_storage} | study={args.study_name}")
        else:
            study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

        timeout = None
        if args.timeout_minutes is not None and args.timeout_minutes > 0:
            timeout = int(args.timeout_minutes * 60)

        # Optimize
        study.optimize(
            lambda t: optuna_objective(t, args, study_dir, ds_tr, ds_va, processor, model, device),
            n_trials=args.optuna_trials,
            timeout=timeout,
            gc_after_trial=True
        )

        logger.info(f"Optuna best value: {study.best_value:.6f}")
        logger.info(f"Optuna best params: {study.best_params}")

        # Save best results
        (args.out_dir / "optuna_best.json").write_text(
            json.dumps({"value": study.best_value, "params": study.best_params}, indent=2),
            encoding="utf-8"
        )

        # Save Optuna plots (clean styling)
        save_optuna_figs(study, args.out_dir / "optuna_plots",
                         skip_if_single_trial=(args.optuna_trials <= 1), topk_params=4)

        # Load best trial checkpoint (for archival convenience)
        if study.best_trial.user_attrs.get("best_ckpt"):
            best_trial_ckpt = Path(study.best_trial.user_attrs["best_ckpt"])
            if best_trial_ckpt.exists():
                logger.info(f"Loading best checkpoint from trial: {best_trial_ckpt}")
                best = study.best_params
                dropout_val = best.get("dropout", args.dropout)
                projector = build_projector(args.adapter,
                                            dropout=(dropout_val if args.adapter == "mlp" else 0.0)).to(device)
                ckpt_manager = CheckpointManager(args.out_dir)
                ckpt_manager.load_if_exists(projector, best_trial_ckpt, strict=True)
                final_best = args.out_dir / "projector.final.best.pt"
                ckpt_manager.save(projector, final_best,
                                  {"optuna_params": best, "optuna_value": study.best_value})
                logger.info(f"Saved final best checkpoint to {final_best}")

        # Retrain with best params (optionally warm-start from user ckpt)
        best = study.best_params
        cfg = TrainConfig(
            epochs=args.epochs,
            batch=int(best.get("batch", args.batch)),
            lr=float(best.get("lr", args.lr)),
            temperature=float(best.get("temperature", best.get("temp", args.temperature))),
            clip_grad=float(best.get("clip_grad", args.clip_grad)),
            warmup_ratio=float(best.get("warmup_ratio", args.warmup_ratio)),
            min_lr=float(best.get("min_lr", args.min_lr)),
            weight_decay=float(best.get("weight_decay", args.weight_decay)),
            dropout=float(best.get("dropout", args.dropout)),
        )

        projector = build_projector(args.adapter,
                                    dropout=(cfg.dropout if args.adapter == "mlp" else 0.0)).to(device)

        # Optional user warm-start for final training
        if args.warm_start_ckpt:
            cm_final = CheckpointManager(args.out_dir)
            loaded = cm_final.load_if_exists(projector, args.warm_start_ckpt, strict=False)
            logger.info(f"Warm-start from {args.warm_start_ckpt}: {'ok' if loaded else 'not found'}")

        final_dir = args.out_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        result = fit_once(args, cfg, final_dir, ds_tr, ds_va, processor, model,
                          projector, device, load_best=args.load_best)

        (args.out_dir / "final_result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    else:
        # Single training run
        projector = build_projector(args.adapter,
                                    dropout=(args.dropout if args.adapter == "mlp" else 0.0)).to(device)

        # Optional user warm-start for single run
        if args.warm_start_ckpt:
            cm_single = CheckpointManager(args.out_dir)
            loaded = cm_single.load_if_exists(projector, args.warm_start_ckpt, strict=False)
            logging.info(f"Warm-start from {args.warm_start_ckpt}: {'ok' if loaded else 'not found'}")

        cfg = TrainConfig(
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            temperature=args.temperature,
            clip_grad=args.clip_grad,
            warmup_ratio=args.warmup_ratio,
            min_lr=args.min_lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
        )

        result = fit_once(args, cfg, args.out_dir, ds_tr, ds_va, processor, model,
                          projector, device, load_best=args.load_best)

        (args.out_dir / "train_result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()