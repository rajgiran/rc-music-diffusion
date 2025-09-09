# aldm1_retrieval_infer.py
# -*- coding: utf-8 -*-
"""
AudioLDM1 — Retrieval-conditioned inference (safe cond-only residual mix)

Key points
---------
- Keep ALDM1's own CLAP text embedding (distribution-preserving).
- Mix retrieval style ONLY into the conditional half of CFG:
    cond_new = norm((1-β)*cond_te + β*blend(text_CLAP, style_CLAP))
  or orthogonal/residual variants. Uncond stays unchanged.
- CLI parity with your previous runs (style2lm_map accepted/ignored, style_token_scale, per_item_viz).
- Metrics (CLAP-to-text, CLAP-to-style, RMS), visualizations, exp1-style HTML report.

Tested with:
  torch >= 2.1, transformers >= 4.42, diffusers 0.35.1
"""

from __future__ import annotations
import os, json, math, argparse, random, platform, logging, time, csv, warnings
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import soundfile as sf
import librosa
import librosa.display
from scipy import signal

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm
from transformers import ClapModel, ClapProcessor
from diffusers import AudioLDMPipeline, DPMSolverMultistepScheduler, DDIMScheduler

warnings.filterwarnings("ignore")

# ------------------------- Logging -------------------------
log = logging.getLogger("aldm1_retrieval")
def setup_logging(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "console.log"
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt,
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")])

# Hygiene
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.backends.cudnn.benchmark = False

# --------------------- Optional external adapter ---------------------
_has_external = False
ExternalClapAudioProjector = None
ext_blend_text_and_style = None
ext_debug_blend_stats = None
try:
    from data.aldm_style_adapter import (
        ClapAudioProjector as ExternalClapAudioProjector,
        blend_text_and_style as ext_blend_text_and_style,
        debug_blend_stats as ext_debug_blend_stats,
    )
    _has_external = True
except Exception:
    try:
        from aldm_style_adapter import (
            ClapAudioProjector as ExternalClapAudioProjector,
            blend_text_and_style as ext_blend_text_and_style,
            debug_blend_stats as ext_debug_blend_stats,
        )
        _has_external = True
    except Exception:
        _has_external = False

# --------------------- Fallbacks ---------------------
def _l2norm(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def fallback_blend_text_and_style(zt: torch.Tensor, zs: torch.Tensor,
                                  mode: str = "orth", alpha: float = 0.08, renorm: bool = True) -> torch.Tensor:
    # zt, zs in CLAP space (512-d), normalized
    if mode == "orth":
        proj = (zs * zt).sum(dim=-1, keepdim=True) * zt
        mixed = zt + alpha * (zs - proj)
    elif mode == "residual":
        mixed = zt + alpha * zs
    elif mode == "concat":
        mixed = (1 - alpha) * zt + alpha * zs
    else:
        mixed = zt
    return _l2norm(mixed) if renorm else mixed

def fallback_debug_blend_stats(zt: torch.Tensor, zs: torch.Tensor) -> Tuple[float,float,float]:
    return float(zt.norm(dim=-1).mean()), float(zs.norm(dim=-1).mean()), float((zt - zs).norm(dim=-1).mean())

# ───────────────────────── Internal MLP projector ─────────────────────────
class MLPProjector(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        # Match the architecture from your checkpoint (1024 hidden dim)
        self.net = nn.Sequential(
            nn.Linear(512, 1024),  # net.0: 512 -> 1024
            nn.GELU(),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(1024, 512),  # net.3: 1024 -> 512
            nn.LayerNorm(512)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return F.normalize(x, dim=-1)

# --------------------- I/O ---------------------
def read_prompts(p: Path) -> List[str]:
    txt = p.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.replace("\ufeff", "").strip() for ln in txt.splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]

def save_wav(path: Path, x: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=np.float32).squeeze()
    x = np.clip(x, -1.0, 1.0)
    sf.write(str(path), x, sr)

def safe_slug(s: str, maxlen=60) -> str:
    s = s[:maxlen]
    return "".join(c if (c.isalnum() or c in " _-") else "_" for c in s).strip().replace(" ", "_")

# --------------------- Audio Analysis ---------------------
def dbfs(x: np.ndarray, eps: float = 1e-12) -> float:
    rms = np.sqrt(np.mean(x**2) + eps)
    return 20.0 * np.log10(rms + eps)

def calculate_rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio**2)))

def calculate_crest_factor(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio)))
    rms = calculate_rms(audio)
    return float(peak / rms) if rms > 0 else 0.0

def calculate_zero_crossing_rate(audio: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2.0)

def calculate_spectral_centroid(audio: np.ndarray, sr: int) -> float:
    mag = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0/sr)
    denom = np.sum(mag)
    return float(np.sum(mag * freqs) / denom) if denom > 0 else 0.0

def calculate_spectral_rolloff(audio: np.ndarray, sr: int, percentile: float = 0.85) -> float:
    mag = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0/sr)
    csum = np.cumsum(mag)
    if csum[-1] <= 0: return 0.0
    thr = csum[-1] * percentile
    idx = np.searchsorted(csum, thr)
    idx = min(idx, len(freqs)-1)
    return float(freqs[idx])

def calculate_snr(audio: np.ndarray) -> float:
    sig_power = float(np.mean(audio**2))
    noise = audio - signal.medfilt(audio, kernel_size=5)
    noise_power = float(np.mean(noise**2))
    return float(10.0 * np.log10(sig_power / noise_power)) if noise_power > 0 else 100.0

def calculate_dynamic_range(audio: np.ndarray) -> float:
    rms = calculate_rms(audio)
    peak = float(np.max(np.abs(audio)))
    return float(20.0 * np.log10(peak / (rms + 1e-12))) if rms > 0 else 0.0

def analyze_audio_quality(audio: np.ndarray, sr: int) -> dict:
    return {
        "rms": calculate_rms(audio),
        "crest_factor": calculate_crest_factor(audio),
        "zero_crossing_rate": calculate_zero_crossing_rate(audio),
        "spectral_centroid": calculate_spectral_centroid(audio, sr),
        "spectral_rolloff": calculate_spectral_rolloff(audio, sr),
        "snr_db": calculate_snr(audio),
        "dynamic_range_db": calculate_dynamic_range(audio),
        "duration_seconds": float(len(audio) / sr),
        "sample_rate": int(sr),
        "rms_dbfs": dbfs(audio),
        "max_amp": float(np.max(np.abs(audio))),
    }

def per_item_analysis_panel(audio: np.ndarray, sr: int, out_png: Path, title: str = ""):
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    if title: fig.suptitle(title, fontsize=14)
    # Waveform
    t = np.linspace(0, len(audio)/sr, num=len(audio))
    axes[0,0].plot(t, audio)
    axes[0,0].set_title("Waveform"); axes[0,0].set_xlabel("Time (s)"); axes[0,0].set_ylabel("Amp"); axes[0,0].grid(True)
    # Mel spectrogram
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=min(8000, sr//2))
    Sdb = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(Sdb, sr=sr, x_axis='time', y_axis='mel', ax=axes[0,1])
    axes[0,1].set_title("Mel Spectrogram")
    fig.colorbar(img, ax=axes[0,1], format="%+2.0f dB")
    # Log magnitude spectrum
    freqs = np.fft.rfftfreq(len(audio), 1/sr)
    fft_vals = np.abs(np.fft.rfft(audio))
    idx = freqs > 20
    axes[1,0].semilogx(freqs[idx], 20*np.log10(fft_vals[idx] + 1e-10))
    axes[1,0].set_title("Frequency Spectrum"); axes[1,0].set_xlabel("Hz"); axes[1,0].set_ylabel("dB"); axes[1,0].grid(True, alpha=0.4)
    # Centroid + rolloff over time
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    times = librosa.times_like(centroid)
    axes[1,1].plot(times, centroid, label="Centroid")
    axes[1,1].plot(times, rolloff, label="Rolloff")
    axes[1,1].set_title("Spectral Features over Time"); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.4)
    # RMS over time
    rms = librosa.feature.rms(y=audio)[0]
    trms = librosa.times_like(rms)
    axes[2,0].plot(trms, rms); axes[2,0].set_title("RMS over Time"); axes[2,0].grid(True, alpha=0.4)
    # Amplitude histogram
    axes[2,1].hist(audio, bins=50)
    axes[2,1].set_title("Amplitude Distribution"); axes[2,1].grid(True, alpha=0.4)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

# --------------------- CLAP ---------------------
def load_clap_safe(device: str, clap_model_id: str = "laion/clap-htsat-unfused"):
    proc = ClapProcessor.from_pretrained(clap_model_id)
    clap = ClapModel.from_pretrained(clap_model_id, use_safetensors=True).to(device).eval()
    return proc, clap

@torch.no_grad()
def clap_text_feats(texts: List[str], proc: ClapProcessor, clap: ClapModel, device: str) -> torch.Tensor:
    ins = proc(text=texts, return_tensors="pt", padding=True).to(device)
    z = clap.get_text_features(**ins)
    return F.normalize(z, dim=-1)

@torch.no_grad()
def clap_audio_feats(waves: List[np.ndarray], sr: int, proc: ClapProcessor, clap: ClapModel, device: str) -> torch.Tensor:
    ins = proc(audios=[w.astype(np.float32) for w in waves], sampling_rate=sr,
               return_tensors="pt", padding=True).to(device)
    z = clap.get_audio_features(**ins)
    return F.normalize(z, dim=-1)

# --------------------- Style selection ---------------------
def select_style_rows(style_bank: np.ndarray, n: int, mode: str = "roundrobin", seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = style_bank.shape[0]
    if M == 0:
        raise ValueError("Style bank is empty.")
    if style_bank.shape[1] != 512:
        raise ValueError(f"Style bank must be [*,512], got {style_bank.shape}")
    if mode == "roundrobin":
        idxs = [i % M for i in range(n)]
    elif mode == "random":
        idxs = rng.integers(0, M, size=n).tolist()
    else:
        idxs = [min(i, M-1) for i in range(n)]
    return style_bank[idxs].astype(np.float32)

# --------------------- Viz ---------------------
def plot_hist(vals: np.ndarray, out_png: Path, title: str, xlabel: str):
    plt.figure(figsize=(6,4))
    plt.hist(vals, bins=24)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("count")
    plt.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150); plt.close()

def plot_scatter(x: np.ndarray, y: np.ndarray, out_png: Path, title: str, xlabel: str, ylabel: str):
    plt.figure(figsize=(6,4))
    plt.scatter(x, y, s=18)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150); plt.close()

def plot_spectrogram_grid(files: List[Path], out_png: Path, max_n: int = 16):
    n = min(max_n, len(files))
    cols = 4; rows = math.ceil(n/cols)
    plt.figure(figsize=(4*cols, 3*rows))
    for i in range(n):
        y, sr = sf.read(files[i], dtype="float32", always_2d=False)
        if y.ndim > 1: y = y.mean(axis=1)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=sr//2)
        Sdb = librosa.power_to_db(S, ref=np.max)
        ax = plt.subplot(rows, cols, i+1)
        librosa.display.specshow(Sdb, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        ax.set_title(files[i].stem, fontsize=8); ax.set_xlabel(""); ax.set_ylabel("")
    plt.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150); plt.close()

def plot_spec_single(y: np.ndarray, sr: int, out_png: Path, title: str = ""):
    plt.figure(figsize=(4,3))
    if y.ndim > 1: y = y.mean(axis=1)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=sr//2)
    Sdb = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(Sdb, sr=sr, x_axis='time', y_axis='mel')
    if title: plt.title(title, fontsize=9)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=120); plt.close()

# --------------------- Projector ---------------------
def load_projector(adapter: str, device: str, ckpt_path: Optional[Path] = None, dropout: float = 0.0):
    if adapter == "external":
        if not _has_external:
            raise RuntimeError("Adapter 'external' requested but aldm2_style_adapter not found.")
        proj = ExternalClapAudioProjector().to(device).eval()
    elif adapter == "mlp":
        proj = MLPProjector(dropout=dropout).to(device).eval()
    elif adapter == "none":
        proj = nn.Identity().to(device).eval()
    else:
        raise ValueError(f"Unknown adapter: {adapter}")

    loaded = False
    if ckpt_path and adapter != "none":
        try:
            sd = torch.load(ckpt_path, map_location=device)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            if isinstance(sd, dict):
                new_sd = { (k[len("module."): ] if k.startswith("module.") else k): v for k, v in sd.items() }
                proj.load_state_dict(new_sd, strict=False)
                loaded = True
        except Exception as e:
            log.warning(f"Failed to load projector weights from {ckpt_path}: {e}")
    return proj, loaded

# --------------------- Report (exp1 look) ---------------------
def write_report_like_exp1(out_dir: Path, args, audio_files: List[Path], prompts: List[str],
                           sims_text: Optional[np.ndarray],
                           sims_style: Optional[np.ndarray],
                           rms_dbfs: Optional[List[float]],
                           per_item_specs: Optional[List[Path]] = None,
                           summary: Optional[dict] = None):
    # Put the report at out_dir so relative paths to WAVs are trivial
    per_rows = []
    for i, (p, pr, s, rdb, ss) in enumerate(zip(audio_files, prompts, sims_text, rms_dbfs, sims_style)):
        audio_rel = f"wav/{p.name}"  # audio in wav/ subfolder
        per_viz = out_dir / "visualizations" / f"{i:03d}_analysis.png"
        viz_rel = f"visualizations/{per_viz.name}" if per_viz.exists() else None
        per_rows.append({
            "idx": i,
            "file": p.name,
            "prompt": pr,
            "sim_text": float(s),
            "sim_style": float(ss),
            "rms_dbfs": float(rdb),
            "audio_rel": audio_rel,
            "viz_rel": viz_rel,
        })

    html_head = """<!doctype html><html><head><meta charset="utf-8">
<title>AudioLDM1 Retrieval Report</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;}
h1,h2{margin:8px 0 12px} .row{padding:10px 12px;border:1px solid #eee;border-radius:10px;margin-bottom:10px}
audio{width:100%} .meta{font-size:13px;color:#444}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
.small{font-size:12px;color:#666}
code{background:#f6f8fa;padding:2px 4px;border-radius:4px}
img{max-width:100%}
</style></head><body>"""

    header_line = (
        f"<p class=\"meta\"><b>model:</b> {args.model_id} &nbsp;|&nbsp; "
        f"<b>steps:</b> {args.steps} &nbsp;|&nbsp; "
        f"<b>guidance:</b> {args.guidance} &nbsp;|&nbsp; "
        f"<b>seconds:</b> {args.seconds} &nbsp;|&nbsp; "
        f"<b>blend:</b> {args.blend_mode} α={args.alpha_style} &nbsp;|&nbsp; "
        f"<b>adapter:</b> {args.adapter} &nbsp;|&nbsp; "
        f"<b>style2lm_map:</b> {args.style2lm_map} &nbsp;|&nbsp; "
        f"<b>style_token_scale:</b> {args.style_token_scale} &nbsp;|&nbsp; "
        f"<b>cond_mix:</b> {args.cond_mix} β={args.cond_mix_beta}</p>"
    )

    summary_html = f"""
<h1>AudioLDM1 Retrieval-Conditioned Report</h1>
{header_line}
<pre>{json.dumps(summary, indent=2)}</pre>
<h2>Overview Plots</h2>
<div class="grid">
  <div><h3>CLAP similarity to Text</h3><img src="metrics/clap_sim_text_hist.png"/></div>
  <div><h3>CLAP similarity to Style</h3><img src="metrics/clap_sim_style_hist.png"/></div>
  <div><h3>Similarity vs loudness</h3><img src="metrics/sim_vs_loudness.png"/></div>
</div>
<h3>Mel-spectrogram grid (first 16)</h3>
<img src="metrics/spec_grid.png"/>
<h2>Generated Audio</h2>
"""

    rows_html = []
    for r in per_rows:
        viz_img = f'<div><img src="{r["viz_rel"]}"/></div>' if r["viz_rel"] else ""
        rows_html.append(f"""
<div class="row">
  <div class="grid">
    <div>
      <p><b>#{r["idx"]:03d}</b> — <span class="small">{r["file"]}</span></p>
      <p><b>Prompt:</b> {r["prompt"]}</p>
      <audio controls src="{r["audio_rel"]}"></audio>
      <p class="small">CLAP (to text): {r["sim_text"]:.3f} &nbsp; | &nbsp; 
      CLAP (to style): {r["sim_style"]:.3f} &nbsp; | &nbsp; 
      RMS dBFS: {r["rms_dbfs"]:.2f}</p>
    </div>
    {viz_img}
  </div>
</div>
""")
    html_tail = "</body></html>"
    (out_dir / "report.html").write_text(html_head + summary_html + "\n".join(rows_html) + html_tail, encoding="utf-8")
    log.info(f"HTML report → {out_dir/'report.html'}")

# --------------------- Main ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", choices=["external","mlp","none"], default="external",
                    help="Projector to map 512→512 (external/mlp) or identity (none).")
    ap.add_argument("--projector", type=Path, default=None)

    ap.add_argument("--model_id", default="cvssp/audioldm")
    ap.add_argument("--prompts_file", type=Path, required=True)
    ap.add_argument("--style_bank", type=Path, required=True, help="agg_emb.filtered.npy [*,512]")
    ap.add_argument("--out_dir", type=Path, required=True)

    ap.add_argument("--blend_mode", choices=["residual","orth","concat"], default="orth")
    ap.add_argument("--alpha_style", type=float, default=0.06)
    ap.add_argument("--style_select", choices=["roundrobin","random","index"], default="roundrobin")
    ap.add_argument("--negative", type=str,
                    default="speech, vocals, singing, rapping, talking, narration, choir, crowd, whisper, breath")

    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["auto","fp16","fp32"], default="fp16")
    ap.add_argument("--scheduler", choices=["dpm","ddim"], default="ddim")

    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--debug", action="store_true")

    ap.add_argument("--clap_model_id", type=str, default="laion/clap-htsat-unfused")

    # CLI compatibility + control
    ap.add_argument("--style2lm_map", choices=["pad","linear"], default="pad",
                    help="ALDM1 uses CLAP 512 directly. Accepted for CLI compatibility; ignored.")
    ap.add_argument("--style_token_scale", type=float, default=1.0,
                    help="Scales style influence: alpha_eff = alpha_style * style_token_scale.")
    ap.add_argument("--per_item_viz", action="store_true",
                    help="Include per-item mel images in the report.")

    # NEW: safer injection controls
    ap.add_argument("--cond_mix", choices=["residual","orth","replace"], default="residual",
                    help="How to mix retrieval into the conditional half of CFG.")
    ap.add_argument("--cond_mix_beta", type=float, default=0.18,
                    help="Mix strength β for cond-only fusion (recommended 0.10–0.25).")

    args = ap.parse_args()
    setup_logging(args.out_dir)
    log.info("Starting AudioLDM1 retrieval-conditioned inference …")

    if args.style2lm_map != "pad":
        log.warning("ALDM1 uses 512-d CLAP directly; --style2lm_map is ignored (set to 'pad').")

    # Persist config
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    # Device/dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16 if device == "cuda" else torch.float32
    else:
        dtype = torch.float16 if device == "cuda" else torch.float32

    # Snapshot
    snapshot = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "diffusers": __import__("diffusers").__version__,
        "transformers": __import__("transformers").__version__,
        "device": device,
    }
    (args.out_dir / "hardware.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    log.info(f"Env snapshot: {snapshot}")

    # Data
    prompts = read_prompts(args.prompts_file)
    N = len(prompts)
    if N == 0:
        raise ValueError("No prompts loaded.")
    log.info(f"Loaded {N} prompts from {args.prompts_file}")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # Style bank
    bank = np.load(args.style_bank)
    if bank.dtype != np.float32:
        log.warning(f"Casting style bank from {bank.dtype} to float32")
        bank = bank.astype(np.float32)
    style_rows = select_style_rows(bank, N, mode=args.style_select, seed=args.seed)
    style_rows = style_rows / (np.linalg.norm(style_rows, axis=1, keepdims=True) + 1e-9)

    # CLAP
    proc, clap = load_clap_safe(device, args.clap_model_id)
    with torch.no_grad():
        Zt = clap_text_feats(prompts, proc, clap, device)  # (N,512)

    # Projector + style blend
    projector, loaded = load_projector(args.adapter, device, args.projector, dropout=0.0)
    log.info(f"Projector loaded: {loaded}")
    with torch.no_grad():
        Za = torch.from_numpy(style_rows).to(device)      # (N,512)
        Zs = projector(Za) if args.adapter != "none" else Za
        alpha_eff = float(args.alpha_style) * float(args.style_token_scale)
        if _has_external:
            if args.debug:
                t,s,d = ext_debug_blend_stats(Zt, Zs); log.info(f"||t||={t:.3f} ||s||={s:.3f} Δ={d:.3f} α_eff={alpha_eff:.3f}")
            Zm = ext_blend_text_and_style(Zt, Zs, mode=args.blend_mode, alpha=alpha_eff, renorm=True)
        else:
            if args.debug:
                t,s,d = fallback_debug_blend_stats(Zt, Zs); log.info(f"||t||={t:.3f} ||s||={s:.3f} Δ={d:.3f} α_eff={alpha_eff:.3f}")
            Zm = fallback_blend_text_and_style(Zt, Zs, mode=args.blend_mode, alpha=alpha_eff, renorm=True)

    # Pipeline
    pipe = AudioLDMPipeline.from_pretrained(args.model_id, dtype=dtype).to(device)
    if args.scheduler == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    gen = torch.Generator(device=device).manual_seed(args.seed)
    sr = getattr(pipe, "audio_sample_rate", 16000)

    log.info("Text hidden size (CLAP): 512")

    # -------------------- Generate (safe cond-only mix) --------------------
    B = max(1, int(args.batch))
    audio_files: List[Path] = []
    wav_dir = args.out_dir / "wav"; wav_dir.mkdir(exist_ok=True)
    metrics_dir = args.out_dir / "metrics"; metrics_dir.mkdir(exist_ok=True)
    viz_dir = args.out_dir / "visualizations"; viz_dir.mkdir(exist_ok=True)
    per_item_dir = viz_dir / "spec_items"; per_item_dir.mkdir(parents=True, exist_ok=True) if args.per_item_viz else None

    t0 = time.time()
    for i in tqdm(range(0, N, B), desc="Generating"):
        text_chunk = prompts[i:i+B]
        bsz = len(text_chunk)

        # 1) get native ALDM1 CLAP embeddings (already handles CFG duplication)
        enc = pipe._encode_prompt(
            prompt=text_chunk,
            device=device,
            num_waveforms_per_prompt=1,
            do_classifier_free_guidance=(args.guidance > 1.0),
            negative_prompt=[args.negative]*bsz if args.negative else [""]*bsz,
        )
        # Some versions return Tensor; some (older) might return (embeds, mask)
        if isinstance(enc, tuple):
            text_embeds = enc[0]
        else:
            text_embeds = enc
        # shape: (bsz) or (2*bsz, 512)
        if args.guidance > 1.0:
            uncond_te = text_embeds[:bsz]
            cond_te   = text_embeds[bsz:]
        else:
            uncond_te = None
            cond_te   = text_embeds

        # 2) prepare retrieval mix target (CLAP 512)
        mix_target = Zm[i:i+bsz].to(device)  # (bsz,512)

        # 3) cond-only fusion (distribution-preserving)
        beta = float(args.cond_mix_beta)
        mode = args.cond_mix
        if mode == "replace":
            cond_new = mix_target
        elif mode == "orth":
            proj = (mix_target * cond_te).sum(dim=-1, keepdim=True) * cond_te
            cond_new = _l2norm(cond_te + beta * (mix_target - proj))
        else:  # residual (default)
            cond_new = _l2norm((1.0 - beta) * cond_te + beta * mix_target)

        if args.guidance > 1.0:
            new_prompt_embeds = torch.cat([uncond_te, cond_new], dim=0)
        else:
            new_prompt_embeds = cond_new

        # 4) call pipeline with embeddings only (no masks for ALDM1)
        with torch.autocast(device_type="cuda",
                            dtype=dtype if device == "cuda" else torch.float32,
                            enabled=(device == "cuda")):
            out = pipe(
                prompt=None,
                negative_prompt=None,
                prompt_embeds=new_prompt_embeds,
                audio_length_in_s=float(args.seconds),
                num_inference_steps=int(args.steps),
                guidance_scale=float(args.guidance),
                generator=gen,
            )

        audios = out.audios  # list[np.ndarray]
        if args.guidance > 1.0 and len(audios) == 2 * bsz:
            audios = audios[bsz:]  # keep cond half only
        elif len(audios) != bsz:
            log.warning(f"Unexpected audio count: got {len(audios)} for batch {bsz}; "
                        f"keeping last {bsz}.")
            audios = audios[-bsz:]  # fallback: last bsz items

        for j, aud in enumerate(audios):
            pidx = i + j
            if pidx >= len(prompts):
                # Extra safety for odd pipeline behavior
                log.warning(f"pidx {pidx} >= num_prompts {len(prompts)}; skipping save.")
                continue
            base = safe_slug(prompts[pidx])
            wav = wav_dir / f"{pidx:03d}__{base}.wav"
            save_wav(wav, aud, sr)
            audio_files.append(wav)

            if args.per_item_viz:
                try:
                    plot_spec_single(aud, sr, per_item_dir / f"{pidx:03d}.png", title="")
                except Exception as e:
                    log.warning(f"Per-item viz failed for #{pidx}: {e}")

    log.info(f"Generated {len(audio_files)} waveforms in {time.time()-t0:.1f}s")

    # -------------------- Metrics --------------------
    sims_text = sims_style = None
    rms_dbfs: Optional[List[float]] = None
    durations = []; max_amps = []; zcrs = []
    crest = []; centroid = []; rolloff = []; snr_db = []; dyn_rng = []
    per_item_paths = []
    
    if args.metrics or args.visualize or args.report:
        waves_48k: List[np.ndarray] = []
        for p in audio_files:
            y, sr0 = sf.read(p, dtype="float32", always_2d=False)
            if y.ndim > 1: y = y.mean(axis=1)
            if sr0 != 48000:
                y = librosa.resample(y, orig_sr=sr0, target_sr=48000)
            waves_48k.append(y)
        with torch.no_grad():
            Za_out = clap_audio_feats(waves_48k, 48000, proc, clap, device).cpu().numpy()
            Zt_np  = Zt[:len(Za_out)].cpu().numpy()
            Zs_np  = Zs[:len(Za_out)].cpu().numpy()
            sims_text  = np.sum(Za_out * Zt_np, axis=-1)
            sims_style = np.sum(Za_out * Zs_np, axis=-1)

        # Audio quality metrics
        for idx, p in enumerate(audio_files):
            y, srr = sf.read(p, dtype="float32", always_2d=False)
            if y.ndim > 1: y = y.mean(axis=1)
            m = analyze_audio_quality(y, srr)
            durations.append(m["duration_seconds"]); max_amps.append(m["max_amp"])
            zcrs.append(m["zero_crossing_rate"])
            crest.append(m["crest_factor"]); centroid.append(m["spectral_centroid"])
            rolloff.append(m["spectral_rolloff"]); snr_db.append(m["snr_db"])
            dyn_rng.append(m["dynamic_range_db"])
            
            if args.per_item_viz:
                pp = viz_dir / f"{idx:03d}_analysis.png"
                per_item_analysis_panel(y, srr, pp, title=audio_files[idx].name)
                per_item_paths.append(pp)

        rms_dbfs = []
        for p in audio_files:
            y, _ = sf.read(p, dtype="float32", always_2d=False)
            if y.ndim > 1: y = y.mean(axis=1)
            r = 20.0 * np.log10(np.sqrt(np.mean(y**2) + 1e-12) + 1e-12)
            rms_dbfs.append(r)

        # diversity from audio embeddings
        Za_np = Za_out
        K = Za_np @ Za_np.T  # cosine sim
        N = K.shape[0]
        triu = K[np.triu_indices(N, k=1)]
        diversity = 1.0 - triu  # cosine distance
        div_mean = float(np.mean(diversity)) if diversity.size else 0.0
        div_p10 = float(np.percentile(diversity, 10)) if diversity.size else 0.0
        div_p90 = float(np.percentile(diversity, 90)) if diversity.size else 0.0

        rows = []
        for idx, (p, pr, st, ss, rdb, d, m, z, c, ce, ro, sn, dr) in enumerate(zip(
            audio_files, prompts, sims_text, sims_style, rms_dbfs, durations, max_amps, 
            zcrs, crest, centroid, rolloff, snr_db, dyn_rng
        )):
            rows.append({
                "index": idx, "file": p.name, "prompt": pr,
                "clap_similarity_text": float(st),
                "clap_similarity_style": float(ss),
                "rms_dbfs": float(rdb),
                "duration_seconds": float(d),
                "max_amp": float(m),
                "zero_crossing_rate": float(z),
                "crest_factor": float(c),
                "spectral_centroid": float(ce),
                "spectral_rolloff": float(ro),
                "snr_db": float(sn),
                "dynamic_range_db": float(dr),
            })
        (metrics_dir / "per_item.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        with (metrics_dir / "per_item.csv").open("w", encoding="utf-8") as f:
            f.write("index,file,prompt,clap_similarity_text,clap_similarity_style,rms_dbfs,duration_s,max_amp,zcr,crest,centroid,rolloff,snr_db,dyn_range_db\n")
            for r in rows:
                f.write(f"{r['index']},{r['file']},\"{r['prompt']}\",{r['clap_similarity_text']:.4f},{r['clap_similarity_style']:.4f},{r['rms_dbfs']:.2f},{r['duration_seconds']:.3f},{r['max_amp']:.4f},{r['zero_crossing_rate']:.4f},{r['crest_factor']:.3f},{r['spectral_centroid']:.1f},{r['spectral_rolloff']:.1f},{r['snr_db']:.2f},{r['dynamic_range_db']:.2f}\n")

        summary = {
            "n_items": len(audio_files),
            "clap_text_mean": float(np.mean(sims_text)),
            "clap_text_std":  float(np.std(sims_text)),
            "clap_style_mean": float(np.mean(sims_style)),
            "clap_style_std":  float(np.std(sims_style)),
            "rms_dbfs_mean": float(np.mean(rms_dbfs)),
            "rms_dbfs_std":  float(np.std(rms_dbfs)),
            "diversity_mean_cosdist": div_mean,
            "diversity_p10": div_p10,
            "diversity_p90": div_p90,
            "max_amp_mean": float(np.mean(max_amps)) if max_amps else 0.0,
            "snr_db_mean": float(np.mean(snr_db)) if snr_db else 0.0,
            "dyn_range_db_mean": float(np.mean(dyn_rng)) if dyn_rng else 0.0,
        }
        (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log.info(f"Metrics summary: {summary}")

    # -------------------- Plots & Report --------------------
    if args.visualize or args.report:
        if sims_text is not None:
            plot_hist(np.array(sims_text), metrics_dir / "clap_sim_text_hist.png", "CLAP Prompt Adherence", "cosine similarity (to text)")
        if sims_style is not None:
            plot_hist(np.array(sims_style), metrics_dir / "clap_sim_style_hist.png", "CLAP Style Adherence", "cosine similarity (to style)")
        if rms_dbfs is not None and sims_text is not None:
            plot_scatter(np.array(rms_dbfs), sims_text, metrics_dir / "sim_vs_loudness.png",
                         "Prompt Adherence vs Loudness", "RMS (dBFS)", "CLAP similarity")
        plot_spectrogram_grid(audio_files, metrics_dir / "spec_grid.png", max_n=16)

    if args.report:
        write_report_like_exp1(args.out_dir, args, audio_files, prompts, sims_text, sims_style, rms_dbfs, per_item_paths, summary)

    log.info(f"DONE — wrote {len(audio_files)} files → {args.out_dir}")

if __name__ == "__main__":
    main()