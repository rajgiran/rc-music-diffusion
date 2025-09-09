# aldm2_retrieval_infer.py
# -*- coding: utf-8 -*-
"""
Retrieval-conditioned inference for AudioLDM2 using a CLAP-audio style bank.

- External adapter (aldm2_style_adapter.ClapAudioProjector) or internal MLP.
- No AudioLDM2.encode_prompt()/LM calls at generation time (avoids GPT2Model issues).
- We pass: prompt_embeds/negative_prompt_embeds (T5 space) + generated_prompt_embeds/negative_generated_prompt_embeds (LM space).
- Style selection; blend modes; metrics; visualizations.

Reporting
-----------------------------------------------------
• Saves plots to metrics/ with the same filenames used in the prompt-only script:
    - metrics/clap_sim_hist.png
    - metrics/sim_vs_loudness.png
    - metrics/spec_grid.png
• report.html layout matches exp1: summary block, overview plots, mel-grid, and per-item panels.
• Optional per-item analysis PNGs under visualizations/ (enabled with --per_item_viz).

Notes
-----
• If you hit "ModuleNotFoundError: No module named 'soxr'" during CLAP import, install it:
    pip install soxr     # or: conda install -c conda-forge soxr-python
• Keep TRANSFORMERS_NO_TORCHVISION=1 in env to avoid torchvision import.
• Tested with: diffusers>=0.31, transformers>=4.42, torch>=2.1
"""

from __future__ import annotations

import os, json, math, argparse, random, platform, logging, time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import soundfile as sf
import librosa
import librosa.display

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm

# ───────────────────────── Logging & env hygiene ──────────────────────────
log = logging.getLogger("aldm2_infer")

def setup_logging(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "console.log"
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt,
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")])

# Minimize extraneous thread usage on shared HPC
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Torch determinism hints
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(False)
except Exception:
    pass

# ───────────────────────── Dependencies (deferred) ────────────────────────
# Defer heavy imports to provide better error messages for missing deps.
try:
    from transformers import ClapModel, ClapProcessor
except ModuleNotFoundError as e:
    if "soxr" in str(e):
        raise ModuleNotFoundError(
            "Missing optional dependency 'soxr'. Install with `pip install soxr` or `conda install -c conda-forge soxr-python`."
        ) from e
    raise

from diffusers import AudioLDM2Pipeline, DPMSolverMultistepScheduler, DDIMScheduler

# ───────────────────────── Optional external adapter ──────────────────────
_has_external = False
ExternalClapAudioProjector = None
ext_blend_text_and_style = None
ext_debug_blend_stats = None
try:
    from aldm_style_adapter import (
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

# ───────────────────────── Fallback blend utils ───────────────────────────
def _l2norm(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def fallback_blend_text_and_style(
    zt: torch.Tensor, zs: torch.Tensor, mode: str = "orth", alpha: float = 0.08, renorm: bool = True
) -> torch.Tensor:
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

def fallback_debug_blend_stats(zt: torch.Tensor, zs: torch.Tensor) -> Tuple[float, float, float]:
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
        )
        # Separate LayerNorm as in the checkpoint
        self.ln = nn.LayerNorm(512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = self.ln(x)
        return F.normalize(x, dim=-1)

# ───────────────────────────── I/O helpers ────────────────────────────────
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

# ──────────────────── CLAP loading (safetensors) ──────────────────────────
def load_clap_safe(device: str, clap_model_id: str = "laion/clap-htsat-unfused"):
    proc = ClapProcessor.from_pretrained(clap_model_id)
    clap = ClapModel.from_pretrained(clap_model_id, use_safetensors=True).to(device).eval()
    return proc, clap

# ──────────────────────────── CLAP utilities ──────────────────────────────
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

# ────────────────────────── Style selection ───────────────────────────────
def select_style_rows(style_bank: np.ndarray, n: int, mode: str = "roundrobin", seed: int = 0) -> np.ndarray:
    """
    Returns [n,512] selected rows.
    """
    rng = np.random.default_rng(seed)
    M = style_bank.shape[0]
    if M == 0:
        raise ValueError("Style bank is empty.")
    if style_bank.shape[1] != 512:
        raise ValueError(f"Style bank must have dim 512, got {style_bank.shape[1]}")

    if mode == "roundrobin":
        idxs = [i % M for i in range(n)]
    elif mode == "random":
        idxs = rng.integers(0, M, size=n).tolist()
    else:  # "index" or fallback: clamp to range
        idxs = [min(i, M - 1) for i in range(n)]
    return style_bank[idxs].astype(np.float32)

# ─────────────────────────── Viz helpers ──────────────────────────────────
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
    freqs = np.fft.rfftfreq(len(audio), 1/sr); fft_vals = np.abs(np.fft.rfft(audio)); idx = freqs > 20
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
    rms = librosa.feature.rms(y=audio)[0]; trms = librosa.times_like(rms)
    axes[2,0].plot(trms, rms); axes[2,0].set_title("RMS over Time"); axes[2,0].grid(True, alpha=0.4)
    # Amplitude histogram
    axes[2,1].hist(audio, bins=50); axes[2,1].set_title("Amplitude Distribution"); axes[2,1].grid(True, alpha=0.4)
    plt.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()

# ───────────────────────── Projector loading ──────────────────────────────
def load_projector(adapter: str, device: str, ckpt_path: Optional[Path] = None, dropout: float = 0.0):
    if adapter == "external":
        if not _has_external:
            raise RuntimeError("Adapter 'external' requested but aldm2_style_adapter not found.")
        proj = ExternalClapAudioProjector().to(device).eval()
    elif adapter == "mlp":
        proj = MLPProjector(dropout=dropout).to(device).eval()
    else:
        raise ValueError(f"Unknown adapter: {adapter}")

    loaded = False
    if ckpt_path:
        try:
            sd = torch.load(ckpt_path, map_location=device)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            if isinstance(sd, dict):
                new_sd = { (k[len("module."): ] if k.startswith("module.") else k): v for k, v in sd.items() }
                missing, unexpected = proj.load_state_dict(new_sd, strict=False)
                loaded = True
                if missing:
                    log.warning(f"Projector missing keys: {missing}")
                if unexpected:
                    log.warning(f"Projector unexpected keys: {unexpected}")
            else:
                log.warning("Projector checkpoint not a state-dict; skipping load.")
        except Exception as e:
            log.warning(f"Failed to load projector weights from {ckpt_path}: {e}")
    return proj, loaded

# ──────────────── CLAP→LM mapping & seq helpers ───────────────
def map_last_to_hidden(x_3d: torch.Tensor, hidden: int, method: str, seed: int) -> torch.Tensor:
    """
    x_3d: (B, S, D_in=512). Map last dim to `hidden` using pad/truncation or a deterministic linear map.
    """
    cur = int(x_3d.shape[-1])
    if cur == hidden:
        return x_3d
    if method == "pad":
        if cur > hidden:
            return x_3d[..., :hidden]
        pad = torch.zeros(*x_3d.shape[:-1], hidden - cur, device=x_3d.device, dtype=x_3d.dtype)
        return torch.cat([x_3d, pad], dim=-1)
    # deterministic linear projection (stateless)
    g = torch.Generator(device=x_3d.device).manual_seed(int(seed))
    W = torch.empty(hidden, cur, device=x_3d.device, dtype=x_3d.dtype)
    with torch.no_grad():
        W.uniform_(-1.0 / math.sqrt(cur), 1.0 / math.sqrt(cur), generator=g)
    return torch.einsum("bsh,hk->bsk", x_3d, W.t())

def match_seq_len(x_3d: torch.Tensor, target_S: int) -> torch.Tensor:
    return x_3d if int(x_3d.shape[1]) == target_S else x_3d.repeat(1, target_S, 1)

# ─────────────────────────── Pipeline helpers ─────────────────────────────
def call_pipe_with_backcompat(pipe: AudioLDM2Pipeline, **kwargs):
    """Call pipeline, dropping unsupported kwargs for older diffusers versions (e.g., negative_attention_mask)."""
    try:
        return pipe(**kwargs)
    except TypeError as e:
        if "negative_attention_mask" in str(e):
            _ = kwargs.pop("negative_attention_mask", None)
            log.warning("Pipeline does not accept negative_attention_mask; retrying without it.")
            return pipe(**kwargs)
        raise

# ───────────────────────────────── Main ───────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", choices=["external", "mlp"], default="external")
    ap.add_argument("--model_id", default="cvssp/audioldm2", help="cvssp/audioldm2 or cvssp/audioldm2-music")

    ap.add_argument("--prompts_file", type=Path, required=True)
    ap.add_argument("--style_bank", type=Path, required=True, help="agg_emb.filtered.npy (float32 [N,512])")
    ap.add_argument("--projector", type=Path, default=None, help="trained projector.pt (recommended)")

    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--blend_mode", choices=["residual","orth","concat"], default="orth")
    ap.add_argument("--alpha_style", type=float, default=0.08)
    ap.add_argument("--style_select", choices=["roundrobin","random","index"], default="roundrobin")
    ap.add_argument("--negative", type=str, default="speech, vocals, singing, rapping, talking, narration, choir, crowd")

    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--guidance", type=float, default=3.2)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["auto","fp16","fp32"], default="fp16")
    ap.add_argument("--scheduler", choices=["dpm","ddim"], default="ddim")

    ap.add_argument("--style2lm_map", choices=["pad","linear"], default="pad",
                    help="Map blended 512-d CLAP → LM hidden (pad/truncate or deterministic linear).")

    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--per_item_viz", action="store_true")
    ap.add_argument("--debug", action="store_true")

    ap.add_argument("--clap_model_id", type=str, default="laion/clap-htsat-unfused")

    args = ap.parse_args()

    setup_logging(args.out_dir)
    log.info("Starting AudioLDM2 retrieval-conditioned inference …")

    # Persist run config
    (args.out_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )

    # Device & dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16 if device == "cuda" else torch.float32
    else:
        dtype = torch.float16 if device == "cuda" else torch.float32

    # Seed
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # I/O dirs
    out_dir = args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = out_dir / "wav"; wav_dir.mkdir(exist_ok=True)
    metrics_dir = out_dir / "metrics"; metrics_dir.mkdir(exist_ok=True)
    viz_dir = out_dir / "visualizations"; viz_dir.mkdir(exist_ok=True)

    # System snapshot
    snapshot = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "diffusers": __import__("diffusers").__version__,
        "transformers": __import__("transformers").__version__,
        "device": device,
    }
    (out_dir / "hardware.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    log.info(f"Env snapshot: {snapshot}")

    # Data
    prompts = read_prompts(args.prompts_file)
    N = len(prompts)
    if N == 0:
        raise ValueError("No prompts loaded.")
    log.info(f"Loaded {N} prompts from {args.prompts_file}")

    # Style bank
    bank = np.load(args.style_bank)
    if bank.dtype != np.float32:
        log.warning(f"Casting style bank from {bank.dtype} to float32")
        bank = bank.astype(np.float32)
    style_rows = select_style_rows(bank, N, mode=args.style_select, seed=args.seed)
    style_rows = style_rows / (np.linalg.norm(style_rows, axis=1, keepdims=True) + 1e-9)

    # CLAP (safetensors)
    proc, clap = load_clap_safe(device, args.clap_model_id)

    # Text CLAP (N,512)
    with torch.no_grad():
        Zt = clap_text_feats(prompts, proc, clap, device)

    # Projector
    projector, loaded = load_projector(adapter=args.adapter, device=device, ckpt_path=args.projector, dropout=0.0)
    log.info(f"Projector weights loaded: {loaded} from {args.projector}")

    # Style → Projected style (N,512) and blend
    with torch.no_grad():
        Za = torch.from_numpy(style_rows).to(device)
        Zs = projector(Za)  # (N,512)
        if _has_external:
            if args.debug:
                tnorm, snorm, diff = ext_debug_blend_stats(Zt, Zs)
                log.info(f"mean||text||={tnorm:.3f} mean||style||={snorm:.3f} Δ={diff:.3f} α={args.alpha_style}")
            Zm = ext_blend_text_and_style(Zt, Zs, mode=args.blend_mode, alpha=args.alpha_style, renorm=True)
        else:
            if args.debug:
                tnorm, snorm, diff = fallback_debug_blend_stats(Zt, Zs)
                log.info(f"mean||text||={tnorm:.3f} mean||style||={snorm:.3f} Δ={diff:.3f} α={args.alpha_style}")
            Zm = fallback_blend_text_and_style(Zt, Zs, mode=args.blend_mode, alpha=args.alpha_style, renorm=True)

    # Pipeline (no LM calls)
    pipe = AudioLDM2Pipeline.from_pretrained(args.model_id, dtype=dtype).to(device)
    if args.scheduler == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    # Hidden sizes
    # LM hidden for generated_prompt_embeds
    lm_cfg = getattr(pipe, "language_model", None)
    lm_hidden = 768
    if lm_cfg is not None and getattr(lm_cfg, "config", None) is not None:
        lm_hidden = getattr(lm_cfg.config, "n_embd", None) or getattr(lm_cfg.config, "hidden_size", 768)
    target_Sg = 1  # sequence length for generated embeds

    # Text hidden for prompt_embeds (T5 encoder)
    txt_hidden = getattr(getattr(pipe, "text_encoder_2", None), "config", None)
    txt_hidden = getattr(txt_hidden, "d_model", None) or getattr(txt_hidden, "hidden_size", 768)
    tok = getattr(pipe, "tokenizer_2", None)
    assert tok is not None and getattr(pipe, "text_encoder_2", None) is not None, "Pipeline lacks tokenizer_2/text_encoder_2."

    gen = torch.Generator(device=device).manual_seed(args.seed)
    sr = getattr(pipe, "audio_sample_rate", 16000)

    # ───────────────────────── Generate ────────────────────────────────────
    B = max(1, int(args.batch))
    audio_files: List[Path] = []

    t0 = time.time()
    for i in tqdm(range(0, N, B), desc="Generating"):
        text_chunk  = prompts[i:i+B]
        bsz = len(text_chunk)

        # 1) prompt_embeds / negative_prompt_embeds (text encoder space)
        max_len = getattr(tok, "model_max_length", None)
        tok_pos = tok(text_chunk, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len).to(device)
        tok_neg = tok([args.negative]*bsz if args.negative else [""]*bsz,
                      return_tensors="pt", padding="max_length", truncation=True, max_length=max_len).to(device)

        with torch.no_grad():
            te_pos = pipe.text_encoder_2(**tok_pos).last_hidden_state  # (b, S, H_text)
            te_neg = pipe.text_encoder_2(**tok_neg).last_hidden_state  # (b, S, H_text)
        attn_mask_pos = tok_pos.attention_mask
        attn_mask_neg = tok_neg.attention_mask

        # 2) generated_prompt_embeds / negative_generated_prompt_embeds (LM space)
        mix_3d = Zm[i:i+bsz].to(device).unsqueeze(1)                                # (b,1,512)
        mix_3d = map_last_to_hidden(mix_3d, lm_hidden, args.style2lm_map, args.seed) # (b,1,H_lm)
        mix_3d = match_seq_len(mix_3d, target_Sg)                                    # (b,Sg,H_lm)

        with torch.no_grad():
            clap_neg = clap_text_feats([""] * bsz, proc, clap, device).unsqueeze(1)    # (b,1,512)
            clap_neg = map_last_to_hidden(clap_neg, lm_hidden, args.style2lm_map, args.seed)  # (b,1,H_lm)
            clap_neg = match_seq_len(clap_neg, target_Sg)

        if args.debug:
            log.info(
                f"prompt_embeds={tuple(te_pos.shape)} neg_prompt_embeds={tuple(te_neg.shape)} "
                f"gen_embeds={tuple(mix_3d.shape)} neg_gen_embeds={tuple(clap_neg.shape)} "
                f"attn_pos={tuple(attn_mask_pos.shape)} attn_neg={tuple(attn_mask_neg.shape)}"
            )

        # 3) Call pipeline with embeddings only (no LM / no encode_prompt)
        with torch.autocast(device_type="cuda",
                            dtype=dtype if device == "cuda" else torch.float32,
                            enabled=(device == "cuda")):
            out = call_pipe_with_backcompat(
                pipe,
                prompt=None, negative_prompt=None,
                prompt_embeds=te_pos,
                negative_prompt_embeds=te_neg,
                attention_mask=attn_mask_pos,
                negative_attention_mask=attn_mask_neg,
                generated_prompt_embeds=mix_3d,
                negative_generated_prompt_embeds=clap_neg,
                audio_length_in_s=float(args.seconds),
                num_inference_steps=int(args.steps),
                guidance_scale=float(args.guidance),
                generator=gen,
            )

        audios = out.audios  # list[np.ndarray]
        for j, aud in enumerate(audios):
            pidx = i + j
            base = safe_slug(prompts[pidx])
            wav = wav_dir / f"{pidx:03d}__{base}.wav"
            save_wav(wav, aud, sr)
            audio_files.append(wav)

    elapsed = time.time() - t0
    log.info(f"Generated {len(audio_files)} waveforms in {elapsed:.1f}s")

    # ─────────────────────────── Metrics ───────────────────────────────────
    sims_text, sims_style, rms_dbfs = None, None, None
    summary = None

    if args.metrics or args.report or args.visualize:
        # CLAP adherence vs. prompts (text) + style
        waves_48k: List[np.ndarray] = []
        for p in audio_files:
            y, sr0 = sf.read(p, dtype="float32", always_2d=False)
            if y.ndim > 1: y = y.mean(axis=1)
            if sr0 != 48000:
                y = librosa.resample(y, orig_sr=sr0, target_sr=48000)
            waves_48k.append(y)
        with torch.no_grad():
            Za_out = clap_audio_feats(waves_48k, 48000, proc, clap, device).cpu().numpy()  # (N,512)
            Zt_np  = Zt[:len(Za_out)].cpu().numpy()
            sims_text  = np.sum(Za_out * Zt_np, axis=-1)
            # style adherence relative to projected style vector
            Zs_np = Zs[:len(Za_out)].cpu().numpy()
            sims_style = np.sum(Za_out * Zs_np, axis=-1)

        # loudness proxy
        rms_dbfs = []
        for p in audio_files:
            y, _ = sf.read(p, dtype="float32", always_2d=False)
            if y.ndim > 1: y = y.mean(axis=1)
            r = 20.0 * np.log10(np.sqrt(np.mean(y**2) + 1e-12) + 1e-12)
            rms_dbfs.append(r)

        # per-item CSV/JSON
        per_rows = []
        for i, (p, pr, s_txt, s_sty, rdb) in enumerate(zip(audio_files, prompts, sims_text, sims_style, rms_dbfs)):
            per_rows.append({
                "index": i, "file": p.name, "prompt": pr,
                "clap_similarity_text": float(s_txt),
                "clap_similarity_style": float(s_sty),
                "rms_dbfs": float(rdb),
            })
        (metrics_dir / "per_item.json").write_text(json.dumps(per_rows, indent=2), encoding="utf-8")
        with (metrics_dir / "per_item.csv").open("w", encoding="utf-8") as f:
            f.write("index,file,prompt,clap_similarity_text,clap_similarity_style,rms_dbfs\n")
            for r in per_rows:
                f.write(f"{r['index']},{r['file']},\"{r['prompt']}\",{r['clap_similarity_text']:.4f},{r['clap_similarity_style']:.4f},{r['rms_dbfs']:.2f}\n")

        # summary metrics
        summary = {
            "n_items": len(audio_files),
            "seconds": args.seconds,
            "steps": args.steps,
            "guidance": args.guidance,
            "elapsed_sec": elapsed,
            "clap_text_mean": float(np.mean(sims_text)),
            "clap_text_std":  float(np.std(sims_text)),
            "clap_style_mean": float(np.mean(sims_style)),
            "clap_style_std":  float(np.std(sims_style)),
            "rms_dbfs_mean": float(np.mean(rms_dbfs)),
            "rms_dbfs_std":  float(np.std(rms_dbfs)),
        }
        (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log.info(f"Metrics summary: {summary}")

    # ──────────────────────── Visualizations (exp1-style) ─────────────────
    # Save with the SAME filenames used by exp1_audioldm2_prompt.py
    if args.visualize or args.report:
        if sims_text is not None:
            plot_hist(np.array(sims_text), metrics_dir / "clap_sim_hist.png",
                      "CLAP Prompt Adherence", "cosine similarity")
            plot_scatter(np.array(rms_dbfs), np.array(sims_text), metrics_dir / "sim_vs_loudness.png",
                         "Prompt Adherence vs Loudness", "RMS (dBFS)", "CLAP similarity")
        plot_spectrogram_grid(audio_files, metrics_dir / "spec_grid.png", max_n=16)

    # Optional per-item analysis PNGs (under visualizations/)
    if args.per_item_viz:
        for i, p in enumerate(audio_files):
            y, srr = sf.read(p, dtype="float32", always_2d=False)
            if y.ndim > 1: y = y.mean(axis=1)
            per_item_analysis_panel(y, srr, viz_dir / f"{i:03d}_analysis.png", title=p.name)

    # ─────────────────────────── HTML report (exp1-matched) ───────────────
    if args.report:
        # Build rows for HTML
        rows = []
        if sims_text is None:
            for i, p in enumerate(audio_files):
                rows.append({
                    "idx": i, "file": p.name, "prompt": prompts[i],
                    "sim": None, "rms": None,
                    "audio_rel": f"wav/{p.name}",
                    "viz_rel": f"visualizations/{i:03d}_analysis.png" if (viz_dir / f"{i:03d}_analysis.png").exists() else None
                })
        else:
            for i, (p, pr, s, rdb) in enumerate(zip(audio_files, prompts, sims_text, rms_dbfs)):
                rows.append({
                    "idx": i, "file": p.name, "prompt": pr,
                    "sim": float(s), "rms": float(rdb),
                    "audio_rel": f"wav/{p.name}",
                    "viz_rel": f"visualizations/{i:03d}_analysis.png" if (viz_dir / f"{i:03d}_analysis.png").exists() else None
                })

        html_head = """<!doctype html><html><head><meta charset="utf-8">
<title>AudioLDM2 Retrieval Report</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;}
h1,h2{margin:8px 0 12px} .row{padding:10px 12px;border:1px solid #eee;border-radius:10px;margin-bottom:10px}
audio{width:100%} .meta{font-size:13px;color:#444}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
.small{font-size:12px;color:#666}
code{background:#f6f8fa;padding:2px 4px;border-radius:4px}
img{max-width:100%}
</style></head><body>"""

        # Summary block: align with exp1 styling/content
        summary_for_html = {
            "model": args.model_id,
            "items": len(prompts),
            "seconds": args.seconds,
            "steps": args.steps,
            "guidance": args.guidance,
            "adapter": args.adapter,
            "blend_mode": args.blend_mode,
            "alpha_style": args.alpha_style,
            "style_select": args.style_select,
            "style2lm_map": args.style2lm_map,
        }
        if summary is not None:
            summary_for_html.update(summary)

        summary_html = f"""
<h1>AudioLDM2 Retrieval Report</h1>
<p class="meta"><b>Model:</b> {args.model_id} |
<b>Items:</b> {len(prompts)} |
<b>Seconds:</b> {args.seconds} |
<b>Steps:</b> {args.steps} |
<b>Guidance:</b> {args.guidance}</p>
<pre>{json.dumps(summary_for_html, indent=2)}</pre>
<h2>Overview Plots</h2>
<div class="grid">
  <div><h3>CLAP similarity histogram</h3><img src="metrics/clap_sim_hist.png"/></div>
  <div><h3>Similarity vs loudness</h3><img src="metrics/sim_vs_loudness.png"/></div>
</div>
<h3>Mel-spectrogram grid (first 16)</h3>
<img src="metrics/spec_grid.png"/>
<h2>Generated Audio</h2>
"""

        rows_html = []
        for r in rows:
            sim_txt = f"{r['sim']:.3f}" if r["sim"] is not None else "NA"
            rms_txt = f"{r['rms']:.2f}" if r["rms"] is not None else "NA"
            viz_img = f'<div><img src="{r["viz_rel"]}"/></div>' if r["viz_rel"] else ""
            rows_html.append(f"""
<div class="row">
  <div class="grid">
    <div>
      <p><b>#{r["idx"]:03d}</b> — <span class="small">{r["file"]}</span></p>
      <p><b>Prompt:</b> {r["prompt"]}</p>
      <audio controls src="{r["audio_rel"]}"></audio>
      <p class="small">CLAP sim: {sim_txt} &nbsp; | &nbsp; RMS dBFS: {rms_txt}</p>
    </div>
    {viz_img}
  </div>
</div>
""")

        html_tail = "</body></html>"
        (out_dir / "report.html").write_text(html_head + summary_html + "\n".join(rows_html) + html_tail, encoding="utf-8")
        log.info(f"[REPORT] {out_dir/'report.html'}")

    log.info(f"DONE — wrote {len(audio_files)} files → {out_dir}")

if __name__ == "__main__":
    main()