#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp2a: AudioLDM1 with retrieval conditioning + comprehensive metrics, visualizations, and HTML report.
Two modes:
  A) --mode augment  (RECOMMENDED)
     Use retrieval to derive style words (from neighbors.jsonl and a simple path→tags map),
     append them to each text prompt, and generate as normal.
  B) --mode embed
     Blend CLAP text embeddings with precomputed style embeddings (agg_emb.npy),
     then call AudioLDMPipeline with prompt_embeds / negative_prompt_embeds.
     Defaults are conservative to avoid noise (alpha_style small, renorm on).
Outputs:
  - wav/, plots/, metrics/{metrics.csv, summary.json}, report.html
  - Comprehensive metrics: CLAP adherence, diversity, audio quality
  - Visualizations: histograms, scatter plots, mel-spectrogram grids, per-item analysis
"""
from __future__ import annotations
import os, csv, math, json, time, argparse, random, warnings
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import soundfile as sf
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa, librosa.display
from scipy import signal
from tqdm import tqdm
from diffusers import AudioLDMPipeline, DPMSolverMultistepScheduler
from transformers import ClapModel, ClapProcessor
warnings.filterwarnings("ignore")

# ---------------------- I/O helpers ----------------------
def convert_numpy_types(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_numpy_types(x) for x in obj]
    return obj

def read_prompts(p: Path) -> List[str]:
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]

def read_jsonl(p: Path) -> List[dict]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows

def save_wav(path: Path, x: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=np.float32).squeeze()
    x = np.clip(x, -1.0, 1.0)
    sf.write(str(path), x, sr)

def safe_slug(s: str, maxlen=60) -> str:
    s = s[:maxlen]
    return "".join(c if (c.isalnum() or c in " _-") else "_" for c in s).strip().replace(" ", "_")

# ---------------------- comprehensive audio analysis ----------------------
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

def spectral_flatness(x: np.ndarray) -> float:
    n = 2048
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    seg = x[:n] * np.hanning(n)
    p = np.abs(np.fft.rfft(seg))**2 + 1e-12
    gm = np.exp(np.mean(np.log(p)))
    am = np.mean(p)
    return float(gm / am)

def non_silence_ratio(x: np.ndarray, thr: float = 0.02) -> float:
    return float(np.mean(np.abs(x) > thr))

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
        "flatness": spectral_flatness(audio),
        "ns_ratio": non_silence_ratio(audio),
    }

# ---------------------- visualization functions ----------------------
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

def plot_hist(vals: np.ndarray, out_png: Path, title: str, xlabel: str):
    plt.figure(figsize=(6,4))
    plt.hist(vals, bins=20)
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

# ---------------------- CLAP analysis ----------------------
def clap_text_emb(texts: List[str], model: ClapModel, proc: ClapProcessor, device: str) -> torch.Tensor:
    with torch.no_grad():
        inputs = proc(text=texts, return_tensors="pt", padding=True).to(device)
        z = model.get_text_features(**inputs)
        z = torch.nn.functional.normalize(z, dim=-1)
        return z

def clap_audio_emb(wavs: List[np.ndarray], model: ClapModel, proc: ClapProcessor, device: str, target_sr: int = 48000) -> torch.Tensor:
    waves = []
    for x in wavs:
        if x.ndim > 1: x = x.mean(axis=1)
        waves.append(x.astype(np.float32))
    with torch.no_grad():
        inputs = proc(audios=waves, sampling_rate=target_sr, return_tensors="pt", padding=True).to(device)
        z = model.get_audio_features(**inputs)
        z = torch.nn.functional.normalize(z, dim=-1)
        return z

# ---------------------- retrieval → style (augment mode) ----------------------
def default_style_map_from_path(path: str) -> List[str]:
    """Very lightweight heuristics to derive style keywords from filenames/dirs."""
    s = path.lower()
    tags = []
    # instrumentation
    for k in ["piano","violin","cello","flute","clarinet","trumpet","guitar","harp","marimba","sax","drum","timpani","strings","orchestra"]:
        if k in s: tags.append(k)
    # mood/texture
    for k in ["ambient","pad","noise","gliss","etude","concerto","suite","nocturne","baroque","romantic","classical","jazz","club","hall","chamber","studio","intimate","bright","warm","dry","lush"]:
        if k in s: tags.append(k)
    if not tags:
        tags = ["instrumental","clean recording"]
    # uniqueness
    seen, uniq = set(), []
    for t in tags:
        if t not in seen:
            seen.add(t); uniq.append(t)
    return uniq[:5]

def build_augmented_prompts(
    base_prompts: List[str],
    neighbors_jsonl: Path,
    style_top_m: int = 5,
) -> List[str]:
    """Use top-M neighbor paths to extract a few style words and append to prompt."""
    rows = read_jsonl(neighbors_jsonl)
    aug = []
    for i, p in enumerate(base_prompts):
        if i < len(rows) and rows[i].get("neighbors"):
            paths = [n.get("path","") for n in rows[i]["neighbors"][:style_top_m]]
            tags = []
            for ph in paths:
                tags += default_style_map_from_path(ph or "")
            # condense
            seen, keep = set(), []
            for t in tags:
                if t not in seen:
                    seen.add(t); keep.append(t)
            style_phrase = ", ".join(keep[:6])
            aug.append(f"{p}, {style_phrase}, high quality, natural timbre")
        else:
            aug.append(f"{p}, high quality, natural timbre")
    return aug

# ---------------------- CLAP embed blend (embed mode) ----------------------
def norm_t(t: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(t, dim=-1)

def load_clap(device: str):
    # Import lazily to avoid torchvision import issues
    from transformers import ClapProcessor, ClapModel
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device).eval()
    return proc, model

# ---------- NEW: stable blend helpers ----------
def orthogonalize(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Remove component of v along u (both [B,D], expected L2-normalized), then re-normalize."""
    proj = (v * u).sum(dim=-1, keepdim=True) * u
    return torch.nn.functional.normalize(v - proj, dim=-1)

def spectral_gate_denoise(y: np.ndarray, sr: int, strength: float=0.12) -> np.ndarray:
    """Light spectral gate: suppress bins near the median floor."""
    D = librosa.stft(y, n_fft=1024, hop_length=256)
    S = np.abs(D)**2
    floor = np.median(S, axis=1, keepdims=True)
    mask = (S > (floor * (1.0 + strength))).astype(np.float32)
    Y = librosa.istft(D * mask, hop_length=256)
    if len(Y) != len(y):
        Y = librosa.util.fix_length(Y, len(y))
    return np.clip(Y, -1.0, 1.0).astype(np.float32)

# ------------------------------ main ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["augment","embed"], default="augment",
                    help="augment = append style words to text (stable). embed = blend CLAP text + style embeddings (advanced).")
    # Shared inputs
    ap.add_argument("--prompts_file", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    # Augment-mode inputs
    ap.add_argument("--neighbors_jsonl", type=Path, help="Precomputed neighbors.jsonl (for augment mode)")
    ap.add_argument("--style_top_m", type=int, default=5)
    # Embed-mode inputs
    ap.add_argument("--agg_npy", type=Path, help="agg_emb.npy from precompute_neighbors (for embed mode)")
    ap.add_argument("--alpha_style", type=float, default=0.12, help="Blend amount for style (keep small to avoid noise)")
    ap.add_argument("--normalize_style", action="store_true", help="L2-normalize style embeddings first")
    ap.add_argument("--normalize_after_blend", action="store_true", help="L2-normalize after mixing")
    # ---------- NEW: stability flags (non-breaking additions) ----------
    ap.add_argument("--blend_mode", choices=["plain","orth"], default="orth",
                    help="How to mix style with text in embed mode.")
    ap.add_argument("--center_style_bank", action="store_true",
                    help="Mean-center the style bank before use (recommended for embed).")
    ap.add_argument("--speech_push", type=float, default=0.0,
                    help="Small push away from a 'speech' text vector in cond branch (0–0.2).")
    ap.add_argument("--post_denoise", choices=["none","spectralgate"], default="none",
                    help="Optional post denoise on generated audio.")
    ap.add_argument("--denoise_strength", type=float, default=0.12,
                    help="Strength for spectral gate denoise (0.08–0.18 typical).")
    # Text prior / negatives / guards (kept from previous versions)
    ap.add_argument("--text_prior", type=str, default=None,
                    help='Optional text prior like "instrumental music, no vocals"')
    ap.add_argument("--prior_weight", type=float, default=0.0,
                    help="How much of the text prior to blend into the text embedding (0-1).")
    ap.add_argument("--negative_text", type=str, default="",
                    help="Optional negative prompt that forms the unconditional CLAP text vector")
    ap.add_argument("--music_anchor_weight", type=float, default=0.0,
                    help="(Legacy) weight toward a generic 'music' vector (0-0.4 typical)")
    ap.add_argument("--style_gate_margin", type=float, default=0.0,
                    help="(Legacy) if style⋅text < margin, downweight style influence")
    # Generation
    ap.add_argument("--model_id", default="cvssp/audioldm-m-full")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["auto","fp16","fp32"], default="auto")
    ap.add_argument("--scheduler", choices=["dpm","ddim"], default="dmp")
    ap.add_argument("--num_items", type=int, default=None)
    ap.add_argument("--num_waveforms_per_prompt", type=int, default=1)
    ap.add_argument("--negative", type=str, default=(
        "white noise, hiss, static, clipping, distortion, crackle, silence, "
        "background noise, low fidelity, bandlimited, metallic artifacts, "
        "detuned, off-key, glitch"
    ))
    # Viz + metrics
    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--per_item_viz", action="store_true", help="Save per-item analysis PNGs")
    ap.add_argument("--clap_device", type=str, default=None, help="cuda or cpu; default: cuda if available")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # Hygiene
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    gen_device = "cuda" if torch.cuda.is_available() else "cpu"
    clap_device = args.clap_device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype == "fp16" or (args.dtype == "auto" and gen_device == "cuda"):
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Prompts
    prompts = read_prompts(args.prompts_file)
    if args.num_items is not None:
        prompts = prompts[: args.num_items]

    # Mode A: augment text prompts using neighbors
    if args.mode == "augment":
        if not args.neighbors_jsonl or not Path(args.neighbors_jsonl).exists():
            raise SystemExit("--neighbors_jsonl is required for --mode augment")
        use_prompts = build_augmented_prompts(prompts, args.neighbors_jsonl, style_top_m=args.style_top_m)
    else:
        use_prompts = prompts

    # Output dirs
    out_dir = args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = out_dir / "wav"; wav_dir.mkdir(exist_ok=True)
    metrics_dir = out_dir / "metrics"; metrics_dir.mkdir(exist_ok=True)
    viz_dir = out_dir / "visualizations"; viz_dir.mkdir(exist_ok=True)

    # Pipeline
    print("The AudioLDMPipeline is deprecated but works on diffusers >= 0.33.1.")
    print(f"[INFO] mode={args.mode}  model={args.model_id}  steps={args.steps}  guidance={args.guidance}")
    print(f"[INFO] items={len(use_prompts)}  seconds={args.seconds}  batch={args.batch}  dtype={torch_dtype}")

    pipe = AudioLDMPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    if args.scheduler == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(gen_device)
    try: pipe.enable_attention_slicing()
    except: pass
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    sr = getattr(pipe, "audio_sample_rate", 16000)
    gen = torch.Generator(device=gen_device).manual_seed(args.seed)

    # If embed mode, set up CLAP + style bank
    processor = clap_model = None
    S = None
    if args.mode == "embed":
        if not args.agg_npy or not Path(args.agg_npy).exists():
            raise SystemExit("--agg_npy is required for --mode embed")
        S = np.load(args.agg_npy, mmap_mode="r").astype(np.float32)
        if args.center_style_bank:
            S = S - S.mean(axis=0, keepdims=True)
        if args.normalize_style:
            S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-9)
        processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(gen_device).eval()

    # Generation loop
    audio_files: List[Path] = []
    B = max(1, int(args.batch))
    total = len(use_prompts)
    t0 = time.time()

    for i in tqdm(range(0, total, B), desc="Generating"):
        batch_prompts = use_prompts[i:i+B]
        negative_list = [args.negative] * len(batch_prompts)

        if args.mode == "augment":
            # Standard text prompting
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(gen_device=="cuda" and torch_dtype==torch.float16)):
                out = pipe(
                    prompt=batch_prompts,
                    negative_prompt=negative_list,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    audio_length_in_s=args.seconds,
                    generator=gen,
                    num_waveforms_per_prompt=args.num_waveforms_per_prompt,
                )
        else:
            # Embed blend: compute CLAP text + blend with style rows (sequential)
            idxs = [(i + k) % S.shape[0] for k in range(len(batch_prompts))]
            z_style = torch.from_numpy(S[idxs]).to(gen_device)
            z_style = torch.nn.functional.normalize(z_style, dim=-1)

            # CLAP text embeds (cond / uncond / optional priors)
            with torch.no_grad():
                txt_inputs = processor(text=batch_prompts, return_tensors="pt", padding=True).to(gen_device)
                z_txt = clap_model.get_text_features(**txt_inputs)
                z_txt = torch.nn.functional.normalize(z_txt, dim=-1)

                # unconditional
                if args.negative_text:
                    neg_inputs = processor(text=[args.negative_text]*len(batch_prompts), return_tensors="pt", padding=True).to(gen_device)
                else:
                    neg_inputs = processor(text=[""]*len(batch_prompts), return_tensors="pt", padding=True).to(gen_device)
                z_neg = clap_model.get_text_features(**neg_inputs)
                z_neg = torch.nn.functional.normalize(z_neg, dim=-1)

                # optional text prior
                if args.text_prior and args.prior_weight > 0.0:
                    pr_inputs = processor(text=[args.text_prior]*len(batch_prompts), return_tensors="pt", padding=True).to(gen_device)
                    z_prior = clap_model.get_text_features(**pr_inputs)
                    z_prior = torch.nn.functional.normalize(z_prior, dim=-1)
                    w = float(args.prior_weight)
                    z_txt = torch.nn.functional.normalize((1.0 - w) * z_txt + w * z_prior, dim=-1)

                # optional generic music anchor (legacy)
                if args.music_anchor_weight and args.music_anchor_weight > 0.0:
                    mus_inputs = processor(text=["instrumental music, no vocals"]*len(batch_prompts), return_tensors="pt", padding=True).to(gen_device)
                    z_mus = torch.nn.functional.normalize(clap_model.get_text_features(**mus_inputs), dim=-1)
                    mw = float(args.music_anchor_weight)
                    z_txt = torch.nn.functional.normalize((1.0 - mw) * z_txt + mw * z_mus, dim=-1)

                z_txt_cond = z_txt
                z_txt_uncond = z_neg

            # style gating (legacy, optional)
            if args.style_gate_margin and args.style_gate_margin > 0.0:
                # downweight style if cosine(text, style) < margin
                margin = float(args.style_gate_margin)
                cos_ts = (z_txt_cond * z_style).sum(dim=-1, keepdim=True)  # [B,1]
                gate = torch.clamp((cos_ts - margin) / max(1e-6, (1.0 - margin)), 0.0, 1.0)
                z_style = torch.nn.functional.normalize(gate * z_style + (1.0 - gate) * 0.0, dim=-1)

            # ---------- NEW: orthogonalized or plain blend ----------
            if args.blend_mode == "orth":
                z_style_use = orthogonalize(z_txt_cond, z_style)
            else:
                z_style_use = z_style

            a = float(args.alpha_style)
            z_cond = (1.0 - a) * z_txt_cond + a * z_style_use
            if args.normalize_after_blend:
                z_cond = torch.nn.functional.normalize(z_cond, dim=-1)

            # ---------- NEW: tiny push away from "speech" ----------
            if args.speech_push and args.speech_push > 0.0:
                with torch.no_grad():
                    speech_text = "speech, vocals, singing, human voice, spoken word"
                    sp_in = processor(text=[speech_text]*len(batch_prompts), return_tensors="pt", padding=True).to(gen_device)
                    z_sp = torch.nn.functional.normalize(clap_model.get_text_features(**sp_in), dim=-1)
                beta = float(args.speech_push)
                z_cond = torch.nn.functional.normalize(z_cond - beta * z_sp, dim=-1)

            if args.debug and i == 0:
                cn = torch.norm(z_cond, dim=-1).mean().item()
                un = torch.norm(z_txt_uncond, dim=-1).mean().item()
                mean_gap = torch.norm(z_cond - z_txt_uncond, dim=-1).mean().item()
                print(f"[DEBUG] z_cond mean norm={cn:.3f}  z_uncond mean norm={un:.3f}  mean||cond-uncond||={mean_gap:.3f}  alpha={a}")

            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(gen_device=="cuda" and torch_dtype==torch.float16)):
                out = pipe(
                    prompt_embeds=z_cond,                 # [B,512] (CLAP space)
                    negative_prompt_embeds=z_txt_uncond,  # [B,512]
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    audio_length_in_s=args.seconds,
                    generator=gen,
                    num_waveforms_per_prompt=args.num_waveforms_per_prompt,
                )

        audios = out.audios  # list[np.ndarray]

        # Save audio files
        for j, aud in enumerate(audios):
            # ---------- NEW: optional light post-denoise ----------
            if args.post_denoise == "spectralgate":
                try:
                    aud = spectral_gate_denoise(aud, sr, strength=args.denoise_strength)
                except Exception:
                    pass

            pidx = i + (j // args.num_waveforms_per_prompt)
            v = j % args.num_waveforms_per_prompt
            base = safe_slug(prompts[pidx])  # original base prompt for filename
            name = f"{pidx:03d}__{base}"
            if args.num_waveforms_per_prompt > 1:
                name += f"__v{v}"
            wav_path = wav_dir / f"{name}.wav"
            save_wav(wav_path, aud, sr)
            audio_files.append(wav_path)

    elapsed = time.time() - t0
    print(f"[DONE] wrote {len(audio_files)} files → {out_dir} in {elapsed:.1f}s")

    if not (args.metrics or args.visualize or args.report or args.per_item_viz):
        return

    # --------- CLAP metrics ---------
    print("[METRICS] CLAP embeddings …")
    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(clap_device).eval()
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    Zt = clap_text_emb(use_prompts, clap, proc, clap_device).cpu()  # [N,512] - use actual prompts used
    wavs_48k = []
    for p in audio_files:
        y, sr0 = sf.read(p, dtype="float32", always_2d=False)
        if y.ndim > 1: y = y.mean(axis=1)
        if sr0 != 48000:
            y = librosa.resample(y, orig_sr=sr0, target_sr=48000)
        wavs_48k.append(y)
    Za = clap_audio_emb(wavs_48k, clap, proc, clap_device, target_sr=48000).cpu()  # [N,512]
    sims = (Zt * Za).sum(dim=-1).numpy()  # cosine similarity

    # --------- audio stats & per-item viz ---------
    durations = []; max_amps = []; rms_db = []; zcrs = []
    crest = []; centroid = []; rolloff = []; snr_db = []; dyn_rng = []
    flatness = []; ns_ratios = []
    per_item_paths = []

    for idx, p in enumerate(audio_files):
        y, srr = sf.read(p, dtype="float32", always_2d=False)
        if y.ndim > 1: y = y.mean(axis=1)
        m = analyze_audio_quality(y, srr)
        durations.append(m["duration_seconds"]); max_amps.append(m["max_amp"])
        rms_db.append(m["rms_dbfs"]); zcrs.append(m["zero_crossing_rate"])
        crest.append(m["crest_factor"]); centroid.append(m["spectral_centroid"])
        rolloff.append(m["spectral_rolloff"]); snr_db.append(m["snr_db"])
        dyn_rng.append(m["dynamic_range_db"]); flatness.append(m["flatness"])
        ns_ratios.append(m["ns_ratio"])

        if args.per_item_viz:
            pp = viz_dir / f"{idx:03d}_analysis.png"
            per_item_analysis_panel(y, srr, pp, title=audio_files[idx].name)
            per_item_paths.append(pp)

    # diversity from audio embeddings
    Za_np = Za.numpy()
    K = Za_np @ Za_np.T  # cosine sim
    N = K.shape[0]
    triu = K[np.triu_indices(N, k=1)]
    diversity = 1.0 - triu  # cosine distance
    div_mean = float(np.mean(diversity)) if diversity.size else 0.0
    div_p10 = float(np.percentile(diversity, 10)) if diversity.size else 0.0
    div_p90 = float(np.percentile(diversity, 90)) if diversity.size else 0.0

    # --------- write CSV/JSON ---------
    csv_path = metrics_dir / "per_item.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file","prompt","clap_sim","duration_s","max_amp","rms_dbfs","zcr","crest","centroid","rolloff","snr_db","dyn_range_db","flatness","non_sil_ratio"])
        for p, pr, s, d, m, r, z, c, ce, ro, sn, dr, fl, ns in zip(
            audio_files, use_prompts, sims, durations, max_amps, rms_db, zcrs, crest, centroid, rolloff, snr_db, dyn_rng, flatness, ns_ratios
        ):
            w.writerow([str(p), pr, f"{float(s):.4f}", f"{float(d):.3f}", f"{float(m):.4f}",
                        f"{float(r):.2f}", f"{float(z):.4f}", f"{float(c):.3f}", f"{float(ce):.1f}",
                        f"{float(ro):.1f}", f"{float(sn):.2f}", f"{float(dr):.2f}", f"{float(fl):.5f}",
                        f"{float(ns):.5f}"])

    summary = {
        "mode": args.mode,
        "model": args.model_id,
        "seconds": args.seconds,
        "steps": args.steps,
        "guidance": args.guidance,
        "batch": args.batch,
        "dtype": str(torch_dtype),
        "scheduler": args.scheduler,
        "num_waveforms_per_prompt": args.num_waveforms_per_prompt,
        "n_files": len(audio_files),
        "elapsed_sec": elapsed,
        "clap_sim_mean": float(np.mean(sims)),
        "clap_sim_p10": float(np.percentile(sims, 10)),
        "clap_sim_p90": float(np.percentile(sims, 90)),
        "diversity_mean_cosdist": div_mean,
        "diversity_p10": div_p10,
        "diversity_p90": div_p90,
        "rms_dbfs_mean": float(np.mean(rms_db)) if rms_db else 0.0,
        "max_amp_mean": float(np.mean(max_amps)) if max_amps else 0.0,
        "snr_db_mean": float(np.mean(snr_db)) if snr_db else 0.0,
        "dyn_range_db_mean": float(np.mean(dyn_rng)) if dyn_rng else 0.0,
        "alpha_style": float(args.alpha_style) if args.mode=="embed" else None,
        "normalize_style": bool(args.normalize_style) if args.mode=="embed" else None,
        "normalize_after_blend": bool(args.normalize_after_blend) if args.mode=="embed" else None,
        # NEW flags in summary
        "blend_mode": args.blend_mode if args.mode=="embed" else None,
        "center_style_bank": bool(args.center_style_bank) if args.mode=="embed" else None,
        "speech_push": float(args.speech_push) if args.mode=="embed" else None,
        "post_denoise": args.post_denoise if args.mode=="embed" else None,
    }
    (metrics_dir / "summary.json").write_text(json.dumps(convert_numpy_types(summary), indent=2), encoding="utf-8")

    # --------- plots ---------
    if args.visualize or args.report:
        plot_hist(sims, viz_dir / "clap_sim_hist.png", "CLAP Prompt Adherence", "cosine similarity")
        plot_scatter(np.array(rms_db), sims, viz_dir / "sim_vs_loudness.png",
                     "Prompt Adherence vs Loudness", "RMS (dBFS)", "CLAP similarity")
        plot_spectrogram_grid(audio_files, viz_dir / "spec_grid.png", max_n=16)

    # --------- HTML report with audio players ---------
    if args.report:
        per_rows = []
        for i, (p, pr, s, rdb, sn) in enumerate(zip(audio_files, use_prompts, sims, rms_db, snr_db)):
            audio_rel = f"wav/{p.name}"  # audio in wav/ subfolder
            per_viz = (viz_dir / f"{i:03d}_analysis.png")
            viz_rel = f"visualizations/{per_viz.name}" if per_viz.exists() else None
            per_rows.append({
                "idx": i,
                "file": p.name,
                "prompt": pr,
                "sim": float(s),
                "rms_dbfs": float(rdb),
                "snr_db": float(sn),
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

        summary_html = f"""
<h1>AudioLDM1 Retrieval-Conditioned Report</h1>
<p class="meta"><b>Mode:</b> {args.mode} |
<b>Model:</b> {args.model_id} |
<b>Items:</b> {len(use_prompts)} |
<b>Seconds:</b> {args.seconds} |
<b>Steps:</b> {args.steps} |
<b>Guidance:</b> {args.guidance} |
<b>Elapsed:</b> {elapsed:.1f}s</p>
<pre>{json.dumps(summary, indent=2)}</pre>
<h2>Overview Plots</h2>
<div class="grid">
  <div><h3>CLAP similarity histogram</h3><img src="visualizations/clap_sim_hist.png"/></div>
  <div><h3>Similarity vs loudness</h3><img src="visualizations/sim_vs_loudness.png"/></div>
</div>
<h3>Mel-spectrogram grid (first 16)</h3>
<img src="visualizations/spec_grid.png"/>
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
      <p class="small">CLAP sim: {r["sim"]:.3f} &nbsp; | &nbsp; RMS dBFS: {r["rms_dbfs"]:.2f} &nbsp; | &nbsp; SNR: {r["snr_db"]:.2f} dB</p>
    </div>
    {viz_img}
  </div>
</div>
""")
        html_tail = "</body></html>"
        (out_dir / "report.html").write_text(html_head + summary_html + "\n".join(rows_html) + html_tail, encoding="utf-8")
        print(f"HTML report: {out_dir / 'report.html'}")

    print(f"[DONE] Wrote {len(audio_files)} files to {out_dir}")
    print(f"Metrics at: {metrics_dir}")

if __name__ == "__main__":
    main()