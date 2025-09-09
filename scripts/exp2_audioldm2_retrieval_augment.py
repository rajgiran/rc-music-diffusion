#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp2 (AudioLDM2): Retrieval-conditioned generation via text augmentation + full metrics/viz/report.

Why this script?
- AudioLDM2 conditions on TEXT (T5). So we push retrieval information into the prompt text,
  not via CLAP prompt_embeds injection (which is unstable / unsupported here).
- Mirrors your folder layout: wav/, metrics/, visualizations/, report.html
- Keeps your evaluation stack: per-item metrics, CLAP adherence, diversity, plots, HTML with players.

Two modes:
  --mode augment     (recommended)  : build style phrase from neighbors.jsonl and append to prompt
  --mode textonly                 : use raw prompts (helpful for A/B against augment)

Usage example:
  python exp2_audioldm2_retrieval.py \
    --mode augment \
    --prompts_file /path/prompts.txt \
    --neighbors_jsonl /path/neighbors.jsonl \
    --out_dir /path/exp_out \
    --model_id cvssp/audioldm2-large --seconds 8 --steps 150 --guidance 3.0 \
    --negative_text "speech, vocals, singing, rap, human voice, narration, crowd" \
    --text_prior "instrumental music, no vocals" --prior_weight 0.35 \
    --metrics --visualize --report --per_item_viz
"""
from __future__ import annotations
import os, csv, json, math, time, argparse, warnings
from pathlib import Path
from typing import List
import numpy as np
import soundfile as sf

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa, librosa.display
from tqdm import tqdm

from diffusers import AudioLDM2Pipeline, DPMSolverMultistepScheduler
from transformers import ClapModel, ClapProcessor

warnings.filterwarnings("ignore")

# ---------------------- I/O helpers ----------------------
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

def convert_numpy_types(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_numpy_types(x) for x in obj]
    return obj

# ---------------------- simple style-tag heuristics ----------------------
def default_style_map_from_path(path: str) -> List[str]:
    s = path.lower()
    tags = []
    for k in ["piano","violin","cello","flute","clarinet","trumpet","guitar","harp","marimba","sax","drum","timpani","strings","orchestra","bass","pad"]:
        if k in s: tags.append(k)
    for k in ["ambient","pad","noise","gliss","etude","concerto","suite","nocturne","baroque","romantic","classical","jazz","club","hall","chamber","studio","intimate","bright","warm","dry","lush","minimal","meditative","expressive"]:
        if k in s: tags.append(k)
    if not tags:
        tags = ["instrumental","clean recording"]
    # de-dupe
    seen, keep = set(), []
    for t in tags:
        if t not in seen:
            seen.add(t); keep.append(t)
    return keep[:6]

def build_augmented_prompts(base_prompts: List[str], neighbors_jsonl: Path, top_m: int = 5,
                            text_prior: str | None = None, prior_weight: float = 0.0) -> List[str]:
    rows = read_jsonl(neighbors_jsonl)
    out = []
    for i, p in enumerate(base_prompts):
        style_phrase = ""
        if i < len(rows) and rows[i].get("neighbors"):
            paths = [n.get("path", "") for n in rows[i]["neighbors"][:top_m]]
            tags = []
            for ph in paths:
                tags += default_style_map_from_path(ph or "")
            # uniq & concise
            seen, keep = set(), []
            for t in tags:
                if t not in seen:
                    seen.add(t); keep.append(t)
            if keep:
                style_phrase = ", " + ", ".join(keep[:6])
        # prior as plain-text hint (ALDM2 text encoder handles this well)
        prior = ""
        if text_prior and prior_weight > 0.0:
            # keep it short; repeating priors in text works better than trying to weight embeddings
            prior = f", {text_prior}"
        out.append(f"{p}{style_phrase}{prior}")
    return out

# ---------------------- light analysis ----------------------
def dbfs(x: np.ndarray, eps=1e-12) -> float:
    rms = np.sqrt(np.mean(x**2) + eps)
    return float(20.0 * np.log10(rms + eps))

def analyze_audio_quality(y: np.ndarray, sr: int) -> dict:
    # fast metrics used in your previous scripts
    rms = float(np.sqrt(np.mean(y**2)))
    peak = float(np.max(np.abs(y)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(y)))) / 2.0)
    # dynamic range (peak/rms)
    dyn = float(20.0 * np.log10(max(peak,1e-9) / max(rms,1e-9))) if rms > 0 else 0.0
    # spectral centroid rough
    mag = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1.0/sr)
    denom = float(np.sum(mag)) if np.sum(mag) > 0 else 1.0
    centroid = float(np.sum(mag * freqs) / denom)
    return dict(rms=rms, rms_dbfs=dbfs(y), peak=peak, zcr=zcr, dynamic_range_db=dyn, spectral_centroid=centroid)

# ---------------------- viz ----------------------
def per_item_panel(y: np.ndarray, sr: int, path: Path, title: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    t = np.linspace(0, len(y)/sr, num=len(y))
    axes[0,0].plot(t, y); axes[0,0].set_title("Waveform"); axes[0,0].grid(True, alpha=0.3)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96, fmax=min(8000, sr//2))
    Sdb = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(Sdb, sr=sr, x_axis="time", y_axis="mel", ax=axes[0,1]); axes[0,1].set_title("Mel")
    rms = librosa.feature.rms(y=y)[0]; trms = librosa.times_like(rms); axes[1,0].plot(trms, rms); axes[1,0].set_title("RMS"); axes[1,0].grid(True, alpha=0.3)
    axes[1,1].hist(y, bins=60); axes[1,1].set_title("Amplitude hist"); axes[1,1].grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()

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

# ---------------------- CLAP eval ----------------------
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

# ------------------------------ main ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["augment","textonly"], default="augment")
    ap.add_argument("--prompts_file", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--neighbors_jsonl", type=Path, help="For --mode augment")
    ap.add_argument("--style_top_m", type=int, default=5)

    # Generation
    ap.add_argument("--model_id", default="cvssp/audioldm2-large")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["auto","fp16","fp32"], default="auto")
    ap.add_argument("--scheduler", choices=["dpm","ddim"], default="dpm")

    # Prompt controls
    ap.add_argument("--negative_text", type=str, default="speech, vocals, singing, rap, spoken word, dialogue, narration, choir, chant, crowd, human voice, noise, hiss, static, distortion, clipping")
    ap.add_argument("--text_prior", type=str, default=None)
    ap.add_argument("--prior_weight", type=float, default=0.0)

    # Eval/report
    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--per_item_viz", action="store_true")
    ap.add_argument("--clap_device", type=str, default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    gen_device = "cuda" if torch.cuda.is_available() else "cpu"
    clap_device = args.clap_device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype == "fp16" or (args.dtype == "auto" and gen_device == "cuda"):
        torch_dtype = torch.float16
    elif args.dtype == "fp32":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16 if gen_device == "cuda" else torch.float32

    prompts = read_prompts(args.prompts_file)
    if args.mode == "augment":
        if not args.neighbors_jsonl or not Path(args.neighbors_jsonl).exists():
            raise SystemExit("--neighbors_jsonl is required for --mode augment")
        use_prompts = build_augmented_prompts(prompts, args.neighbors_jsonl,
                                             top_m=args.style_top_m,
                                             text_prior=args.text_prior,
                                             prior_weight=args.prior_weight)
    else:
        # Add prior directly if requested
        if args.text_prior and args.prior_weight > 0.0:
            use_prompts = [f"{p}, {args.text_prior}" for p in prompts]
        else:
            use_prompts = prompts

    out_dir = args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = out_dir / "wav"; wav_dir.mkdir(exist_ok=True)
    metrics_dir = out_dir / "metrics"; metrics_dir.mkdir(exist_ok=True)
    viz_dir = out_dir / "visualizations"; viz_dir.mkdir(exist_ok=True)

    # Pipeline
    print(f"[INFO] mode={args.mode}  model={args.model_id}  steps={args.steps}  guidance={args.guidance}")
    print(f"[INFO] items={len(use_prompts)}  seconds={args.seconds}  batch={args.batch}  dtype={torch_dtype}")

    pipe = AudioLDM2Pipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    if args.scheduler == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(gen_device)
    try: pipe.enable_attention_slicing()
    except: pass

    sr = getattr(pipe, "audio_sample_rate", 16000)
    gen = torch.Generator(device=gen_device).manual_seed(args.seed)

    # Generate
    audio_files: List[Path] = []
    B = max(1, int(args.batch))
    t0 = time.time()
    for i in tqdm(range(0, len(use_prompts), B), desc="Generating"):
        batch_prompts = use_prompts[i:i+B]
        negs = [args.negative_text] * len(batch_prompts)

        with torch.autocast(device_type="cuda", dtype=torch_dtype if torch_dtype==torch.float16 else None,
                            enabled=(gen_device=="cuda" and torch_dtype==torch.float16)):
            out = pipe(
                prompt=batch_prompts,
                negative_prompt=negs,
                num_inference_steps=args.steps,
                audio_length_in_s=args.seconds,
                guidance_scale=args.guidance,
                generator=gen,
            )

        for j, aud in enumerate(out.audios):
            idx = i + j
            base = safe_slug(prompts[idx])
            wav_path = wav_dir / f"{idx:03d}__{base}.wav"
            save_wav(wav_path, aud, sr)
            audio_files.append(wav_path)

    elapsed = time.time() - t0
    print(f"[DONE] wrote {len(audio_files)} files → {out_dir} in {elapsed:.1f}s")

    if not (args.metrics or args.visualize or args.report or args.per_item_viz):
        return

    # CLAP eval
    print("[METRICS] CLAP embeddings …")
    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(clap_device).eval()
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    Zt = clap_text_emb(use_prompts, clap, proc, clap_device).cpu()
    wavs_48k = []
    for p in audio_files:
        y, sr0 = sf.read(p, dtype="float32", always_2d=False)
        if y.ndim > 1: y = y.mean(axis=1)
        if sr0 != 48000:
            y = librosa.resample(y, orig_sr=sr0, target_sr=48000)
        wavs_48k.append(y)
    Za = clap_audio_emb(wavs_48k, clap, proc, clap_device, target_sr=48000).cpu()
    sims = (Zt * Za).sum(dim=-1).numpy()

    # Per-item metrics
    rows = []
    rms_db_list, snr_proxy, centroids = [], [], []
    for idx, p in enumerate(audio_files):
        y, srr = sf.read(p, dtype="float32", always_2d=False)
        if y.ndim > 1: y = y.mean(axis=1)
        m = analyze_audio_quality(y, srr)
        rows.append(dict(
            file=str(p.name), prompt=use_prompts[idx], clap_sim=float(sims[idx]),
            rms=float(m["rms"]), rms_dbfs=float(m["rms_dbfs"]), peak=float(m["peak"]),
            zcr=float(m["zcr"]), dynamic_range_db=float(m["dynamic_range_db"]),
            spectral_centroid=float(m["spectral_centroid"])
        ))
        rms_db_list.append(m["rms_dbfs"]); centroids.append(m["spectral_centroid"])
    # diversity (cosine distance on CLAP audio)
    Za_np = Za.numpy(); K = Za_np @ Za_np.T
    N = K.shape[0]; triu = K[np.triu_indices(N, 1)]
    diversity = 1.0 - triu
    div_mean = float(np.mean(diversity)) if diversity.size else 0.0

    # Save metrics
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(metrics_dir / "per_item.csv", index=False)
    summary = dict(
        model=args.model_id, mode=args.mode, seconds=args.seconds, steps=args.steps,
        guidance=args.guidance, n_files=len(audio_files), elapsed_sec=elapsed,
        clap_sim_mean=float(np.mean(sims)), clap_sim_p10=float(np.percentile(sims,10)),
        clap_sim_p90=float(np.percentile(sims,90)), diversity_mean_cosdist=div_mean,
        rms_dbfs_mean=float(np.mean(rms_db_list)) if rms_db_list else 0.0
    )
    (metrics_dir / "summary.json").write_text(json.dumps(convert_numpy_types(summary), indent=2), encoding="utf-8")

    # Visualizations
    if args.visualize or args.report:
        plot_hist(sims, viz_dir / "clap_sim_hist.png", "CLAP Prompt Adherence", "cosine similarity")
        plot_scatter(np.array(rms_db_list), sims, viz_dir / "sim_vs_loudness.png",
                     "Prompt Adherence vs Loudness", "RMS (dBFS)", "CLAP similarity")
        plot_spectrogram_grid(audio_files, viz_dir / "spec_grid.png", max_n=16)

    if args.per_item_viz:
        for i, p in enumerate(audio_files):
            y, srr = sf.read(p, dtype="float32", always_2d=False)
            if y.ndim > 1: y = y.mean(axis=1)
            per_item_panel(y, srr, viz_dir / f"{i:03d}_analysis.png", title=p.name)

    # HTML report
    if args.report:
        per_rows = []
        for i, p in enumerate(audio_files):
            rel_audio = f"wav/{p.name}"
            img = viz_dir / f"{i:03d}_analysis.png"
            rel_img = f"visualizations/{img.name}" if img.exists() else None
            per_rows.append(dict(idx=i, file=p.name, prompt=use_prompts[i],
                                 sim=float(sims[i]),
                                 audio_rel=rel_audio, viz_rel=rel_img))

        html = """<!doctype html><html><head><meta charset="utf-8">
<title>AudioLDM2 Retrieval Report</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;}
h1,h2{margin:8px 0 12px} .row{padding:10px 12px;border:1px solid #eee;border-radius:10px;margin-bottom:10px}
audio{width:100%} .meta{font-size:13px;color:#444}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
.small{font-size:12px;color:#666} img{max-width:100%}
</style></head><body>"""
        html += f"""
<h1>AudioLDM2 Retrieval-Conditioned Report</h1>
<p class="meta"><b>Mode:</b> {args.mode} | <b>Model:</b> {args.model_id} | <b>Items:</b> {len(use_prompts)}
 | <b>Seconds:</b> {args.seconds} | <b>Steps:</b> {args.steps} | <b>Guidance:</b> {args.guidance}
 | <b>Elapsed:</b> {elapsed:.1f}s</p>
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
        for r in per_rows:
            viz = f'<div><img src="{r["viz_rel"]}"/></div>' if r["viz_rel"] else ""
            html += f"""
<div class="row">
  <div class="grid">
    <div>
      <p><b>#{r["idx"]:03d}</b> — <span class="small">{r["file"]}</span></p>
      <p><b>Prompt:</b> {r["prompt"]}</p>
      <audio controls src="{r["audio_rel"]}"></audio>
      <p class="small">CLAP sim: {r["sim"]:.3f}</p>
    </div>
    {viz}
  </div>
</div>
"""
        html += "</body></html>"
        (out_dir / "report.html").write_text(html, encoding="utf-8")
        print(f"HTML report: {out_dir/'report.html'}")

    print(f"[DONE] Wrote {len(audio_files)} files to {out_dir}")
    print(f"Metrics at: {metrics_dir}")

if __name__ == "__main__":
    main()