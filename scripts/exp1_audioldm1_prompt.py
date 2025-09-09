#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AudioLDM1 prompt-only generation + comprehensive metrics, visualizations, and HTML report with audio players.
- Works with cvssp/audioldm variants (e.g., cvssp/audioldm-m-full, cvssp/audioldm-s-full-v2, cvssp/audioldm).
- Negative prompt batching fixed (list length == prompt batch).
- Metrics:
  * CLAP prompt adherence (cosine(audio_emb, text_emb))
  * Diversity (cosine distance between audio CLAP embeddings)
  * Audio quality: RMS, crest factor, ZCR, spectral centroid, spectral rolloff, SNR (median-filter approx),
    dynamic range (peak/RMS), duration
- Visualizations:
  * CLAP similarity histogram
  * Similarity vs loudness (RMS dBFS) scatter
  * Mel-spectrogram grid (first 16)
  * Optional per-item analysis panel (waveform, mel, FFT, centroid/rolloff over time, RMS over time, amplitude histogram)
- HTML report with embedded audio players for every generated file
Tip: keep prompts, seconds, steps, guidance the same when comparing checkpoints.
"""
from __future__ import annotations
import os, argparse, time, json, math, csv, warnings
from pathlib import Path
from typing import List
import numpy as np
import torch
import soundfile as sf
import matplotlib.pyplot as plt
import librosa, librosa.display
from scipy import signal
from diffusers import AudioLDMPipeline, DPMSolverMultistepScheduler
from transformers import ClapModel, ClapProcessor
warnings.filterwarnings("ignore")
# ============================ helpers ============================
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
def sanitize(s: str, maxlen: int = 60) -> str:
    s = s[:maxlen]
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in s).strip("_")
def save_wav(path: Path, x: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=np.float32).squeeze()
    x = np.clip(x, -1.0, 1.0)
    sf.write(str(path), x, sr)
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
# ============================== main ==============================
def main():
    ap = argparse.ArgumentParser()
    # generation
    ap.add_argument("--model_id", type=str, default="cvssp/audioldm-m-full",
                    help="e.g., cvssp/audioldm-m-full, cvssp/audioldm-s-full-v2, cvssp/audioldm")
    ap.add_argument("--prompts_file", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["auto","fp16","fp32"], default="auto")
    ap.add_argument("--scheduler", choices=["dpm","ddim"], default="dpm")
    ap.add_argument("--num_waveforms_per_prompt", type=int, default=1)
    ap.add_argument("--negative", type=str, default=(
        "white noise, hiss, static, clipping, distortion, crackle, silence, "
        "background noise, low fidelity, bandlimited, metallic artifacts, "
        "detuned, off-key, glitch"
    ))
    # metrics/report
    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--per_item_viz", action="store_true", help="Save per-item analysis PNGs")
    ap.add_argument("--clap_device", type=str, default=None, help="cuda or cpu; default: cuda if available")
    args = ap.parse_args()
    # Hygiene (avoid torchvision import path inside transformers CLAP)
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    gen_device = "cuda" if torch.cuda.is_available() else "cpu"
    clap_device = args.clap_device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = (torch.float16 if (args.dtype=="fp16" or (args.dtype=="auto" and gen_device=="cuda")) else torch.float32)
    prompts = read_prompts(args.prompts_file)
    out_dir = args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = out_dir / "wav"; wav_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = out_dir / "metrics"; metrics_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_dir / "visualizations"; viz_dir.mkdir(parents=True, exist_ok=True)
    # Pipeline
    print(f"[INFO] model={args.model_id}  scheduler={args.scheduler}  steps={args.steps}  guidance={args.guidance}")
    print(f"[INFO] items={len(prompts)}  seconds={args.seconds}  batch={args.batch}  dtype={torch_dtype}")
    pipe = AudioLDMPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    if args.scheduler == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(gen_device)
    try: pipe.enable_attention_slicing()
    except: pass
    if hasattr(pipe, "safety_checker"):  # disable NSFW/safety for research
        pipe.safety_checker = None
    sr = getattr(pipe, "audio_sample_rate", 16000)
    g = torch.Generator(device=gen_device).manual_seed(args.seed)
    B = max(1, int(args.batch))
    # --------- generation ---------
    t0 = time.time()
    audio_files: List[Path] = []
    for i in range(0, len(prompts), B):
        batch_prompts = prompts[i:i+B]
        negative_list = [args.negative] * len(batch_prompts)  # ✅ batch-match
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=(gen_device=="cuda" and torch_dtype==torch.float16)):
            out = pipe(
                prompt=batch_prompts,
                negative_prompt=negative_list,
                num_inference_steps=int(args.steps),
                guidance_scale=float(args.guidance),
                audio_length_in_s=float(args.seconds),
                generator=g,
                num_waveforms_per_prompt=int(args.num_waveforms_per_prompt),
            )
        audios = out.audios  # list[np.ndarray]
        for j, aud in enumerate(audios):
            pidx = i + (j // args.num_waveforms_per_prompt)
            v = j % args.num_waveforms_per_prompt
            name = f"{pidx:03d}__{sanitize(prompts[pidx])}"
            if args.num_waveforms_per_prompt > 1:
                name += f"__v{v}"
            fpath = wav_dir / f"{name}.wav"
            save_wav(fpath, aud, sr)
            audio_files.append(fpath)
    elapsed = time.time() - t0
    print(f"[DONE] wrote {len(audio_files)} files → {out_dir} in {elapsed:.1f}s")
    if not (args.metrics or args.visualize or args.report or args.per_item_viz):
        return
    # --------- CLAP metrics ---------
    print("[METRICS] CLAP embeddings …")
    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(clap_device).eval()
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    Zt = clap_text_emb(prompts, clap, proc, clap_device).cpu()  # [N,512]
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
    per_item_paths = []
    for idx, p in enumerate(audio_files):
        y, srr = sf.read(p, dtype="float32", always_2d=False)
        if y.ndim > 1: y = y.mean(axis=1)
        m = analyze_audio_quality(y, srr)
        durations.append(m["duration_seconds"]); max_amps.append(m["max_amp"])
        rms_db.append(m["rms_dbfs"]); zcrs.append(m["zero_crossing_rate"])
        crest.append(m["crest_factor"]); centroid.append(m["spectral_centroid"])
        rolloff.append(m["spectral_rolloff"]); snr_db.append(m["snr_db"])
        dyn_rng.append(m["dynamic_range_db"])
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
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file","prompt","clap_sim","duration_s","max_amp","rms_dbfs","zcr","crest","centroid","rolloff","snr_db","dyn_range_db"])
        for p, pr, s, d, m, r, z, c, ce, ro, sn, dr in zip(
            audio_files, prompts, sims, durations, max_amps, rms_db, zcrs, crest, centroid, rolloff, snr_db, dyn_rng
        ):
            w.writerow([str(p), pr, f"{float(s):.4f}", f"{float(d):.3f}", f"{float(m):.4f}",
                        f"{float(r):.2f}", f"{float(z):.4f}", f"{float(c):.3f}", f"{float(ce):.1f}",
                        f"{float(ro):.1f}", f"{float(sn):.2f}", f"{float(dr):.2f}"])
    summary = {
        "model_id": args.model_id,
        "items": len(prompts),
        "seconds": args.seconds,
        "steps": args.steps,
        "guidance": args.guidance,
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
    }
    (metrics_dir / "summary.json").write_text(json.dumps(convert_numpy_types(summary), indent=2))
    # --------- plots ---------
    if args.visualize or args.report:
        plot_hist(sims, metrics_dir / "clap_sim_hist.png", "CLAP Prompt Adherence", "cosine similarity")
        plot_scatter(np.array(rms_db), sims, metrics_dir / "sim_vs_loudness.png",
                     "Prompt Adherence vs Loudness", "RMS (dBFS)", "CLAP similarity")
        plot_spectrogram_grid(audio_files, metrics_dir / "spec_grid.png", max_n=16)
    # --------- HTML report with audio players ---------
    if args.report:
        # Put the report at out_dir so relative paths to WAVs are trivial
        per_rows = []
        for i, (p, pr, s, rdb, sn) in enumerate(zip(audio_files, prompts, sims, rms_db, snr_db)):
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
<title>AudioLDM1 Prompt Report</title>
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
<h1>AudioLDM1 Prompt-Only Report</h1>
<p class="meta"><b>Model:</b> {args.model_id} |
<b>Items:</b> {len(prompts)} |
<b>Seconds:</b> {args.seconds} |
<b>Steps:</b> {args.steps} |
<b>Guidance:</b> {args.guidance} |
<b>Elapsed:</b> {elapsed:.1f}s</p>
<pre>{json.dumps(summary, indent=2)}</pre>
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
        print(f"[REPORT] {out_dir/'report.html'}")

if __name__ == "__main__":
    main()