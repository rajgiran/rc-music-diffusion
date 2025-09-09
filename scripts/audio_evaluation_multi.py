#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Model Audio Evaluation
- Any number of models: name=directory pairs
- Objective metrics (RMS, SNR, spectral stats, dynamic range)
- CLAP prompt adherence per model
- Pairwise FAD matrix (VGGish via TF-Hub; optional)
- Clean HTML report with inline playable audio samples

Example:
  python audio_evaluation_multi.py \
    --models "ALDM1=/path/exp1a" "ALDM2=/path/exp1b" "ALDM2-music=/path/exp1c" \
    --prompts /path/prompts.txt \
    --out_dir /path/eval_out \
    --clap laion/clap-htsat-unfused --clap-device cuda \
    --fad-device cpu \
    --html_samples 6
"""

import os
import re
import json
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

from scipy import stats
from scipy.spatial.distance import pdist
from scipy.linalg import sqrtm

warnings.filterwarnings("ignore")

# Optional TensorFlow (for FAD / VGGish)
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# Optional CLAP via Transformers
HF_CLAP_AVAILABLE = True
try:
    import torch
    from transformers import AutoProcessor, ClapModel
except Exception:
    HF_CLAP_AVAILABLE = False


# ----------------------------
# Utilities
# ----------------------------
def natural_key(path: Path) -> List:
    """Sort files naturally (e.g., 2 before 10)."""
    s = str(path.name)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def read_prompts(p: Path) -> List[str]:
    txt = p.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.replace("\ufeff", "").strip() for ln in txt.splitlines()]
    prompts = [ln for ln in lines if ln and not ln.startswith("#")]
    return prompts

def load_audio_mono(path, target_sr=None):
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y)
    return y, sr

def rms_db(x, eps=1e-12):
    r = np.sqrt(np.mean(x * x) + eps)
    return 20.0 * np.log10(r + eps)

def safe_div(a, b, eps=1e-12):
    return a / (b + eps)

def quick_snr_estimate(y, frame_length=2048, hop_length=512, eps=1e-12):
    """Heuristic SNR: noise ~ 10th percentile frame RMS."""
    frame_rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length).squeeze(0)
    if frame_rms.size == 0:
        return 0.0
    signal_rms = np.mean(frame_rms)
    noise_rms = np.percentile(frame_rms, 10)
    snr = 20.0 * np.log10(safe_div(signal_rms, noise_rms, eps))
    return float(snr)

def compute_objective_features(path, sr_analysis=22050):
    try:
        y, sr = load_audio_mono(path, target_sr=sr_analysis)
        duration = len(y) / float(sr)

        rms_lin = float(np.sqrt(np.mean(y ** 2)))
        rms_db_val = float(rms_db(y))
        peak = float(np.max(np.abs(y)) + 1e-12)
        snr = quick_snr_estimate(y)

        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
        spec_centroid = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
        spec_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr)))
        spec_flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))

        frame_rms = librosa.feature.rms(S=S).squeeze(0)
        if frame_rms.size > 0:
            p95 = np.percentile(frame_rms, 95)
            p05 = np.percentile(frame_rms, 5)
            dyn_range = float(20.0 * np.log10(safe_div(p95, p05)))
        else:
            dyn_range = 0.0

        return dict(
            file=str(path),
            duration=duration,
            rms=rms_lin,
            rms_db=rms_db_val,
            peak=peak,
            snr=snr,
            spectral_centroid=spec_centroid,
            spectral_bandwidth=spec_bandwidth,
            frequency_diversity=spec_flatness,
            dynamic_range=dyn_range,
        )
    except Exception as e:
        return dict(file=str(path), error=str(e))


# ----------------------------
# CLAP helpers
# ----------------------------
def load_clap(checkpoint: str, device: str):
    if not HF_CLAP_AVAILABLE:
        return None, None, "cpu"
    try:
        from transformers import AutoProcessor, ClapModel
        proc = AutoProcessor.from_pretrained(checkpoint)
        model = ClapModel.from_pretrained(checkpoint)
        dev = device
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        model.to(dev).eval()
        return model, proc, dev
    except Exception as e:
        print(f"[CLAP] Disabled (load error: {e})")
        return None, None, "cpu"

@torch.inference_mode()
def clap_text_emb(prompts: List[str], model, proc, device: str) -> torch.Tensor:
    assert isinstance(prompts, list) and len(prompts) > 0, "prompts must be non-empty list[str]"
    inputs = proc(text=prompts, return_tensors="pt", padding=True).to(device)
    out = model.get_text_features(**inputs)
    out = out / out.norm(dim=-1, keepdim=True)
    return out

@torch.inference_mode()
def clap_audio_text_sim(wav_path: Path, prompt: str, model, proc, device: str, sr_target=48000) -> float:
    y, _ = librosa.load(wav_path, sr=sr_target, mono=True)
    ins = proc(text=[prompt], audios=[y], sampling_rate=sr_target, return_tensors="pt", padding=True)
    ins = {k: v.to(device) for k, v in ins.items()}
    outs = model(**ins)
    at = outs.audio_embeds / outs.audio_embeds.norm(dim=-1, keepdim=True)
    tt = outs.text_embeds / outs.text_embeds.norm(dim=-1, keepdim=True)
    return float((at @ tt.T).squeeze().item())


# ----------------------------
# FAD helpers (VGGish)
# ----------------------------
def load_vggish_for_fad(device_pref: str):
    if not TF_AVAILABLE:
        print("[FAD] Disabled (TensorFlow not available)")
        return None
    try:
        if device_pref == "cpu":
            try:
                tf.config.set_visible_devices([], "GPU")
            except Exception:
                pass
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("[FAD] Loading VGGish from TF-Hub …")
        model = hub.load("https://tfhub.dev/google/vggish/1")
        print("[FAD] VGGish ready")
        return model
    except Exception as e:
        print(f"[FAD] Disabled (load error: {e})")
        return None

def vggish_embed_files(vggish_model, files: List[Path], target_sr=16000):
    if vggish_model is None:
        return None
    embs = []
    for p in files:
        try:
            y, _ = load_audio_mono(p, target_sr=target_sr)
            min_len = int(0.96 * target_sr)
            if len(y) < min_len:
                y = np.pad(y, (0, min_len - len(y)))
            x = tf.convert_to_tensor(y, dtype=tf.float32)
            e = vggish_model(x).numpy()
            if e.ndim == 2:
                e = e.mean(axis=0)
            embs.append(e[None, :])
        except Exception as e:
            print(f"[FAD] embed fail {p.name}: {e}")
    return np.vstack(embs) if embs else None

def fad_from_embs(A: np.ndarray, B: np.ndarray) -> float:
    mu1, mu2 = A.mean(0), B.mean(0)
    s1, s2 = np.cov(A, rowvar=False), np.cov(B, rowvar=False)
    diff = mu1 - mu2
    eps = 1e-6
    s1 = s1 + eps * np.eye(s1.shape[0])
    s2 = s2 + eps * np.eye(s2.shape[0])
    try:
        sp = sqrtm(s1 @ s2)
        if np.iscomplexobj(sp):
            sp = sp.real
    except Exception:
        sp = np.eye(s1.shape[0])
    return float(diff @ diff + np.trace(s1 + s2 - 2 * sp))


# ----------------------------
# Evaluator
# ----------------------------
class MultiModelEvaluator:
    def __init__(
        self,
        model_dirs: Dict[str, Path],
        prompts_file: Path,
        out_dir: Path,
        clap_ckpt: str = "laion/clap-htsat-unfused",
        clap_device: str = "cuda",
        fad_device: str = "cpu",
        html_samples: int = 6,
    ):
        self.model_dirs = model_dirs
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # NEW: categorized folders
        self.metrics_dir = self.out_dir / "metrics"
        self.viz_dir = self.out_dir / "visualizations"
        self.metrics_dir.mkdir(exist_ok=True)
        self.viz_dir.mkdir(exist_ok=True)

        self.prompts = read_prompts(prompts_file)
        if len(self.prompts) == 0:
            raise SystemExit(f"[ERROR] No prompts found in {prompts_file} (non-empty, non-# lines required).")
        print(f"[INFO] Loaded {len(self.prompts)} prompts from {prompts_file}")
        for ex in self.prompts[:3]:
            print("  ·", ex)

        # list wavs per model
        self.model_files: Dict[str, List[Path]] = {}
        for name, d in self.model_dirs.items():
            files = sorted(d.glob("*.wav"), key=natural_key)
            print(f"[{name}] {len(files)} wav files")
            self.model_files[name] = files

        # CLAP & VGGish
        self.clap_model, self.clap_proc, self.clap_dev = load_clap(clap_ckpt, clap_device)
        self.vggish = load_vggish_for_fad(fad_device)
        self.html_samples = max(0, int(html_samples))

    # ---------- metrics ----------
    def compute_quality_metrics(self) -> pd.DataFrame:
        rows = []
        for name, files in self.model_files.items():
            print(f"[Eval] Objective features • {name}")
            for f in files:
                m = compute_objective_features(f)
                m["model"] = name
                rows.append(m)
        df = pd.DataFrame(rows)
        df.to_csv(self.metrics_dir / "objective_metrics.csv", index=False)
        return df

    def compute_diversity(self) -> pd.DataFrame:
        print("[Eval] Diversity …")
        out = []
        for name, files in self.model_files.items():
            embs = []
            for f in files:
                try:
                    y, sr = librosa.load(f, sr=16000, mono=True)
                    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                    mel_db = librosa.power_to_db(mel, ref=np.max)
                    feat = np.concatenate([mel_db.mean(1), mel_db.std(1), mel_db.max(1)])
                    embs.append(feat)
                except Exception as e:
                    print(f"[Diversity] fail {Path(f).name}: {e}")
            if len(embs) >= 2:
                D = pdist(np.stack(embs), metric="cosine")
                out.append(dict(model=name,
                                mean_pairwise_distance=float(np.mean(D)),
                                std_pairwise_distance=float(np.std(D)),
                                min_distance=float(np.min(D)),
                                max_distance=float(np.max(D)),
                                diversity_score=float(np.mean(D)),
                                num_samples=int(len(embs))))
            else:
                out.append(dict(model=name, mean_pairwise_distance=np.nan,
                                std_pairwise_distance=np.nan, min_distance=np.nan,
                                max_distance=np.nan, diversity_score=np.nan, num_samples=len(embs)))
        df = pd.DataFrame(out)
        df.to_csv(self.metrics_dir / "diversity_metrics.csv", index=False)
        return df

    def compute_clap_prompt_adherence(self) -> pd.DataFrame | None:
        if self.clap_model is None:
            print("[CLAP] Skipped (no model).")
            return None

        print("[Eval] CLAP prompt adherence …")
        rows = []
        for name, files in self.model_files.items():
            n = min(len(files), len(self.prompts))
            for i in range(n):
                try:
                    s = clap_audio_text_sim(files[i], self.prompts[i], self.clap_model, self.clap_proc, self.clap_dev)
                except Exception as e:
                    s = None
                    print(f"[CLAP] fail {files[i].name}: {e}")
                rows.append(dict(index=i, model=name, file=files[i].name, prompt=self.prompts[i], clap_similarity=s))
        df = pd.DataFrame(rows)
        df.to_csv(self.metrics_dir / "clap_prompt_adherence.csv", index=False)
        # summary
        if not df.empty:
            df.groupby("model")["clap_similarity"].mean().to_csv(self.metrics_dir / "clap_prompt_adherence_summary.csv")
        return df

    def compute_pairwise_fad(self) -> pd.DataFrame | None:
        if self.vggish is None:
            print("[FAD] Skipped (no VGGish).")
            return None
        print("[Eval] Pairwise FAD …")
        names = list(self.model_files.keys())
        embs = {}
        for name in names:
            e = vggish_embed_files(self.vggish, self.model_files[name])
            if e is None or e.shape[0] == 0:
                print(f"[FAD] No embeddings for {name}; skipping matrix.")
                return None
            embs[name] = e

        mat = np.zeros((len(names), len(names)), dtype=np.float64)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i == j:
                    mat[i, j] = 0.0
                elif j < i:
                    mat[i, j] = mat[j, i]
                else:
                    mat[i, j] = fad_from_embs(embs[a], embs[b])
        df = pd.DataFrame(mat, index=names, columns=names)
        df.to_csv(self.metrics_dir / "pairwise_fad.csv")
        return df

    # ---------- plots & report ----------
    def plots(self, quality_df: pd.DataFrame):
        print("[Eval] Plots …")
        plt.style.use("seaborn-v0_8")
        metrics = ["rms", "snr", "spectral_centroid", "dynamic_range", "frequency_diversity", "spectral_bandwidth"]
        n = len(metrics)
        rows, cols = 2, 3
        fig, axes = plt.subplots(rows, cols, figsize=(18, 10))
        for k, m in enumerate(metrics):
            r, c = divmod(k, cols)
            ax = axes[r, c]
            try:
                quality_df.boxplot(column=m, by="model", ax=ax)
                ax.set_title(m.replace("_", " ").title())
                ax.set_xlabel("Model")
                ax.set_ylabel(m.replace("_", " ").title())
            except Exception:
                ax.set_title(f"{m} (unavailable)")
        plt.tight_layout()
        plt.savefig(self.viz_dir / "model_comparison_boxplots.png", dpi=200, bbox_inches="tight")
        plt.close()

    def copy_sample_audio(self) -> Dict[str, List[Tuple[str, str]]]:
        """Copy up to N samples per model into out_dir/samples/<model>/ and return [(fname, relpath)]."""
        if self.html_samples <= 0:
            return {}
        base = self.out_dir / "samples"
        base.mkdir(exist_ok=True)
        mapping = {}
        for name, files in self.model_files.items():
            tgt = base / name
            tgt.mkdir(parents=True, exist_ok=True)
            keep = files[: self.html_samples]
            pairs = []
            for f in keep:
                dest = tgt / f.name
                try:
                    shutil.copy2(f, dest)
                except Exception:
                    # fall back to read/write
                    y, sr = sf.read(f)
                    sf.write(dest, y, sr)
                rel = Path("samples") / name / f.name
                pairs.append((f.name, str(rel)))
            mapping[name] = pairs
        return mapping

    def build_html_report(
        self,
        quality_df: pd.DataFrame,
        clap_df: pd.DataFrame | None,
        fad_df: pd.DataFrame | None,
        samples_map: Dict[str, List[Tuple[str, str]]],
    ):
        # Summary table of means/stds
        wanted = ["rms", "snr", "spectral_centroid", "spectral_bandwidth", "dynamic_range", "frequency_diversity"]
        rows = []
        for name in sorted(self.model_files.keys()):
            sub = quality_df[quality_df["model"] == name]
            row = {"Model": name}
            for m in wanted:
                if m in sub.columns:
                    row[m + " mean"] = np.nanmean(sub[m])
                    row[m + " std"] = np.nanstd(sub[m])
            rows.append(row)
        summary = pd.DataFrame(rows)
        summary.to_csv(self.metrics_dir / "quality_summary.csv", index=False)

        # CLAP summary
        clap_summary_html = ""
        if clap_df is not None and not clap_df.empty:
            cs = clap_df.groupby("model")["clap_similarity"].mean().reset_index()
            clap_summary_html = cs.to_html(index=False, float_format=lambda x: f"{x:.4f}")

        # FAD matrix HTML
        fad_html = fad_df.to_html(float_format=lambda x: f"{x:.4f}") if fad_df is not None else "<p>FAD not computed.</p>"

        # Samples HTML
        sample_blocks = []
        for name, pairs in samples_map.items():
            block = [f"<h3>{name} – Samples</h3>"]
            for fname, rel in pairs:
                block.append(f"""
<div style="margin:8px 0;padding:8px;border:1px solid #eee;">
  <div class="mono">{fname}</div>
  <audio controls preload="none" style="width:100%;">
    <source src="{rel}" type="audio/wav">
    Your browser does not support the audio element.
  </audio>
</div>
""")
            sample_blocks.append("\n".join(block))
        samples_html = "\n".join(sample_blocks) if sample_blocks else "<p>No sample audio embedded.</p>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Multi-Model Retrieval Audio Evaluation</title>
<style>
 body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 20px; }}
 h1, h2, h3 {{ margin: 0.6em 0 0.3em; }}
 table {{ border-collapse: collapse; width: 100%; }}
 th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
 th {{ background: #f6f6f6; }}
 .mono {{ font-family: ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size: 12px; }}
 .grid {{ display:grid; grid-template-columns: 1fr; gap: 16px; }}
</style>
</head>
<body>
<h1>Multi-Model Retrieval Audio Evaluation</h1>
<p class="mono">Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<h2>Quality Summary (means ± std)</h2>
{summary.to_html(index=False, float_format=lambda x: f"{x:.4f}")}

<h2>CLAP Prompt Adherence (mean by model)</h2>
{clap_summary_html if clap_summary_html else "<p>CLAP not computed.</p>"}

<h2>Pairwise FAD (lower is better)</h2>
{fad_html}

<h2>Boxplots</h2>
<p><img src="visualizations/model_comparison_boxplots.png" style="max-width:100%;"/></p>

<h2>Inline Audio Samples</h2>
<div class="grid">
{samples_html}
</div>

</body>
</html>"""
        (self.out_dir / "report.html").write_text(html, encoding="utf-8")
        print(f"[DONE] HTML report → {self.out_dir/'report.html'}")

    # ---------- run all ----------
    def run(self):
        quality = self.compute_quality_metrics()
        diversity = self.compute_diversity()  # saved, not directly plotted here
        clap_df = self.compute_clap_prompt_adherence()
        fad_df = self.compute_pairwise_fad()
        self.plots(quality)
        samples_map = self.copy_sample_audio()
        self.build_html_report(quality, clap_df, fad_df, samples_map)
        print(f"[DONE] Artifacts written to {self.out_dir}")


# ----------------------------
# CLI
# ----------------------------
def parse_models(pairs: List[str]) -> Dict[str, Path]:
    out = {}
    for s in pairs:
        if "=" not in s:
            raise SystemExit(f"--models expects name=dir pairs, got: {s}")
        name, d = s.split("=", 1)
        name = name.strip()
        d = Path(d.strip())
        if not d.exists():
            raise SystemExit(f"Model directory not found: {d}")
        out[name] = d
    return out

def main():
    import argparse
    parser = argparse.ArgumentParser("Multi-Model Audio Evaluation")
    parser.add_argument("--models", nargs="+", required=True, help='One or more "Name=/path/to/wavs" pairs.')
    parser.add_argument("--prompts", required=True, help="Prompts file used for generation.")
    parser.add_argument("--out_dir", required=True, help="Output directory for metrics/report.")
    parser.add_argument("--clap", default="laion/clap-htsat-unfused", help="CLAP checkpoint id.")
    parser.add_argument("--clap-device", default="cuda", choices=["cuda","cpu"])
    parser.add_argument("--fad-device", default="cpu", choices=["cpu","gpu"])
    parser.add_argument("--html_samples", type=int, default=6, help="# of WAVs to embed per model in HTML.")
    args = parser.parse_args()

    # Keep torchvision out of the way for CLAP + Transformers
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

    model_dirs = parse_models(args.models)
    out_dir = Path(args.out_dir)

    evaluator = MultiModelEvaluator(
        model_dirs=model_dirs,
        prompts_file=Path(args.prompts),
        out_dir=out_dir,
        clap_ckpt=args.clap,
        clap_device=args.clap_device,
        fad_device=args.fad_device,
        html_samples=args.html_samples,
    )
    evaluator.run()

if __name__ == "__main__":
    main()