# Retrieval‑Conditioned Music Generation (AudioLDM1/2)

> **TL;DR** — This repo reproduces prompt‑only, prompt+augmentation, and retrieval‑conditioned music generation on **AudioLDM1**, **AudioLDM2**, and **AudioLDM2‑Music**, using **CLAP** embeddings and optional **style adapters**. It includes compact training/evaluation scripts, reproducible configs, and HTML reports.

---

## 1) Repo layout

```
data/
  raw/maestro/ …, raw/urmp/ …, raw/fsd50k/ …
  processed/ …                      # embeddings, banks, manifests, indices
  fsd50k_filter_config.json         # tag/length filters (example)
experiments/
  final/exp*                        # saved runs + reports
scripts/
  exp1_audioldm1_prompt.py
  exp1_audioldm2_prompt.py
  aldm1_retrieval_infer.py
  aldm2_retrieval_infer.py
  exp2_audioldm*_retrieval_augment.py
  train_style2lm_adapter.py
  audio_evaluation_multi.py
src/data/
  aldm_style_adapter.py             # external projector + blend helpers
  build_faiss_from_banks.py, extract_clap_embeddings.py, …
```

Each run writes:
- `run_config.json`, `hardware.json`
- WAVs under `wav/`
- Metrics under `metrics/`
- Visuals under `visualizations/`
- `report.html` (audio players + plots)

---

## 2) Environment

We tested with **PyTorch ≥ 2.1**, **Transformers ≥ 4.42**, **Diffusers 0.35.1**, CUDA 12.x. A minimal env:

```bash
conda env create -f environment.yml   # or install torch, diffusers, transformers, librosa, soundfile
conda activate midi310

# HPC-friendly flags
export TRANSFORMERS_NO_TORCHVISION=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

---

## 3) Data

**MAESTRO** (piano), **URMP** (duos–quartets), **FSD50K** (style bank). Place raw datasets under `data/raw/` in their usual layouts. We render 10‑s segments at 48 kHz for CLAP and at model default SR (16 kHz) for generation.

> If you already have the banks from our runs, you can skip extraction and use the file paths directly.

---

## 4) Precompute CLAP & Banks (optional)

1) **Extract CLAP embeddings** for FSD50K (or your corpus):

```bash
python src/data/extract_clap_embeddings.py   --audio_root data/raw/fsd50k   --out_dir data/processed/retrieval_cache/primary
```

2) **Filter/Audit** and build the **style bank**:

```bash
python src/data/audit_filter_style_bank.py   --embeds data/processed/retrieval_cache/primary/agg_emb_filtered.npy   --mapping data/processed/retrieval_cache/primary/clap_mapping_primary.jsonl
```

3) (Optional) **FAISS** index for fast k‑NN:

```bash
python src/data/build_faiss_from_banks.py   --embeds data/processed/retrieval_cache/primary/agg_emb_filtered.npy   --index_out data/processed/retrieval_cache/primary/bank_flat.index   --metric ip
```

---

## 5) Quick start — Prompt‑only baselines (Exp‑1)

**AudioLDM2 (recommended):**
```bash
PY=python
OUT=experiments/final/exp1b_audioldm2
mkdir -p "$OUT"
$PY scripts/exp1_audioldm2_prompt.py   --model_id cvssp/audioldm2   --prompts_file my_prompts.txt   --out_dir "$OUT" --steps 200 --guidance 3.2 --seconds 10   --visualize --report --seed 42
```

Variants:
- `cvssp/audioldm` (AudioLDM1): `scripts/exp1_audioldm1_prompt.py`
- `cvssp/audioldm2-music`: `scripts/exp1_audioldm2_prompt.py --model_id cvssp/audioldm2-music`

---

## 6) Retrieval‑conditioned inference (Exp‑2)

### A) AudioLDM2 (external adapter or MLP)
```bash
PY=python
SCRIPT=scripts/aldm2_retrieval_infer.py
BANK=data/processed/retrieval_cache/primary/agg_emb_filtered.npy
PROJ=/experiments/optuna_full_ext/projector.final.best.pt
OUT=experiments/final/exp2e_audioldm2_retrieval_external_adapter

$PY "$SCRIPT"   --adapter external --projector "$PROJ"   --model_id cvssp/audioldm2   --prompts_file my_prompts.txt   --style_bank "$BANK"   --out_dir "$OUT"   --blend_mode orth --alpha_style 0.08   --steps 200 --guidance 3.2 --seconds 10 --batch 2   --dtype fp16 --scheduler ddim   --metrics --visualize --report --seed 42
```

To use the internal MLP projector: `--adapter mlp` (no checkpoint needed).

### B) AudioLDM1 (cond‑only safe mix)
```bash
PY=python
SCRIPT=scripts/aldm1_retrieval_infer.py
BANK=data/processed/retrieval_cache/primary/agg_emb_filtered.npy
PROJ=/experiments/optuna_full_ext/projector.final.best.pt
OUT=experiments/final/exp2b_audioldm1_retrieval_external_adapter

$PY "$SCRIPT"   --model_id cvssp/audioldm   --prompts_file my_prompts.txt   --style_bank "$BANK"   --adapter external --projector "$PROJ"   --out_dir "$OUT"   --blend_mode orth --alpha_style 0.03   --cond_mix residual --cond_mix_beta 0.18   --steps 200 --guidance 3.0 --seconds 10 --batch 2   --visualize --report --metrics --seed 42
```

> Tip: If you hear “broken vocal” artefacts, lower `--alpha_style` (0.02–0.06) and `--guidance` (~2.6–3.2), or try `--blend_mode orth` with smaller `--cond_mix_beta` in AudioLDM1.

---

## 7) Style‑adapter training (tiny)

Learn a lightweight 512→H mapping (we keep CLAP space for ALDM1; for ALDM2 it targets LM hidden):

```bash
PY=python
$PY scripts/train_style2lm_adapter.py   --model_id cvssp/audioldm2   --adapter external   --projector /scratch/rc01492/experiments/optuna_full_ext/projector.final.best.pt   --prompts_file my_prompts.txt   --style_bank data/processed/retrieval_cache/primary/agg_emb_filtered.npy   --out_dir experiments/optuna_full_ext   --style_topk 4 --style_temp 0.07   --blend_mode orth --alpha_style 0.04   --batch 16 --epochs 5 --lr 3e-4 --weight_decay 0.01   --seed 42
```

The trainer writes the best checkpoint as `style2lm_adapter.pt` and logs JSON metrics.

---

## 8) Evaluation

Batch evaluation and side‑by‑side comparisons:
```bash
PY=python
$PY scripts/audio_evaluation_multi.py   --roots experiments/final/exp1b_audioldm2 experiments/final/exp2e_audioldm2_retrieval_external_adapter   --out_dir evaluation_results/exp1b_vs_exp2e   --seed 42
```

**Metrics & Plots**
- CLAP similarity (generated vs. prompt text; generated vs. style vector)
- Loudness proxy (RMS dBFS)
- Mel‑spectrogram grids
- Optional FAD (if you provide a reference set; see script flags)

Every run’s `report.html` includes per‑item players and the plots above.

---

## 9) Reproducibility

- Seeds, sampler, steps, CFG, and wall‑clock are captured in `run_config.json` + `hardware.json`.
- Each HTML report embeds the exact CLI and summary stats.
- Style banks are L2‑normalised CLAP vectors in `.npy` format (memory‑mappable).

---

## 10) Notes & Acknowledgements

- Models: `cvssp/audioldm`, `cvssp/audioldm2`, `cvssp/audioldm2-music` (Hugging Face).
- Text/audio embeddings: `laion/clap-htsat-unfused`.
- This repo uses Diffusers pipelines; AudioLDM1 is deprecated upstream, but still works for research.
- Cite dataset licenses (MAESTRO, URMP, FSD50K) when redistributing derived artefacts.