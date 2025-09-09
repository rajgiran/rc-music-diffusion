# train_style2lm_adapter.py
# -*- coding: utf-8 -*-
"""
Train a tiny Style→LM adapter that maps blended CLAP(512) → LM hidden (H_lm) with seq length S_text.

Key points:
- Dataset stores ONLY CPU tensors/arrays (no CUDA in DataLoader workers).
- Proper Subset-based DataLoaders.
- Robust teacher fallback (no LM needed; uses text-encoder features with a fixed projection).
- Early stopping + clean logs/plots/run_config.

Req: diffusers>=0.31, transformers>=4.42, torch>=2.1
"""

from __future__ import annotations
import os, json, math, argparse, random, platform, logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ───────── hygiene ─────────
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

log = logging.getLogger("style2lm_train")
def setup_logging(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out_dir / "train.log", encoding="utf-8")
    sh = logging.StreamHandler()
    logging.basicConfig(level=logging.INFO, handlers=[fh, sh],
                        format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")

# ───────── deps ─────────
from transformers import ClapModel, ClapProcessor
from diffusers import AudioLDM2Pipeline

# ───────── optional external projector/blend ─────────
_has_external = False
ExternalClapAudioProjector = None
ext_blend_text_and_style = None
for mod in ("data.aldm2_style_adapter", "aldm2_style_adapter"):
    try:
        m = __import__(mod, fromlist=["ClapAudioProjector", "blend_text_and_style"])
        ExternalClapAudioProjector = getattr(m, "ClapAudioProjector")
        ext_blend_text_and_style = getattr(m, "blend_text_and_style")
        _has_external = True
        break
    except Exception:
        pass

# ───────── fallbacks ─────────
def _l2(x): return x / (x.norm(dim=-1, keepdim=True) + 1e-9)

def fallback_blend(zt: torch.Tensor, zs: torch.Tensor, mode="orth", alpha=0.04) -> torch.Tensor:
    if mode == "orth":
        proj = (zs * zt).sum(dim=-1, keepdim=True) * zt
        mixed = zt + alpha * (zs - proj)
    elif mode == "residual":
        mixed = zt + alpha * zs
    else:
        mixed = (1 - alpha) * zt + alpha * zs
    return _l2(mixed)

class MLPProjector(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.fc1=nn.Linear(512,512); self.act=nn.GELU()
        self.drop=nn.Dropout(dropout) if dropout and dropout>0 else nn.Identity()
        self.fc2=nn.Linear(512,512); self.ln=nn.LayerNorm(512)
    def forward(self,x): return F.normalize(self.ln(self.fc2(self.drop(self.act(self.fc1(x))))), dim=-1)

# ───────── adapter ─────────
class Style2LMAdapter(nn.Module):
    def __init__(self, hidden:int, seq_len:int, width:int=512, conv_kernel:int=3, dropout:float=0.0):
        super().__init__()
        self.hidden = hidden
        self.seq_len = seq_len
        self.inp = nn.Linear(width, hidden)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, hidden))
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.dw = nn.Conv1d(hidden, hidden, kernel_size=conv_kernel, padding=conv_kernel//2, groups=hidden)
        self.pw = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.ln = nn.LayerNorm(hidden)
        nn.init.trunc_normal_(self.pos, std=0.02)
    def forward(self, z512: torch.Tensor) -> torch.Tensor:
        x=self.inp(z512).unsqueeze(1).expand(-1,self.seq_len,-1)
        x=x+self.pos
        x=self.drop(x)
        y=x.transpose(1,2); y=self.pw(self.dw(y)); y=y.transpose(1,2)
        return self.ln(y)

# ───────── utils ─────────
def read_prompts(p: Path) -> List[str]:
    txt = p.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.replace("\ufeff", "").strip() for ln in txt.splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]

@torch.no_grad()
def clap_text(texts, proc, clap, device):
    ins = proc(text=texts, return_tensors="pt", padding=True).to(device)
    z = clap.get_text_features(**ins)
    return F.normalize(z, dim=-1)

def retrieval_select(style_bank: np.ndarray, Zt: torch.Tensor, topk=4, temp=0.07) -> Tuple[np.ndarray, Dict[str, Any]]:
    N = Zt.size(0); M = style_bank.shape[0]
    bank = torch.from_numpy(style_bank.astype(np.float32)).to(Zt.device)
    bank = F.normalize(bank, dim=-1)
    sims = (Zt @ bank.t())
    K = min(topk, M)
    vals, idx = torch.topk(sims, k=K, dim=1)
    w = torch.softmax(vals / max(temp, 1e-6), dim=1)
    agg = torch.einsum("nk,nkh->nh", w, bank[idx])
    agg = F.normalize(agg, dim=-1).detach().cpu().numpy().astype(np.float32)
    meta = {"indices": idx.cpu().tolist(), "weights": w.cpu().tolist(), "topk": int(K), "temp": float(temp)}
    return agg, meta

def save_plot(points: List[float], out: Path, title: str, ylabel: str):
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(points)
    plt.title(title); plt.xlabel("epoch"); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

# ───────── dataset (CPU only!) ─────────
class StyleDataset(Dataset):
    def __init__(self, prompts: List[str], z512: np.ndarray):
        self.prompts = prompts
        self.z512 = z512.astype(np.float32)  # (N,512) CPU
    def __len__(self): return len(self.prompts)
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx],
                "z512": torch.from_numpy(self.z512[idx])}  # torch CPU tensor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="cvssp/audioldm2")
    ap.add_argument("--adapter", choices=["external","mlp"], default="external")
    ap.add_argument("--projector", type=Path, default=None)
    ap.add_argument("--prompts_file", type=Path, required=True)
    ap.add_argument("--style_bank", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)

    ap.add_argument("--style_topk", type=int, default=4)
    ap.add_argument("--style_temp", type=float, default=0.07)
    ap.add_argument("--blend_mode", choices=["orth","residual","concat"], default="orth")
    ap.add_argument("--alpha_style", type=float, default=0.04)

    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--use_teacher", action="store_true",
                    help="Try to use pipeline.encode_prompt teacher targets. Falls back if incompatible.")
    args = ap.parse_args()
    setup_logging(args.out_dir)

    # persist config
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir/"run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    # seeds
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # env snapshot
    snap = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "diffusers": __import__("diffusers").__version__,
        "transformers": __import__("transformers").__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "device": device,
    }
    (args.out_dir/"env.json").write_text(json.dumps(snap, indent=2), encoding="utf-8")
    log.info(f"env: {snap}")

    # load CLAP
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused", use_safetensors=True).to(device).eval()

    # prompts
    prompts = read_prompts(args.prompts_file)
    N = len(prompts)
    if N == 0: raise ValueError("No prompts.")
    log.info(f"prompts: {N}")

    with torch.no_grad():
        Zt = clap_text(prompts, proc, clap, device)  # (N,512) on device

    # style bank
    bank = np.load(args.style_bank).astype(np.float32)
    bank /= (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-9)

    # retrieval selection
    style_rows, meta = retrieval_select(bank, Zt, topk=args.style_topk, temp=args.style_temp)
    (args.out_dir/"selection.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # projector
    if args.adapter == "external":
        if not _has_external:
            raise RuntimeError("Adapter 'external' requested but aldm2_style_adapter not found.")
        projector = ExternalClapAudioProjector().to(device).eval()
    else:
        projector = MLPProjector().to(device).eval()

    if args.projector and Path(args.projector).exists():
        sd = torch.load(args.projector, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = { (k[7:] if k.startswith("module.") else k): v for k,v in sd.items() }
        projector.load_state_dict(sd, strict=False)
        log.info(f"Loaded projector: {args.projector}")

    with torch.no_grad():
        Za = torch.from_numpy(style_rows).to(device)      # (N,512) on device
        Zs = projector(Za)                                # (N,512)
        if _has_external:
            Zm = ext_blend_text_and_style(Zt, Zs, mode=args.blend_mode, alpha=args.alpha_style, renorm=True)
        else:
            Zm = fallback_blend(Zt, Zs, mode=args.blend_mode, alpha=args.alpha_style)

    # pipeline for sizes / teacher
    pipe = AudioLDM2Pipeline.from_pretrained(args.model_id).to(device)
    if hasattr(pipe, "safety_checker"): pipe.safety_checker = None

    lm_cfg = getattr(pipe, "language_model", None)
    lm_hidden = 768
    if lm_cfg is not None and getattr(lm_cfg, "config", None) is not None:
        lm_hidden = getattr(lm_cfg.config, "n_embd", None) or getattr(lm_cfg.config, "hidden_size", 768)

    tok = getattr(pipe, "tokenizer_2", None)
    te2 = getattr(pipe, "text_encoder_2", None)
    assert tok is not None and te2 is not None, "Pipeline missing tokenizer_2/text_encoder_2."
    max_len = getattr(tok, "model_max_length", 512)

    with torch.no_grad():
        ex_tok = tok([prompts[0]], return_tensors="pt", padding="max_length", truncation=True, max_length=max_len).to(device)
        te_example = te2(**ex_tok).last_hidden_state  # (1,S_text,H_text)
        S_text = te_example.shape[1]
        H_text = te_example.shape[2]

    # adapter
    adapter = Style2LMAdapter(hidden=lm_hidden, seq_len=S_text, dropout=0.05).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # dataset (CPU) — use blended z512 as inputs
    ds = StyleDataset(prompts, Zm.detach().cpu().numpy())
    idxs = np.arange(N); np.random.shuffle(idxs)
    split = max(1, int(0.9 * N))
    train_ds = Subset(ds, indices=idxs[:split])
    val_ds   = Subset(ds, indices=idxs[split:])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=False)

    # teacher fallback: **fix dims** — Tproj is (H_text, H_lm)
    g = torch.Generator(device=device).manual_seed(1234)
    Tproj = torch.empty(H_text, lm_hidden, device=device)  # FIXED ORDER
    with torch.no_grad():
        limit = 1.0 / math.sqrt(H_text)
        Tproj.uniform_(-limit, limit, generator=g)

    def pseudo_teacher(texts: List[str]) -> torch.Tensor:
        tok_pos = tok(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len).to(device)
        with torch.no_grad():
            te_pos = te2(**tok_pos).last_hidden_state  # (B,S,H_text)
        # (B,S,H_text) x (H_text,H_lm) -> (B,S,H_lm)
        tgt = torch.einsum("bsh,hk->bsk", te_pos, Tproj)
        return tgt

    teacher_ok = False
    if args.use_teacher:
        try:
            _ = pipe.encode_prompt(["x"], device=device, num_waveforms_per_prompt=1,
                                   do_classifier_free_guidance=True, negative_prompt=[""])
            teacher_ok = True
            log.info("Teacher path ENABLED (encode_prompt works).")
        except Exception as e:
            log.warning(f"Teacher path disabled: {e}")

    def teacher_targets(texts: List[str]) -> torch.Tensor:
        if not teacher_ok:
            return pseudo_teacher(texts)
        try:
            enc = pipe.encode_prompt(texts, device=device, num_waveforms_per_prompt=1,
                                     do_classifier_free_guidance=True, negative_prompt=[""]*len(texts))
            tensors = [t for t in enc if isinstance(t, torch.Tensor)]
            cand = [t for t in tensors if t.ndim==3 and t.shape[-1]==lm_hidden]
            if len(cand)==0: return pseudo_teacher(texts)
            return cand[-1].detach()
        except Exception:
            return pseudo_teacher(texts)

    def batch_loss(z512_cpu: torch.Tensor, texts: List[str]) -> torch.Tensor:
        z512 = z512_cpu.to(device, non_blocking=True)
        y_hat = adapter(z512)                 # (B,S,H_lm)
        y = teacher_targets(texts)            # (B,S,H_lm)
        mse = F.mse_loss(y_hat, y)
        cos = 1.0 - F.cosine_similarity(y_hat.reshape(y_hat.size(0), -1),
                                        y.reshape(y.size(0), -1), dim=-1).mean()
        return mse + 0.5 * cos

    best = float("inf"); best_path = args.out_dir/"style2lm_adapter.pt"
    train_losses, val_losses = [], []
    patience_left = args.patience

    for epoch in range(1, args.epochs+1):
        adapter.train(); tl = 0.0; n = 0
        for batch in tqdm(train_loader, desc=f"train {epoch}"):
            texts = batch["prompt"]; z512 = batch["z512"]
            loss = batch_loss(z512, texts)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tl += loss.item() * z512.size(0); n += z512.size(0)
        train_losses.append(tl/n)

        adapter.eval(); vl = 0.0; n = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"val {epoch}"):
                texts = batch["prompt"]; z512 = batch["z512"]
                loss = batch_loss(z512, texts)
                vl += loss.item() * z512.size(0); n += z512.size(0)
        val_losses.append(vl/n)

        log.info(f"[epoch {epoch}] train {train_losses[-1]:.4f} | val {val_losses[-1]:.4f}")

        improved = val_losses[-1] < best - 1e-5
        if improved:
            best = val_losses[-1]
            torch.save({"state_dict": adapter.state_dict(),
                        "hidden": lm_hidden, "seq_len": int(S_text)}, best_path)
            log.info(f"saved best → {best_path}")
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info("Early stopping.")
                break

    save_plot(train_losses, args.out_dir/"train_loss.png", "Train loss", "loss")
    save_plot(val_losses,   args.out_dir/"val_loss.png",   "Val loss",   "loss")

    summary = {"best_val": best, "epochs": len(train_losses), "lm_hidden": lm_hidden, "seq_len": int(S_text)}
    (args.out_dir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(f"Done. Best val: {best:.4f}")

if __name__ == "__main__":
    main()