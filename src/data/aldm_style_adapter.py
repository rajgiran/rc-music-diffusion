# aldm2_style_adapter.py
# -*- coding: utf-8 -*-
import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Projector: CLAP-audio(512) -> "text space" (512)
# -----------------------------
class ClapAudioProjector(nn.Module):
    """
    Lightweight MLP projector:
        512 -> 1024 (SiLU) -> 512 (LayerNorm)
    Trains fast with contrastive loss (audio->text).
    """
    def __init__(self, in_dim: int = 512, hidden: int = 1024, out_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden, out_dim)
        self.ln  = nn.LayerNorm(out_dim)

        # Kaiming init
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 512]
        x = self.fc2(self.act(self.fc1(x)))
        x = self.ln(x)
        x = F.normalize(x, dim=-1)
        return x


# -----------------------------
# Blend utilities
# -----------------------------
def _orthogonal_component(v: torch.Tensor, ref: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Remove the component of v along ref direction.
    v, ref: [B, D] (L2 normalized)
    """
    # cosine projection: (v·ref) ref
    dot = (v * ref).sum(dim=-1, keepdim=True)  # [B,1]
    return F.normalize(v - dot * ref, dim=-1, eps=eps)


@torch.no_grad()
def debug_blend_stats(text: torch.Tensor, style: torch.Tensor) -> Tuple[float, float, float]:
    # % only for debug print
    # returns: mean ||text||, mean ||style||, mean ||text-style||
    return (
        text.norm(dim=-1).mean().item(),
        style.norm(dim=-1).mean().item(),
        (text - style).norm(dim=-1).mean().item(),
    )


def blend_text_and_style(
    text_emb: torch.Tensor,               # [B, 512], L2 normalized (CLAP text)
    style_emb: torch.Tensor,              # [B, 512], L2 normalized (projected CLAP audio)
    mode: Literal["residual", "orth", "concat"] = "residual",
    alpha: float = 0.10,                  # style weight
    renorm: bool = True
) -> torch.Tensor:
    """
    Returns a single [B,512] embedding vector for the pipeline.
    - residual: prompt = text + α * style
    - orth:     prompt = text + α * (style ⟂ text)
    - concat:   approximated as residual (single vector route). If you later
                add a true sequence-cond path, you can switch concat->seq.
    """
    if mode == "orth":
        style_ortho = _orthogonal_component(style_emb, text_emb)
        mixed = F.normalize(text_emb + alpha * style_ortho, dim=-1)
    else:
        # residual and concat (approximate)
        mixed = text_emb + alpha * style_emb
        mixed = F.normalize(mixed, dim=-1) if renorm else mixed
    return mixed