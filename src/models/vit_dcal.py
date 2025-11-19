"""
Dual Cross-Attention Learning wrapper around ViT-B/16.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import nn

from attention.rollout import rollout, topk_local_indices
from models.vit_backbone import ViTBackbone, ViTConfig, vit_b16_config


class CrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, kv: torch.Tensor):
        B, Nq, C = query.shape
        _, Nk, _ = kv.shape
        q = self.q(query).reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(kv).reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        v = self.v(kv).reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = torch.matmul(q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, Nq, C)
        out = self.out(out)
        return out, attn


class GlobalLocalCrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross = CrossAttention(hidden_size, num_heads, dropout)
        self.out_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, local_tokens: torch.Tensor, global_tokens: torch.Tensor) -> torch.Tensor:
        out, _ = self.cross(self.norm(local_tokens), global_tokens)
        return self.out_norm(local_tokens + out)


class PairwiseCrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross = CrossAttention(hidden_size, num_heads, dropout)

    def forward(self, anchor: torch.Tensor, distractor: torch.Tensor) -> torch.Tensor:
        kv = torch.cat([anchor, distractor], dim=1)
        out, _ = self.cross(self.norm(anchor), kv)
        return anchor + out


def gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # indices shape (B, K)
    idx = indices.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
    return torch.gather(tokens, dim=1, index=idx)


@dataclass
class DCALConfig:
    img_size: int
    num_classes: int
    local_ratio: float = 0.1
    head_fusion: str = "mean"
    drop_path_rate: float = 0.1
    backbone: ViTConfig = field(default_factory=vit_b16_config)


class DCALViT(nn.Module):
    def __init__(self, cfg: DCALConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = ViTBackbone(cfg.backbone, cfg.img_size, drop_path_rate=cfg.drop_path_rate)
        hidden = cfg.backbone.hidden_size
        self.sa_head = nn.Linear(hidden, cfg.num_classes)
        self.glca = GlobalLocalCrossAttention(hidden, cfg.backbone.num_heads, cfg.backbone.attention_dropout_rate)
        self.glca_head = nn.Linear(hidden, cfg.num_classes)
        self.pwca = PairwiseCrossAttention(hidden, cfg.backbone.num_heads, cfg.backbone.attention_dropout_rate)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        images: torch.Tensor,
        *,
        local_ratio: Optional[float] = None,
        enable_glca: bool = True,
        enable_pwca: bool = False,
        pair_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        seq, attn_stack = self.backbone(images, return_attn=enable_glca)
        cls = seq[:, 0]
        outputs: Dict[str, torch.Tensor] = {
            "tokens": seq,
            "cls": cls,
            "sa_logits": self.sa_head(cls),
        }
        if enable_glca:
            ratio = local_ratio or self.cfg.local_ratio
            rollout_map = rollout(attn_stack, head_fusion=self.cfg.head_fusion)
            idx = topk_local_indices(rollout_map, ratio)
            local_tokens = gather_tokens(seq, idx)
            fused = self.glca(local_tokens, seq)
            glca_repr = fused.mean(dim=1)
            outputs["glca_logits"] = self.glca_head(glca_repr)
            outputs["glca_tokens"] = fused
            outputs["glca_repr"] = glca_repr
        if enable_pwca:
            if pair_indices is None:
                pair_indices = torch.randperm(seq.size(0), device=seq.device)
            distractor = seq[pair_indices]
            fused = self.pwca(seq, distractor)
            outputs["pwca_logits"] = self.sa_head(fused[:, 0])
        return outputs

    def load_pretrained(self, npz_path: str):
        self.backbone.load_from_npz(npz_path)


