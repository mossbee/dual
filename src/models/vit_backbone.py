"""
Minimal ViT-B/16 backbone adapted from refs/ViT-pytorch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from attention.stochastic_depth import StochasticDepth


def np2th(weights: np.ndarray, conv: bool = False) -> torch.Tensor:
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


@dataclass
class ViTConfig:
    patch_size: int = 16
    hidden_size: int = 768
    mlp_dim: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    attention_dropout_rate: float = 0.0
    dropout_rate: float = 0.1


def vit_b16_config() -> ViTConfig:
    return ViTConfig()


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, patch_size: int, hidden_size: int, in_channels: int = 3):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size))
        self.dropout = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.proj(x)  # (B, C, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        return self.dropout(x)

    def resize_pos_embed(self, posemb: torch.Tensor):
        if posemb.shape == self.pos_embed.shape:
            self.pos_embed.data.copy_(posemb)
            return
        cls_posemb = posemb[:, :1]
        patch_posemb = posemb[:, 1:]
        num_patches = patch_posemb.shape[1]
        side = int(math.sqrt(num_patches))
        patch_posemb = patch_posemb.reshape(1, side, side, -1).permute(0, 3, 1, 2)
        new_side = int(math.sqrt(self.num_patches))
        patch_posemb = F.interpolate(patch_posemb, size=(new_side, new_side), mode="bilinear", align_corners=False)
        patch_posemb = patch_posemb.permute(0, 2, 3, 1).reshape(1, new_side * new_side, -1)
        self.pos_embed.data.copy_(torch.cat([cls_posemb, patch_posemb], dim=1))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.all_dim = hidden_size
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn


class MLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int, dropout_rate: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: ViTConfig, drop_path: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = MultiHeadSelfAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            dropout_rate=config.attention_dropout_rate,
        )
        self.mlp = MLP(config.hidden_size, config.mlp_dim, config.dropout_rate)
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class ViTBackbone(nn.Module):
    def __init__(self, config: ViTConfig, img_size: int, drop_path_rate: float = 0.0):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(img_size, config.patch_size, config.hidden_size)
        dpr = torch.linspace(0, drop_path_rate, steps=config.num_layers).tolist()
        self.blocks = nn.ModuleList([TransformerBlock(config, drop) for drop in dpr])
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.patch_embed(x)
        attn_collector: List[torch.Tensor] = []
        for block in self.blocks:
            x, attn = block(x)
            if return_attn:
                attn_collector.append(attn)
        x = self.norm(x)
        return x, attn_collector

    def load_from_npz(self, npz_path: str):
        weights = np.load(npz_path, allow_pickle=True)
        with torch.no_grad():
            self.patch_embed.proj.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.patch_embed.proj.bias.copy_(np2th(weights["embedding/bias"]))
            self.patch_embed.cls_token.copy_(np2th(weights["cls"]))
            self.patch_embed.resize_pos_embed(np2th(weights["Transformer/posembed_input/pos_embedding"]))
            self.norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            for idx, block in enumerate(self.blocks):
                root = f"Transformer/encoderblock_{idx}"
                attn = block.attn
                attn.qkv.weight.copy_(
                    torch.cat(
                        [
                            np2th(weights[f"{root}/MultiHeadDotProductAttention_1/query/kernel"]).view(-1, self.config.hidden_size).t(),
                            np2th(weights[f"{root}/MultiHeadDotProductAttention_1/key/kernel"]).view(-1, self.config.hidden_size).t(),
                            np2th(weights[f"{root}/MultiHeadDotProductAttention_1/value/kernel"]).view(-1, self.config.hidden_size).t(),
                        ]
                    )
                )
                attn.qkv.bias.copy_(
                    torch.cat(
                        [
                            np2th(weights[f"{root}/MultiHeadDotProductAttention_1/query/bias"]).view(-1),
                            np2th(weights[f"{root}/MultiHeadDotProductAttention_1/key/bias"]).view(-1),
                            np2th(weights[f"{root}/MultiHeadDotProductAttention_1/value/bias"]).view(-1),
                        ]
                    )
                )
                attn.proj.weight.copy_(np2th(weights[f"{root}/MultiHeadDotProductAttention_1/out/kernel"]).view(self.config.hidden_size, self.config.hidden_size).t())
                attn.proj.bias.copy_(np2th(weights[f"{root}/MultiHeadDotProductAttention_1/out/bias"]).view(-1))

                block.norm1.weight.copy_(np2th(weights[f"{root}/LayerNorm_0/scale"]))
                block.norm1.bias.copy_(np2th(weights[f"{root}/LayerNorm_0/bias"]))
                block.norm2.weight.copy_(np2th(weights[f"{root}/LayerNorm_2/scale"]))
                block.norm2.bias.copy_(np2th(weights[f"{root}/LayerNorm_2/bias"]))

                block.mlp.fc1.weight.copy_(np2th(weights[f"{root}/MlpBlock_3/Dense_0/kernel"]).t())
                block.mlp.fc1.bias.copy_(np2th(weights[f"{root}/MlpBlock_3/Dense_0/bias"]).t())
                block.mlp.fc2.weight.copy_(np2th(weights[f"{root}/MlpBlock_3/Dense_1/kernel"]).t())
                block.mlp.fc2.bias.copy_(np2th(weights[f"{root}/MlpBlock_3/Dense_1/bias"]).t())


