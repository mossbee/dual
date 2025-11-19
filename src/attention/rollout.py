"""
Attention rollout adapted from refs/vit_rollout.py.
"""
from __future__ import annotations

from typing import Iterable, Literal

import torch


HeadFusion = Literal["mean", "max", "min"]


def fuse_heads(attn: torch.Tensor, mode: HeadFusion) -> torch.Tensor:
    if mode == "mean":
        return attn.mean(dim=1)
    if mode == "max":
        return attn.max(dim=1)[0]
    if mode == "min":
        return attn.min(dim=1)[0]
    raise ValueError(f"Unsupported head_fusion {mode}")


def rollout(
    attentions: Iterable[torch.Tensor],
    head_fusion: HeadFusion = "mean",
    add_residual: bool = True,
) -> torch.Tensor:
    """
    Multiply attention matrices across layers following Eq. (2) in refs/dcal.md.

    Args:
        attentions: iterable of tensors shaped (B, H, T, T)
        head_fusion: fusion method before rollout
        add_residual: whether to blend attention with identity matrix
    Returns:
        Tensor with shape (B, T, T) containing accumulated attention.
    """
    attentions = list(attentions)
    if not attentions:
        raise ValueError("Need at least one attention map for rollout.")
    device = attentions[0].device
    result = None
    for attn in attentions:
        fused = fuse_heads(attn, head_fusion)
        if add_residual:
            eye = torch.eye(fused.size(-1), device=device)
            fused = 0.5 * fused + 0.5 * eye
        fused = fused / fused.sum(dim=-1, keepdim=True)
        result = fused if result is None else fused @ result
    return result


def topk_local_indices(rollout_map: torch.Tensor, ratio: float) -> torch.Tensor:
    """
    Args:
        rollout_map: (B, T, T) matrix returned by rollout()
        ratio: fraction of patch tokens to keep (0, 1]
    Returns:
        Long tensor of shape (B, K) containing selected indices (>0 to avoid CLS).
    """
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    cls_to_patches = rollout_map[:, 0, 1:]
    k = max(1, int(cls_to_patches.size(-1) * ratio))
    scores, idx = torch.topk(cls_to_patches, k=k, dim=-1)
    # shift by 1 to compensate for removed CLS
    return idx + 1


