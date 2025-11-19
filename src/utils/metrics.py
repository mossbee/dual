"""
Metric helpers for classification and ReID.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch


def fused_accuracy(sa_logits: torch.Tensor, glca_logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = torch.softmax(sa_logits, dim=-1) + torch.softmax(glca_logits, dim=-1)
    preds = probs.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def pairwise_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_sq = (a**2).sum(dim=1, keepdim=True)
    b_sq = (b**2).sum(dim=1, keepdim=True).t()
    dist = a_sq + b_sq - 2 * a @ b.t()
    return dist.clamp(min=1e-12).sqrt()


def reid_metrics(
    query_feats: torch.Tensor,
    query_pids: Sequence[int],
    query_camids: Sequence[int],
    gallery_feats: torch.Tensor,
    gallery_pids: Sequence[int],
    gallery_camids: Sequence[int],
) -> Dict[str, float]:
    distmat = pairwise_distance(query_feats, gallery_feats).cpu().numpy()
    q_pids = np.asarray(query_pids)
    g_pids = np.asarray(gallery_pids)
    q_cam = np.asarray(query_camids)
    g_cam = np.asarray(gallery_camids)
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[None, :] == q_pids[:, None])
    all_cmc = []
    all_ap = []
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_cam[q_idx]
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_cam[order] == q_camid)
        order = order[~remove]
        if order.size == 0:
            continue
        match = matches[q_idx][order]
        if not np.any(match):
            continue
        cmc = match.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc)
        rel = np.where(match == 1)[0]
        precisions = [(match[:r + 1].sum() / (r + 1)) for r in rel]
        all_ap.append(np.mean(precisions))
    if not all_cmc:
        raise RuntimeError("No valid queries for ReID eval.")
    max_len = max(len(cmc) for cmc in all_cmc)
    summed = np.zeros(max_len)
    for cmc in all_cmc:
        summed[: len(cmc)] += cmc
    mean_cmc = summed / len(all_cmc)
    mAP = float(np.mean(all_ap))
    rank1 = float(mean_cmc[0])
    return {"mAP": mAP, "Rank-1": rank1}


