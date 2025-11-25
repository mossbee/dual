"""
Metric helpers for classification and ReID.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None


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


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between feature vectors.
    
    Args:
        a: Tensor of shape (N, D) or (D,)
        b: Tensor of shape (M, D) or (D,)
    
    Returns:
        Similarity scores. If a is (N, D) and b is (M, D), returns (N, M).
        If both are 1D, returns scalar.
    """
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)
    return a_norm @ b_norm.t()


def euclidean_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute similarity from euclidean distance.
    Converts distance to similarity using 1 / (1 + distance).
    
    Args:
        a: Tensor of shape (N, D) or (D,)
        b: Tensor of shape (M, D) or (D,)
    
    Returns:
        Similarity scores. If a is (N, D) and b is (M, D), returns (N, M).
        If both are 1D, returns scalar.
    """
    dist = pairwise_distance(a, b)
    return 1.0 / (1.0 + dist)


def verification_metrics(similarities: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Compute comprehensive verification metrics.
    
    Args:
        similarities: Tensor of shape (N,) containing similarity scores for pairs
        labels: Tensor of shape (N,) containing binary labels (0=different, 1=same)
    
    Returns:
        Dictionary with metrics:
        - best_acc: Best accuracy with optimal threshold
        - best_threshold: Threshold that yields best accuracy
        - eer: Equal Error Rate
        - eer_threshold: Threshold at EER
        - roc_auc: Area under ROC curve (NaN if sklearn not available)
        - tar: True Accept Rate at EER threshold
        - far: False Accept Rate at EER threshold
    """
    similarities_np = similarities.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Sort similarities and labels together
    sort_idx = np.argsort(similarities_np)
    sorted_sims = similarities_np[sort_idx]
    sorted_labels = labels_np[sort_idx]
    
    # Compute best accuracy by sweeping thresholds
    num_pairs = len(similarities_np)
    num_genuine = labels_np.sum()
    num_impostor = num_pairs - num_genuine
    
    best_acc = 0.0
    best_threshold = 0.0
    
    # Try thresholds at each similarity value and midpoints
    thresholds = np.concatenate([sorted_sims, (sorted_sims[:-1] + sorted_sims[1:]) / 2])
    thresholds = np.unique(thresholds)
    
    for threshold in thresholds:
        preds = (similarities_np >= threshold).astype(int)
        acc = (preds == labels_np).mean()
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    
    # Compute EER (Equal Error Rate)
    # EER is where FAR = FRR (False Accept Rate = False Reject Rate)
    # FAR = false positives / total impostors
    # FRR = false negatives / total genuine = (1 - TAR)
    # TAR = true positives / total genuine
    
    min_diff = float("inf")
    eer = 1.0
    eer_threshold = 0.0
    tar_at_eer = 0.0
    far_at_eer = 0.0
    
    for threshold in thresholds:
        # Predictions: 1 if similarity >= threshold (accept), 0 otherwise (reject)
        preds = (similarities_np >= threshold).astype(int)
        
        # True positives: predicted 1 and label 1
        tp = ((preds == 1) & (labels_np == 1)).sum()
        # False positives: predicted 1 and label 0
        fp = ((preds == 1) & (labels_np == 0)).sum()
        # False negatives: predicted 0 and label 1
        fn = ((preds == 0) & (labels_np == 1)).sum()
        
        if num_impostor > 0:
            far = fp / num_impostor
        else:
            far = 0.0
        
        if num_genuine > 0:
            tar = tp / num_genuine
            frr = fn / num_genuine
        else:
            tar = 0.0
            frr = 0.0
        
        # Find threshold where FAR ≈ FRR (minimize the difference)
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2.0  # Average of FAR and FRR at EER
            eer_threshold = threshold
            tar_at_eer = tar
            far_at_eer = far
    
    # Compute ROC-AUC
    roc_auc = float("nan")
    if roc_auc_score is not None:
        try:
            roc_auc = float(roc_auc_score(labels_np, similarities_np))
        except Exception:
            pass
    
    return {
        "best_acc": float(best_acc),
        "best_threshold": float(best_threshold),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "roc_auc": roc_auc,
        "tar": float(tar_at_eer),
        "far": float(far_at_eer),
    }


