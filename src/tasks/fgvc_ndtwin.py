"""
Training script for DCAL on NDTWIN (NDTWIN).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import logging

import torch
from torch import nn

from datasets import NDTWINDataset, NDTWINUniqueImageDataset, NDTWINVerificationDataset
from losses.uncertainty import UncertaintyWeighting
from models.vit_dcal import DCALConfig, DCALViT
from utils.data import build_ndtwin_transforms, build_loader
from utils.metrics import fused_accuracy, cosine_similarity, euclidean_similarity, verification_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train DCAL on NDTWIN-2009-2010")
    parser.add_argument("--data-root", default="data/NDTWIN-2009-2010", type=str)
    parser.add_argument("--weights", default="weights/ViT-B_16.npz", type=str)
    parser.add_argument("--output", default="runs/fgvc_ndtwin", type=str)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--eval-batch-size", default=32, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--local-ratio", default=0.1, type=float)
    parser.add_argument("--drop-path", default=0.1, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--log-interval", default=50, type=int, help="log training metrics every N batches")
    parser.add_argument("--val-interval", default=1, type=int, help="run validation every N epochs")
    parser.add_argument("--log-file", default="training.log", type=str, help="name of log file stored under output dir")
    parser.add_argument(
        "--similarity-metric",
        default="cosine",
        choices=["cosine", "euclidean"],
        type=str,
        help="Similarity metric for verification: 'cosine' or 'euclidean'",
    )
    return parser.parse_args()


def setup_logger(log_file: Path) -> logging.Logger:
    """
    Configure a logger that writes both to stdout and a file.
    """
    logger = logging.getLogger("fgvc_ndtwin")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def train_one_epoch(
    model,
    uncertainty,
    loader,
    optimizer,
    criterion,
    device,
    epoch: int,
    log_interval: int,
    logger: logging.Logger,
    global_step_start: int,
):
    model.train()
    uncertainty.train()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    for step_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        perm = torch.randperm(images.size(0), device=device)
        outputs = model(images, enable_glca=True, enable_pwca=True, pair_indices=perm)
        loss_sa = criterion(outputs["sa_logits"], labels)
        loss_glca = criterion(outputs["glca_logits"], labels)
        loss_pwca = criterion(outputs["pwca_logits"], labels)
        loss = uncertainty({"sa": loss_sa, "glca": loss_glca, "pwca": loss_pwca})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_acc = fused_accuracy(outputs["sa_logits"], outputs["glca_logits"], labels)
        total_acc += batch_acc
        steps += 1

        if log_interval > 0 and (step_idx % log_interval == 0 or step_idx == 1):
            global_step = global_step_start + step_idx
            log_msg = (
                f"[Epoch {epoch:03d} | Step {step_idx:04d} | Global {global_step:06d}] "
                f"loss={loss.item():.4f} batch_acc={batch_acc:.4f} lr={optimizer.param_groups[0]['lr']:.6f}"
            )
            logger.info(log_msg)
    return {"loss": total_loss / steps, "acc": total_acc / steps}


@torch.no_grad()
def evaluate(model, data_root: str, device: torch.device, similarity_metric: str, eval_batch_size: int, num_workers: int):
    """
    Evaluate model on verification task using test pairs.
    
    Args:
        model: DCAL model
        data_root: Root directory of dataset
        device: Device to run on
        similarity_metric: 'cosine' or 'euclidean'
        eval_batch_size: Batch size for feature extraction
        num_workers: Number of workers for DataLoader
    
    Returns:
        Dictionary with verification metrics
    """
    model.eval()
    
    # Load verification pairs
    verif_dataset = NDTWINVerificationDataset(data_root, transform=build_ndtwin_transforms(False))
    
    # Get all unique images
    unique_image_paths = verif_dataset.get_unique_images()
    
    # Extract features for all unique images
    unique_image_dataset = NDTWINUniqueImageDataset(unique_image_paths, transform=build_ndtwin_transforms(False))
    unique_loader = build_loader(
        unique_image_dataset, batch_size=eval_batch_size, num_workers=num_workers, shuffle=False
    )
    
    # Cache features
    feature_cache = {}
    for batch in unique_loader:
        images = batch["image"].to(device)
        paths = batch["path"]
        outputs = model(images, enable_glca=True, enable_pwca=False)
        # Use fused representation: concat cls and glca_repr
        fused_feat = torch.cat([outputs["cls"], outputs.get("glca_repr", outputs["cls"])], dim=-1)
        for i, path in enumerate(paths):
            feature_cache[path] = fused_feat[i].cpu()
    
    # Compute similarities for all pairs
    similarities = []
    labels = []
    
    for i in range(len(verif_dataset)):
        pair_data = verif_dataset[i]
        path1 = pair_data["path1"]
        path2 = pair_data["path2"]
        label = pair_data["label"]
        
        feat1 = feature_cache[path1].unsqueeze(0).to(device)
        feat2 = feature_cache[path2].unsqueeze(0).to(device)
        
        if similarity_metric == "cosine":
            sim = cosine_similarity(feat1, feat2).item()
        else:  # euclidean
            sim = euclidean_similarity(feat1, feat2).item()
        
        similarities.append(sim)
        labels.append(label)
    
    # Convert to tensors
    similarities_tensor = torch.tensor(similarities, device=device)
    labels_tensor = torch.tensor(labels, device=device, dtype=torch.long)
    
    # Compute metrics
    metrics = verification_metrics(similarities_tensor, labels_tensor)
    
    return metrics


def main():
    args = parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / args.log_file
    logger = setup_logger(log_file)
    logger.info("Initialized training script for DCAL on NDTWIN")

    train_dataset = NDTWINDataset(args.data_root, split="train", transform=build_ndtwin_transforms(True))
    train_loader = build_loader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    cfg = DCALConfig(img_size=448, num_classes=347, local_ratio=args.local_ratio, drop_path_rate=args.drop_path)
    model = DCALViT(cfg)
    model.load_pretrained(args.weights)
    model.to(device)

    uncertainty = UncertaintyWeighting(["sa", "glca", "pwca"]).to(device)
    params = list(model.parameters()) + list(uncertainty.parameters())
    scaled_lr = (5e-4 / 512) * args.batch_size
    optimizer = torch.optim.AdamW(params, lr=scaled_lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    steps_per_epoch = len(train_loader)
    for epoch in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch:03d} started")
        stats = train_one_epoch(
            model,
            uncertainty,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            args.log_interval,
            logger,
            global_step_start=(epoch - 1) * steps_per_epoch,
        )
        val_metrics = None
        if args.val_interval > 0 and epoch % args.val_interval == 0:
            val_metrics = evaluate(
                model, args.data_root, device, args.similarity_metric, args.eval_batch_size, args.num_workers
            )
            logger.info(
                f"Epoch {epoch:03d}: loss={stats['loss']:.4f} train_acc={stats['acc']:.4f} "
                f"val_best_acc={val_metrics['best_acc']:.4f} val_eer={val_metrics['eer']:.4f} "
                f"val_roc_auc={val_metrics['roc_auc']:.4f} val_tar={val_metrics['tar']:.4f} "
                f"val_far={val_metrics['far']:.4f}"
            )
        else:
            logger.info(
                f"Epoch {epoch:03d}: loss={stats['loss']:.4f} train_acc={stats['acc']:.4f} (validation skipped)"
            )
        scheduler.step()
        if val_metrics is not None and val_metrics["best_acc"] > best_acc:
            best_acc = val_metrics["best_acc"]
            ckpt = {
                "model": model.state_dict(),
                "uncertainty": uncertainty.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_best_acc": val_metrics["best_acc"],
                "val_eer": val_metrics["eer"],
                "val_roc_auc": val_metrics["roc_auc"],
                "val_tar": val_metrics["tar"],
                "val_far": val_metrics["far"],
            }
            torch.save(ckpt, output_dir / "best.pt")
            logger.info(f"New best accuracy {best_acc:.4f} at epoch {epoch:03d}; checkpoint saved to best.pt")
    logger.info("Training completed")


if __name__ == "__main__":
    main()


