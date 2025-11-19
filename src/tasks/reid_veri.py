"""
Training script for DCAL on VeRi-776 (vehicle ReID).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from datasets import VeRiDataset
from losses.uncertainty import UncertaintyWeighting
from models.vit_dcal import DCALConfig, DCALViT
from utils.data import RandomIdentitySampler, build_loader, build_veri_transforms
from utils.metrics import pairwise_distance, reid_metrics
from utils.wandb_logging import maybe_init_wandb, wandb_finish, wandb_log


def parse_args():
    parser = argparse.ArgumentParser(description="Train DCAL on VeRi-776")
    parser.add_argument("--data-root", default="data/VeRi_776", type=str)
    parser.add_argument("--weights", default="weights/ViT-B_16.npz", type=str)
    parser.add_argument("--output", default="runs/reid_veri", type=str)
    parser.add_argument("--epochs", default=120, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--eval-batch-size", default=128, type=int)
    parser.add_argument("--instances-per-id", default=4, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--local-ratio", default=0.3, type=float)
    parser.add_argument("--lr", default=0.008, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--triplet-margin", default=0.3, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--log-interval", default=50, type=int, help="log training metrics every N batches")
    parser.add_argument("--val-interval", default=1, type=int, help="run evaluation every N epochs")
    parser.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="dcal", type=str)
    parser.add_argument("--wandb-run-name", default=None, type=str)
    return parser.parse_args()


def batch_hard_triplet(embeddings: torch.Tensor, labels: torch.Tensor, margin: float):
    dist = pairwise_distance(embeddings, embeddings)
    mask_pos = labels.unsqueeze(1).eq(labels.unsqueeze(0))
    mask_neg = ~mask_pos
    mask_pos.fill_diagonal_(False)
    mask_neg.fill_diagonal_(False)
    dist_pos = (dist * mask_pos.float()).max(dim=1)[0]
    max_dist = dist.max().detach()
    dist_neg = dist.clone()
    dist_neg[mask_neg == 0] = max_dist
    dist_neg = dist_neg.min(dim=1)[0]
    loss = torch.relu(dist_pos - dist_neg + margin)
    return loss.mean()


def train_one_epoch(
    model,
    uncertainty,
    loader,
    optimizer,
    ce_loss,
    margin,
    device,
    epoch: int,
    log_interval: int,
    wandb_run,
    global_step_start: int,
):
    model.train()
    uncertainty.train()
    total_loss = 0.0
    steps = 0
    for step_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device)
        labels = batch["pid"].to(device)
        perm = torch.randperm(images.size(0), device=device)
        outputs = model(images, enable_glca=True, enable_pwca=True, pair_indices=perm)
        loss_sa = ce_loss(outputs["sa_logits"], labels)
        loss_glca = ce_loss(outputs["glca_logits"], labels)
        loss_pwca = ce_loss(outputs["pwca_logits"], labels)
        fused_feat = torch.cat([outputs["cls"], outputs.get("glca_repr", outputs["cls"])], dim=-1)
        triplet = batch_hard_triplet(fused_feat, labels, margin)
        loss = uncertainty({"sa": loss_sa, "glca": loss_glca, "pwca": loss_pwca}) + triplet

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

        if log_interval > 0 and (step_idx % log_interval == 0 or step_idx == 1):
            global_step = global_step_start + step_idx
            wandb_log(
                wandb_run,
                {
                    "epoch": epoch,
                    "step": global_step,
                    "train/batch_loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                },
            )
            print(f"[Epoch {epoch:03d} | Step {step_idx:04d}] loss={loss.item():.4f}")
    return total_loss / steps


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feats = []
    pids = []
    camids = []
    for batch in loader:
        images = batch["image"].to(device)
        outputs = model(images, enable_glca=True, enable_pwca=False)
        fused = torch.cat([outputs["cls"], outputs.get("glca_repr", outputs["cls"])], dim=-1)
        feats.append(fused.cpu())
        pids.extend(batch["pid"].tolist())
        camids.extend(batch["camid"].tolist())
    return torch.cat(feats, dim=0), pids, camids


def main():
    args = parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    Path(args.output).mkdir(parents=True, exist_ok=True)

    train_dataset = VeRiDataset(args.data_root, split="train", transform=build_veri_transforms(True))
    query_dataset = VeRiDataset(args.data_root, split="query", transform=build_veri_transforms(False))
    gallery_dataset = VeRiDataset(args.data_root, split="gallery", transform=build_veri_transforms(False))

    sampler = RandomIdentitySampler(train_dataset.labels, args.batch_size, args.instances_per_id)
    train_loader = build_loader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=sampler)
    query_loader = build_loader(
        query_dataset, batch_size=args.eval_batch_size, num_workers=args.num_workers, shuffle=False
    )
    gallery_loader = build_loader(
        gallery_dataset, batch_size=args.eval_batch_size, num_workers=args.num_workers, shuffle=False
    )

    num_classes = len(set(train_dataset.labels))
    cfg = DCALConfig(img_size=256, num_classes=num_classes, local_ratio=args.local_ratio, drop_path_rate=0.1)
    model = DCALViT(cfg)
    model.load_pretrained(args.weights)
    model.to(device)

    uncertainty = UncertaintyWeighting(["sa", "glca", "pwca"]).to(device)
    params = list(model.parameters()) + list(uncertainty.parameters())
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce_loss = nn.CrossEntropyLoss()

    wandb_run = maybe_init_wandb(
        args.wandb,
        args.wandb_project,
        args.wandb_run_name,
        {
            "task": "reid_veri",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "instances_per_id": args.instances_per_id,
            "lr": args.lr,
            "local_ratio": args.local_ratio,
            "triplet_margin": args.triplet_margin,
        },
    )

    best_map = 0.0
    steps_per_epoch = len(train_loader)
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch:03d}:")
        loss = train_one_epoch(
            model,
            uncertainty,
            train_loader,
            optimizer,
            ce_loss,
            args.triplet_margin,
            device,
            epoch,
            args.log_interval,
            wandb_run,
            global_step_start=(epoch - 1) * steps_per_epoch,
        )
        scheduler.step()
        metrics = None
        if args.val_interval > 0 and epoch % args.val_interval == 0:
            q_feats, q_pids, q_cam = extract_features(model, query_loader, device)
            g_feats, g_pids, g_cam = extract_features(model, gallery_loader, device)
            metrics = reid_metrics(q_feats, q_pids, q_cam, g_feats, g_pids, g_cam)
            print(f"Epoch {epoch:03d}: loss={loss:.4f} mAP={metrics['mAP']:.4f} R1={metrics['Rank-1']:.4f}")
        else:
            print(f"Epoch {epoch:03d}: loss={loss:.4f} (validation skipped)")
        log_payload = {
            "epoch": epoch,
            "train/loss": loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if metrics is not None:
            log_payload["val/mAP"] = metrics["mAP"]
            log_payload["val/Rank-1"] = metrics["Rank-1"]
        wandb_log(wandb_run, log_payload)
        if metrics is not None and metrics["mAP"] > best_map:
            best_map = metrics["mAP"]
            ckpt = {
                "model": model.state_dict(),
                "uncertainty": uncertainty.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
            }
            torch.save(ckpt, Path(args.output) / "best.pt")
    wandb_finish(wandb_run)


if __name__ == "__main__":
    main()


