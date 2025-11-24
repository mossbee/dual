"""
Training script for DCAL on NDTWIN (NDTWIN).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from datasets import NDTWINDataset
from losses.uncertainty import UncertaintyWeighting
from models.vit_dcal import DCALConfig, DCALViT
from utils.data import build_ndtwin_transforms, build_loader
from utils.metrics import fused_accuracy
from utils.wandb_logging import maybe_init_wandb, wandb_finish, wandb_log


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
    parser.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="dcal", type=str)
    parser.add_argument("--wandb-run-name", default=None, type=str)
    return parser.parse_args()


def train_one_epoch(
    model,
    uncertainty,
    loader,
    optimizer,
    criterion,
    device,
    epoch: int,
    log_interval: int,
    wandb_run,
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
            wandb_log(
                wandb_run,
                {
                    "epoch": epoch,
                    "step": global_step,
                    "train/batch_loss": loss.item(),
                    "train/batch_acc": batch_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                },
            )
            print(
                f"[Epoch {epoch:03d} | Step {step_idx:04d}] "
                f"loss={loss.item():.4f} batch_acc={batch_acc:.4f}"
            )
    return {"loss": total_loss / steps, "acc": total_acc / steps}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_acc = 0.0
    steps = 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        outputs = model(images, enable_glca=True, enable_pwca=False)
        total_acc += fused_accuracy(outputs["sa_logits"], outputs["glca_logits"], labels)
        steps += 1
    return total_acc / steps


def main():
    args = parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    Path(args.output).mkdir(parents=True, exist_ok=True)

    train_dataset = NDTWINDataset(args.data_root, split="train", transform=build_ndtwin_transforms(True))
    val_dataset = NDTWINDataset(args.data_root, split="val", transform=build_ndtwin_transforms(False))

    train_loader = build_loader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = build_loader(
        val_dataset, batch_size=args.eval_batch_size, num_workers=args.num_workers, shuffle=False
    )

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

    wandb_run = maybe_init_wandb(
        args.wandb,
        args.wandb_project,
        args.wandb_run_name,
        {
            "task": "fgvc_ndtwin",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": scaled_lr,
            "local_ratio": args.local_ratio,
            "drop_path": args.drop_path,
        },
    )

    best_acc = 0.0
    steps_per_epoch = len(train_loader)
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch:03d}:")
        stats = train_one_epoch(
            model,
            uncertainty,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            args.log_interval,
            wandb_run,
            global_step_start=(epoch - 1) * steps_per_epoch,
        )
        val_acc = None
        if args.val_interval > 0 and epoch % args.val_interval == 0:
            val_acc = evaluate(model, val_loader, device)
            print(
                f"Epoch {epoch:03d}: loss={stats['loss']:.4f} train_acc={stats['acc']:.4f} "
                f"val_acc={val_acc:.4f}"
            )
        else:
            print(f"Epoch {epoch:03d}: loss={stats['loss']:.4f} train_acc={stats['acc']:.4f} (validation skipped)")
        scheduler.step()
        log_payload = {
            "epoch": epoch,
            "train/loss": stats["loss"],
            "train/acc": stats["acc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        if val_acc is not None:
            log_payload["val/acc"] = val_acc
        wandb_log(wandb_run, log_payload)
        if val_acc is not None and val_acc > best_acc:
            best_acc = val_acc
            ckpt = {
                "model": model.state_dict(),
                "uncertainty": uncertainty.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
            }
            torch.save(ckpt, Path(args.output) / "best.pt")
    wandb_finish(wandb_run)


if __name__ == "__main__":
    main()


