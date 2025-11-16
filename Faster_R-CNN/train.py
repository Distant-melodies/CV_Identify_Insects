# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# File       : train
# Time       ：2025/11/12 21:16
# Author     ：Jingyang Dai
# zId        ：z5553615
# Description：
"""
import time
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src import config
from src.model import get_model
from src.DataProcessing.DataLoaders import data_loader_train, data_loader_valid

amp_device = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None, use_amp=False):
    """Single training epoch."""
    model.train()
    epoch_start = time.time()

    running_loss = 0.0
    num_batches = len(data_loader)

    pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1} [Train]", ncols=100)
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with torch.amp.autocast(amp_device):
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()

        loss_value = loss.item()
        running_loss += loss_value

        pbar.set_postfix(loss=f"{loss_value:.4f}")

    avg_loss = running_loss / num_batches
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch + 1} Train Loss: {avg_loss:.4f} (Time: {epoch_time:.2f}s)")

    return avg_loss


def validate_one_epoch(model, data_loader, device, epoch):
    """
    Validation loss for one epoch.

    NOTE: For torchvision detection models, loss is only returned in train() mode.
    So we keep model.train() but wrap with torch.no_grad() to avoid gradient updates.
    """
    model.train()  # important: keep train mode to get loss_dict
    val_start = time.time()

    running_loss = 0.0
    num_batches = len(data_loader)

    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1} [Valid]", ncols=100)
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss_value = loss.item()
            running_loss += loss_value

            pbar.set_postfix(loss=f"{loss_value:.4f}")

    avg_val_loss = running_loss / num_batches
    val_time = time.time() - val_start
    print(f"Epoch {epoch + 1} Valid Loss: {avg_val_loss:.4f} (Time: {val_time:.2f}s)")

    return avg_val_loss


def main():
    # --- Setup ---
    device = config.DEVICE
    print(f"Starting training on device: {device}")

    if str(device) == "cpu":
        print("=" * 50)
        print("WARNING: Training on CPU. This will be extremely slow.")
        print("=" * 50)
        time.sleep(3)

    # Optional: reproducibility
    if hasattr(config, "SEED"):
        torch.manual_seed(config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.SEED)

    model = get_model().to(device)

    # --- Optimizer & Scheduler ---
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # --- AMP (mixed precision) setup ---
    use_amp = (str(device) != "cpu") and torch.cuda.is_available()

    if use_amp:
        scaler = torch.amp.GradScaler(amp_device)
        print("Using mixed precision (AMP) training.")
    else:
        scaler = None
        print("Not using mixed precision (CPU or no CUDA).")

    num_epochs = getattr(config, "NUM_EPOCHS", 1)
    best_val_loss = float("inf")

    print("--- Starting Training Loop ---")
    total_start = time.time()

    train_losses = []
    val_losses = []
    lrs = []

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)
        print(f"\nEpoch {epoch + 1}/{num_epochs} - LR: {current_lr:.6f}")

        # 1) Train
        train_loss = train_one_epoch(
            model,
            optimizer,
            data_loader_train,
            device,
            epoch,
            scaler=scaler,
            use_amp=use_amp,
        )
        train_losses.append(train_loss)

        # 2) Validate
        val_loss = validate_one_epoch(
            model,
            data_loader_valid,
            device,
            epoch,
        )
        val_losses.append(val_loss)

        # 3) Step LR scheduler
        lr_scheduler.step()

        # 4) Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "fasterrcnn_best.pth")
            print(f"*** New best model saved (Val Loss: {best_val_loss:.4f}) ***")

    total_time = time.time() - total_start
    print("\n--- Training Finished ---")
    print(f"Total Training Time: {total_time:.2f}s ({total_time / 60:.2f} min)")

    # Save final model
    torch.save(model.state_dict(), "fasterrcnn_final.pth")
    print("Final model saved to fasterrcnn_final.pth")

    # Result recorded: loss & lr
    try:
        root_dir = config.ROOT_DIR
    except AttributeError:
        root_dir = Path(".").resolve()

    results_dir = Path(root_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save as.npy for easier plotting with numpy/matplotlib
    np.save(results_dir / "train_losses.npy", np.array(train_losses, dtype=np.float32))
    np.save(results_dir / "val_losses.npy", np.array(val_losses, dtype=np.float32))
    np.save(results_dir / "lrs.npy", np.array(lrs, dtype=np.float32))

    # 2Save as json，
    log = {
        "num_epochs": int(num_epochs),
        "best_val_loss": float(best_val_loss),
        "total_time_sec": float(total_time),
        "train_losses": [float(x) for x in train_losses],
        "val_losses": [float(x) for x in val_losses],
        "lrs": [float(x) for x in lrs],
    }
    with open(results_dir / "training_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    print(f"Training logs saved to: {results_dir}")


if __name__ == "__main__":
    main()
