# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# File       : train
# Time       ：2025/11/14 21:16
# Author     ：Jingyang Dai
# zId        ：z5553615
# Description：
"""
import torch
import os
import sys
import time

from src import config
from src.model import get_model
from src.DataProcessing.DataLoaders import data_loader_train, data_loader_valid

num_epochs = 1


def main():
    # --- 1. Setup ---
    print(f"Starting training on device: {config.DEVICE}")
    if config.DEVICE == 'cpu':
        print("=" * 50)
        print("WARNING: Training on CPU. This will be extremely slow.")
        print("=" * 50)
        time.sleep(3)

    model = get_model()
    model.to(config.DEVICE)

    # --- 2. Optimizer & Scheduler ---
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    print("--- Starting Training Loop ---")
    total_start_time = time.time()

    for epoch in range(num_epochs):
        # --- 3. Training Loop ---
        model.train()
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        epoch_start_time = time.time()
        loss_sum = 0
        total_batches = len(data_loader_train)

        # This print is now in Dataset.py
        # print("DataLoader is preparing batches...")

        for batch_idx, (images, targets) in enumerate(data_loader_train, 1):
            # --- DETAILED LOGGING ---
            progress = (batch_idx / total_batches) * 100
            # \r = carriage return (stay on one line)
            print(f"  Training: [Batch {batch_idx}/{total_batches}] {progress:.2f}% - Loading data to GPU...     ",
                  end='\r')

            images = list(image.to(config.DEVICE) for image in images)
            targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]

            print(f"  Training: [Batch {batch_idx}/{total_batches}] {progress:.2f}% - Running Forward Pass...      ",
                  end='\r')
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            loss_sum += loss_value

            print(f"  Training: [Batch {batch_idx}/{total_batches}] {progress:.2f}% - Running Backward Pass...     ",
                  end='\r')
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # Final status update for this batch
            print(f"  Training: [Batch {batch_idx}/{total_batches}] {progress:.2f}% - Loss: {loss_value:.4f}          ",
                  end='\r')

        # --- End of Epoch Summary ---
        print()  # Move to a new line after the progress bar
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        avg_loss = loss_sum / total_batches
        print(f"Epoch {epoch + 1} Training Loss: {avg_loss:.4f} (Took {epoch_duration:.2f}s)")

        lr_scheduler.step()

        # --- 4. Validation Loop (Corrected) ---

        # We set the model to train() mode to get the loss dictionary.
        # However, we wrap it in torch.no_grad() so no gradients
        # are computed and the model does not learn.
        model.train()

        print(f"--- Running Validation for Epoch {epoch + 1} ---")

        with torch.no_grad():  # <-- Gradients are disabled here
            # Get one batch from the validation loader
            images_val, targets_val = next(iter(data_loader_valid))

            images_val = list(image.to(config.DEVICE) for image in images_val)
            targets_val = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets_val]

            # Forward pass (will return a dict of losses)
            val_loss_dict = model(images_val, targets_val)

            # Now this sum() will work correctly
            val_losses = sum(loss for loss in val_loss_dict.values())
            print(f"  Validation Loss (1 batch): {val_losses.item():.4f}")

    total_end_time = time.time()
    total_duration_sec = total_end_time - total_start_time
    total_duration_min = total_duration_sec / 60

    print("\n--- Training Finished ---")
    print(f"Total Training Time: {total_duration_sec:.2f} seconds ({total_duration_min:.2f} minutes)")

    torch.save(model.state_dict(), "fasterrcnn_final.pth")
    print("Model saved to fasterrcnn_final.pth")


if __name__ == "__main__":
    main()
