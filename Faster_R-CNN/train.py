# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# File       : train
# Time       ：2025/11/12 21:16
# Author     ：Jingyang Dai
# zId        ：z5553615
# Description：
"""
import torch
import os
import sys
import time
from tqdm import tqdm  # Import tqdm for validation progress bar

from src import config
from src.model import get_model
from src.DataProcessing.DataLoaders import data_loader_train, data_loader_valid

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

    # --- NEW: Define epochs and best_loss inside main ---
    num_epochs = 10  # Set this to your desired number (e.g., 20)
    best_val_loss = float('inf') # Track the best validation loss

    print("--- Starting Training Loop ---")
    total_start_time = time.time()

    for epoch in range(num_epochs):
        # --- 3. Training Loop ---
        model.train()
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        epoch_start_time = time.time()
        train_loss_sum = 0
        total_train_batches = len(data_loader_train)

        for batch_idx, (images, targets) in enumerate(data_loader_train, 1):
            progress = (batch_idx / total_train_batches) * 100
            print(f"  Training: [Batch {batch_idx}/{total_train_batches}] {progress:.2f}% - Loading data to GPU...     ",
                  end='\r')

            images = list(image.to(config.DEVICE) for image in images)
            targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]

            print(f"  Training: [Batch {batch_idx}/{total_train_batches}] {progress:.2f}% - Running Forward Pass...      ",
                  end='\r')
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            train_loss_sum += loss_value

            print(f"  Training: [Batch {batch_idx}/{total_train_batches}] {progress:.2f}% - Running Backward Pass...     ",
                  end='\r')
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            print(f"  Training: [Batch {batch_idx}/{total_train_batches}] {progress:.2f}% - Loss: {loss_value:.4f}          ",
                  end='\r')

        print() # New line after progress bar
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        avg_train_loss = train_loss_sum / total_train_batches
        print(f"Epoch {epoch + 1} Training Loss: {avg_train_loss:.4f} (Took {epoch_duration:.2f}s)")

        lr_scheduler.step()

        # --- 4. Validation Loop (FIXED) ---
        # Now we iterate over the *entire* validation set
        model.train() # Keep in train() mode to get loss, but use no_grad()
        print(f"--- Running Full Validation for Epoch {epoch + 1} ---")

        val_loss_sum = 0
        total_val_batches = len(data_loader_valid)

        with torch.no_grad():
            # Use tqdm for a validation progress bar
            for images_val, targets_val in tqdm(data_loader_valid, desc="Validating"):
                images_val = list(image.to(config.DEVICE) for image in images_val)
                targets_val = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets_val]

                val_loss_dict = model(images_val, targets_val)
                val_losses = sum(loss for loss in val_loss_dict.values())
                val_loss_sum += val_losses.item()

        avg_val_loss = val_loss_sum / total_val_batches
        print(f"Epoch {epoch + 1} Average Validation Loss: {avg_val_loss:.4f}")

        # --- 5. Best Model Saving (FIXED) ---
        # Save the model *only if* this epoch's validation loss is the best one so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "fasterrcnn_best.pth")
            print(f"*** New best model saved to fasterrcnn_best.pth (Val Loss: {best_val_loss:.4f}) ***")


    total_end_time = time.time()
    total_duration_sec = total_end_time - total_start_time
    total_duration_min = total_duration_sec / 60

    print("\n--- Training Finished ---")
    print(f"Total Training Time: {total_duration_sec:.2f} seconds ({total_duration_min:.2f} minutes)")

    # We still save the final model, but the 'best' one is what truly matters
    torch.save(model.state_dict(), "fasterrcnn_final.pth")
    print("Final model saved to fasterrcnn_final.pth")


if __name__ == "__main__":
    main()
