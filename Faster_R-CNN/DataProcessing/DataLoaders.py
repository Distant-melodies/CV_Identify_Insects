# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# File       : DataLoaders
# Time       ：2025/11/7 13:43
# Author     ：Jingyang Dai
# zId        ：z5553615
# Description：
"""

import torch
from torch.utils.data import DataLoader

# Use absolute imports from the package root 'Faster_R_CNN'
from Dataset import PestDataset, get_transform
from .. import config


def collate_fn(batch):
    """
    Custom collate_fn for object detection.
    Batches images and targets separately.
    """
    return tuple(zip(*batch))


# --- Create Datasets using Config Paths ---
print(f"Loading training data from: {config.TRAIN_DIR}")
dataset_train = PestDataset(
    root_dir=str(config.TRAIN_DIR),
    transform=get_transform(train=True)
)

print(f"Loading validation data from: {config.VALID_DIR}")
dataset_valid = PestDataset(
    root_dir=str(config.VALID_DIR),
    transform=get_transform(train=False)
)

# --- Create DataLoaders ---
data_loader_train = DataLoader(
    dataset_train,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    collate_fn=collate_fn
)

data_loader_valid = DataLoader(
    dataset_valid,
    batch_size=1,  # Validation batch size is often 1
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    collate_fn=collate_fn
)

