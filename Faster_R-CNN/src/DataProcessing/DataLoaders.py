# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# File       : DataLoaders
# Time       ：2025/11/7 13:43
# Author     ：Jingyang Dai
# zId        ：z5553615
# Description：
"""
import os
import random
from collections import defaultdict

from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

# Use absolute imports from the package root 'Faster_R_CNN'
from .Dataset import PestDataset, get_transform
from src import config


def collate_fn(batch):
    """
    Custom collate_fn for object detection.
    Batches images and targets separately.
    """
    return tuple(zip(*batch))


def build_imbalanced_indices(
        train_labels_dir: str,
        image_files,
        num_classes: int,
        minority_classes,
        keep_ratio: float,
        seed: int = 42,
):
    """
    Build a list of indices for an imbalanced training subset.
    :param train_labels_dir:
    :param image_files:
    :param num_classes:
    :param minority_classes:
    :param keep_ratio:
    :param seed:
    :return:
    """
    rng = random.Random(seed)

    # Map primary class -> list of image basenames (without extension)
    class_to_basenames = defaultdict(list)

    for img_name in image_files:
        base = os.path.splitext(img_name)[0]
        label_path = os.path.join(train_labels_dir, base + ".txt")

        if not os.path.exists(label_path):
            # No label file -> skip this image
            continue

        cls_ids = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 1:
                    # Empty line
                    continue
                # YOLO format: class_id x_c y_c w h (class_id starts from 0)
                try:
                    cid = int(parts[0])
                except ValueError:
                    # Invalid class id in this line, skip the line
                    continue
                # Shift to [1..num_classes-1] if your dataset uses that convention
                cls_ids.append(cid + 1)

        # IMPORTANT: If no valid objects were found in this file, skip the image
        if not cls_ids:
            # This avoids ValueError: max() arg is an empty sequence
            continue

        # Primary class = the most frequent class in this image
        primary = max(set(cls_ids), key=cls_ids.count)
        class_to_basenames[primary].append(base)

    # Now sub-sample minority classes
    stats = {}
    selected_basenames = []

    for cid in range(1, num_classes):  # assuming 0 is background / unused
        basenames = class_to_basenames.get(cid, [])
        total = len(basenames)
        if total == 0:
            stats[cid] = {"total": 0, "kept": 0}
            continue

        if cid in minority_classes:
            # Only keep a fraction of images for minority classes
            k = max(1, int(total * keep_ratio))
            rng.shuffle(basenames)
            chosen = basenames[:k]
        else:
            # Keep all images for non-minority classes
            chosen = basenames

        stats[cid] = {"total": total, "kept": len(chosen)}
        selected_basenames.extend(chosen)

    # Convert selected basenames back to dataset indices
    basename_to_index = {
        os.path.splitext(name)[0]: idx for idx, name in enumerate(image_files)
    }

    selected_indices = []
    for base in selected_basenames:
        idx = basename_to_index.get(base, None)
        if idx is not None:
            selected_indices.append(idx)

    selected_indices = sorted(set(selected_indices))
    return selected_indices, stats


def build_sample_weights(
        train_labels_dir: str,
        image_files,
        selected_indices,
        num_classes: int,
        seed: int = 42,
):
    """
    Build per-image sampling weights for the (possibly imbalanced) training set.

    For each image index in selected_indices, we determine its "primary" class
    (most frequent class in this image) and assign a sampling weight
    proportional to 1 / class_frequency(primary_class).
    """
    rng = random.Random(seed)

    # Map index -> primary class
    index_to_primary = {}
    class_counts = [0 for _ in range(num_classes)]

    for idx in selected_indices:
        img_name = image_files[idx]
        base = os.path.splitext(img_name)[0]
        label_path = os.path.join(train_labels_dir, base + ".txt")

        if not os.path.exists(label_path):
            continue

        cls_ids = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 1:
                    continue
                try:
                    cid = int(parts[0])
                except ValueError:
                    continue
                cls_ids.append(cid + 1)

        if not cls_ids:
            continue

        primary = max(set(cls_ids), key=cls_ids.count)
        index_to_primary[idx] = primary
        if 0 <= primary < num_classes:
            class_counts[primary] += 1

    # Build weights list aligned with 'selected_indices' order
    weights = []
    for idx in selected_indices:
        primary = index_to_primary.get(idx, None)
        if primary is None or class_counts[primary] == 0:
            # Fallback small weight if something is wrong
            w = 0.0
        else:
            w = 1.0 / float(class_counts[primary])
        weights.append(w)

    return weights, class_counts


# Create Datasets using Config Paths
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

print(f"Loading test data from: {config.TEST_DIR}")
dataset_test = PestDataset(
    root_dir=str(config.TEST_DIR),
    transform=get_transform(train=False)
)

# Sampler for training data (None means use plain shuffle)
train_sampler = None


# Optionally build an imbalanced training subset
if getattr(config, "USE_IMBALANCED_TRAIN", False):
    print("\n[Imbalance] Building imbalanced training subset...")

    train_labels_dir = os.path.join(str(config.TRAIN_DIR), "labels")

    # PestDataset is expected to expose the list of image file names.
    # Try common attribute names; raise a clear error if not found.
    try:
        image_files = dataset_train.image_files
    except AttributeError:
        try:
            image_files = dataset_train.img_files
        except AttributeError:
            raise AttributeError(
                "PestDataset must expose an attribute like 'image_files' or "
                "'img_files' (list of file names) for the imbalance experiment."
            )

    num_classes = getattr(config, "NUM_CLASSES", 13)
    minority_classes = getattr(config, "MINORITY_CLASSES", [1, 3, 5])
    keep_ratio = getattr(config, "MINORITY_KEEP_RATIO", 0.2)
    imbalance_seed = getattr(config, "IMBALANCE_SEED", 42)

    selected_indices, imbalance_stats = build_imbalanced_indices(
        train_labels_dir=train_labels_dir,
        image_files=image_files,
        num_classes=num_classes,
        minority_classes=minority_classes,
        keep_ratio=keep_ratio,
        seed=imbalance_seed,
    )

    print("[Imbalance] Per-class image counts (total -> kept):")
    for cid in sorted(imbalance_stats.keys()):
        total = imbalance_stats[cid]["total"]
        kept = imbalance_stats[cid]["kept"]
        if total == 0:
            continue
        print(f"  Class {cid:2d}: {total:4d} -> {kept:4d}")

    print(f"[Imbalance] Total training images before: {len(dataset_train)}")
    print(f"[Imbalance] Total training images after:  {len(selected_indices)}")

    # Replace the original training dataset by a Subset with the selected indices
    dataset_train = Subset(dataset_train, selected_indices)
    print("[Imbalance] Using imbalanced Subset for training.\n")

    # Optionally build a class-balanced sampler on top of the imbalanced set
    if getattr(config, "USE_BALANCED_SAMPLER", False):
        print("[Sampler] Building class-balanced WeightedRandomSampler...")

        # Build sampling weights based on primary class frequencies
        weights, class_counts = build_sample_weights(
            train_labels_dir=train_labels_dir,
            image_files=image_files,
            selected_indices=selected_indices,
            num_classes=num_classes,
            seed=imbalance_seed,
        )

        for cid in range(1, num_classes):
            if class_counts[cid] > 0:
                print(f"  [Sampler] Class {cid:2d}: {class_counts[cid]:4d} images")

        train_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        print("[Sampler] Using WeightedRandomSampler for training.\n")

# Create DataLoaders
if train_sampler is not None:
    # When using a sampler, 'shuffle' must be False
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn
    )
else:
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn
    )

data_loader_valid = DataLoader(
    dataset_valid,
    batch_size=1,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    collate_fn=collate_fn
)

data_loader_test = DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    collate_fn=collate_fn
)
