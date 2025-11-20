# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# File       : viz_distortions
# Time       ：2025/11/20 3:42
# Author     ：Jingyang Dai
# zId        ：z5553615
# Description：
"""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: saves to files

import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torchvision.transforms.functional as F

from src import config
from src.DataProcessing.Dataset import PestDataset, get_transform


def apply_gaussian_noise(img_tensor, sigma=0.1):
    """
    Add zero-mean Gaussian noise with standard deviation sigma to an image
    tensor in [0, 1]. Shape: (C, H, W).
    """
    noise = torch.randn_like(img_tensor) * sigma
    noisy = img_tensor + noise
    return torch.clamp(noisy, 0.0, 1.0)


def apply_gaussian_blur(img_tensor, kernel_size=5, sigma=1.0):
    """
    Apply Gaussian blur to an image tensor (C, H, W) in [0, 1].
    """
    blurred = F.gaussian_blur(img_tensor, kernel_size=kernel_size, sigma=sigma)
    return torch.clamp(blurred, 0.0, 1.0)


def apply_dark(img_tensor, factor=0.5):
    """
    Darken the image by a given brightness factor (0 < factor < 1).
    factor = 0.5 means the image becomes about half as bright.
    """
    dark = F.adjust_brightness(img_tensor, factor)
    return torch.clamp(dark, 0.0, 1.0)


def apply_occlusion(img_tensor, area_ratio=0.25):
    """
    Apply a square occlusion mask covering a given area_ratio of the image.
    area_ratio = 0.25 means the square covers 25% of the image area.
    """
    c, h, w = img_tensor.shape
    square_side = int((area_ratio ** 0.5) * min(h, w))
    square_side = max(1, square_side)

    max_x = w - square_side
    max_y = h - square_side
    if max_x < 0 or max_y < 0:
        return img_tensor  # image too small

    x0 = torch.randint(0, max_x + 1, (1,)).item()
    y0 = torch.randint(0, max_y + 1, (1,)).item()

    occluded = img_tensor.clone()
    occluded[:, y0:y0 + square_side, x0:x0 + square_side] = 0.0
    return occluded


def save_distortion_comparisons(n=8, use_test=True):
    """
    For the first n samples, save a 5-column figure:
    Clean vs Gaussian noise vs Gaussian blur vs Dark vs Occlusion.
    """
    # 1) choose split and build dataset (no random augmentations)
    root = config.TEST_DIR if use_test else config.TRAIN_DIR
    dataset = PestDataset(
        root_dir=str(root),
        transform=get_transform(train=False)
    )

    out_dir = Path(getattr(config, "ROOT_DIR", ".")) / "debug_distortions"
    out_dir.mkdir(parents=True, exist_ok=True)
    split_name = "test" if use_test else "train"

    max_idx = min(n, len(dataset))

    for idx in range(max_idx):
        img, _ = dataset[idx]
        img = img.detach().cpu().clamp(0.0, 1.0)

        # 2) generate distorted versions
        img_clean = img
        img_noise = apply_gaussian_noise(img_clean, sigma=0.1)
        img_blur = apply_gaussian_blur(img_clean, kernel_size=5, sigma=1.0)
        img_dark = apply_dark(img_clean, factor=0.5)
        img_occ = apply_occlusion(img_clean, area_ratio=0.25)

        # Tensor(C, H, W) -> numpy(H, W, C)
        def to_np(im_t):
            return im_t.permute(1, 2, 0).numpy().clip(0.0, 1.0)

        clean_np = to_np(img_clean)
        noise_np = to_np(img_noise)
        blur_np = to_np(img_blur)
        dark_np = to_np(img_dark)
        occ_np = to_np(img_occ)

        # 3) plot side-by-side: 5 columns
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))

        axes[0].imshow(clean_np)
        axes[0].set_title("Clean")
        axes[0].axis("off")

        axes[1].imshow(noise_np)
        axes[1].set_title("Gaussian noise")
        axes[1].axis("off")

        axes[2].imshow(blur_np)
        axes[2].set_title("Gaussian blur")
        axes[2].axis("off")

        axes[3].imshow(dark_np)
        axes[3].set_title("Dark (brightness 0.5)")
        axes[3].axis("off")

        axes[4].imshow(occ_np)
        axes[4].set_title("Occlusion (25% area)")
        axes[4].axis("off")

        plt.tight_layout()

        out_path = out_dir / f"sample_{idx}_{split_name}_distortions.png"
        plt.savefig(out_path, dpi=200)
        plt.close(fig)

        print(f"Saved distortion comparison to: {out_path}")


if __name__ == "__main__":
    # generate comparisons for first 8 test images
    save_distortion_comparisons(n=8, use_test=True)
