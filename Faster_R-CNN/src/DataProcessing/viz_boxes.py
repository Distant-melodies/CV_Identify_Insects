# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# File       : viz_boxes
# Time       ：2025/11/20 2:48
# Author     ：Jingyang Dai
# zId        ：z5553615
# Description：
"""
import matplotlib
matplotlib.use("Agg")  # use non-interactive backend: write to files only

import matplotlib.pyplot as plt
from pathlib import Path

from src import config
from src.DataProcessing.Dataset import PestDataset, get_transform


def save_first_n_samples(n=6, train=True):
    """Save side-by-side comparison of original image and image with boxes
    for the first n samples of the dataset.
    """
    # 1) choose split and build dataset
    root = config.TRAIN_DIR if train else config.TEST_DIR
    dataset = PestDataset(
        root_dir=str(root),
        transform=get_transform(train=train)
    )

    # 2) output directory
    out_dir = Path(getattr(config, "ROOT_DIR", ".")) / "debug_boxes_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    split_name = "train" if train else "test"
    max_idx = min(n, len(dataset))

    for idx in range(max_idx):
        img, target = dataset[idx]
        boxes = target["boxes"]      # Tensor [N, 4], [xmin, ymin, xmax, ymax]
        labels = target["labels"]    # Tensor [N]

        # Tensor(C,H,W) -> numpy(H,W,C)
        img_np = img.detach().cpu().permute(1, 2, 0).numpy()
        # just in case: clip to [0,1]
        img_np = img_np.clip(0.0, 1.0)

        # 3) create figure with two subplots: left original, right with boxes
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # left: original image
        axes[0].imshow(img_np)
        axes[0].set_title(f"Sample {idx} ({split_name}) - original")
        axes[0].axis("off")

        # right: image with boxes
        axes[1].imshow(img_np)
        axes[1].set_title(f"Sample {idx} ({split_name}) - with boxes")
        axes[1].axis("off")

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box.tolist()
            w, h = x2 - x1, y2 - y1
            rect = plt.Rectangle(
                (x1, y1),
                w,
                h,
                fill=False,
                linewidth=2,
                edgecolor="red",
            )
            axes[1].add_patch(rect)
            axes[1].text(
                x1,
                y1 - 2,
                str(int(label.item())),
                fontsize=8,
                color="yellow",
                bbox=dict(facecolor="black", alpha=0.5, pad=1),
            )

        plt.tight_layout()

        # 4) save figure to file
        out_path = out_dir / f"sample_{idx}_{split_name}_compare.png"
        plt.savefig(out_path, dpi=200)
        plt.close(fig)

        print(f"Saved comparison to: {out_path}")


if __name__ == "__main__":
    save_first_n_samples(n=8, train=True)