# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# File       : attention_maps_cam
# Time       ：2025/11/17 17:15
# Author     ：Jingyang Dai
# zId        ：z5553615
# Description：
"""
import torch
import numpy as np
from pathlib import Path
import cv2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget

from src import config
from src.model import get_model
from src.DataProcessing.DataLoaders import data_loader_test


def load_trained_model():
    """
    Load the trained Faster R-CNN model from checkpoint and put it on the correct device.
    """
    model = get_model()
    ckpt = Path("fasterrcnn_best.pth")

    # Use weights_only=True when possible to avoid pickle security issues.
    try:
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        # Fallback for older PyTorch versions that do not support weights_only
        state_dict = torch.load(ckpt, map_location="cpu")

    model.load_state_dict(state_dict)
    device = config.DEVICE
    model.to(device)
    model.eval()
    return model


def tensor_to_rgb(img_tensor):
    """
    Convert a tensor image (C,H,W) in [0,1] to a numpy RGB image (H,W,3) in [0,1].
    """
    img = img_tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    return img


def main():
    device = config.DEVICE
    # This flag is just for information; GradCAM no longer needs use_cuda as an argument.
    use_cuda = (str(device) != "cpu") and torch.cuda.is_available()
    print(f"Using CUDA for CAM: {use_cuda}")

    model = load_trained_model()

    # Target layer for Grad-CAM: last conv block of ResNet50 backbone
    target_layers = [model.backbone.body.layer4[-1]]

    # Newer versions of pytorch-grad-cam do not accept 'use_cuda' argument.
    cam = GradCAM(model=model, target_layers=target_layers)

    results_dir = Path(getattr(config, "ROOT_DIR", ".")) / "cam_visualizations"
    results_dir.mkdir(parents=True, exist_ok=True)

    num_images_to_explain = 5
    count = 0

    # Iterate over test dataloader and generate CAMs for a few images
    for images, targets in data_loader_test:
        for idx_in_batch, img in enumerate(images):
            if count >= num_images_to_explain:
                break

            rgb_image = tensor_to_rgb(img)
            input_tensor = img.unsqueeze(0).to(device)

            # Forward pass to get detections (boxes, labels, scores)
            with torch.no_grad():
                outputs = model([input_tensor[0]])[0]

            boxes = outputs["boxes"]
            labels = outputs["labels"]
            scores = outputs["scores"]

            # Keep only high-confidence detections
            keep = scores >= 0.7
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]

            if boxes.numel() == 0:
                print(f"[Image {count}] no high-confidence detections, skip.")
                count += 1
                continue

            # Only visualize top-K detections for this image
            k = min(3, boxes.shape[0])
            boxes_cam = boxes[:k].detach().cpu().numpy()
            labels_cam = labels[:k].detach().cpu().numpy().tolist()

            targets_cam = [FasterRCNNBoxScoreTarget(
                labels=labels_cam,
                bounding_boxes=boxes_cam
            )]

            # Grad-CAM: generate a single grayscale heatmap for this image
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=targets_cam
            )[0]

            # Overlay CAM on the original RGB image
            cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

            # Save visualization to disk
            cam_uint8 = (cam_image * 255).astype(np.uint8)
            bgr = cv2.cvtColor(cam_uint8, cv2.COLOR_RGB2BGR)

            out_path = results_dir / f"cam_img_{count:03d}.png"
            cv2.imwrite(str(out_path), bgr)
            print(f"Saved CAM visualization -> {out_path}")

            count += 1

        if count >= num_images_to_explain:
            break


if __name__ == "__main__":
    main()
