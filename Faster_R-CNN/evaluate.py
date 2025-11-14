# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# File       : evaluate
# Time       ：2025/11/14 23:26
# Author     ：Jingyang Dai
# zId        ：z5553615
# Description：
"""
import torch
import time
from tqdm import tqdm

from src import config
from src.model import get_model
from src.DataProcessing.DataLoaders import data_loader_test
from torchmetrics.detection import MeanAveragePrecision


def evaluate():
    print(f"Starting evaluation on device: {config.DEVICE}")

    model = get_model()
    model.to(config.DEVICE)

    try:
        # Load weights with weights_only=True to silence the warning
        model.load_state_dict(torch.load('fasterrcnn_final.pth', weights_only=True))
    except FileNotFoundError:
        print("Error: 'fasterrcnn_final.pth' not found. Please run train.py first.")
        return

    model.eval()

    # Add 'return_pr_curves=True' to get precision/recall data
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True).to(config.DEVICE)

    start_time = time.time()

    with torch.no_grad():
        for images, targets in tqdm(data_loader_test, desc="Evaluating"):
            images = list(image.to(config.DEVICE) for image in images)
            predictions = model(images)
            targets_formatted = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]

            metric.update(predictions, targets_formatted)

    end_time = time.time()
    total_time = end_time - start_time

    results = metric.compute()

    # --- Streamlined Output ---

    print(f"\nTotal Test Time: {total_time:.2f}s")

    print("\n--- DETECTION Metrics (mAP) ---")
    print(f"mAP (IoU 0.50:0.95):         {results['map']:.4f}")
    print(f"mAP (at 50% IoU / mAP@.50):  {results['map_50']:.4f}")
    print(f"mAP (at 75% IoU / mAP@.75):  {results['map_75']:.4f}")

    print("\n--- CLASSIFICATION Metrics ---")

    try:
        # --- NEW FIX ---
        # We use map_50 as the proxy for Precision and mar_100 as the proxy for Recall

        # 'map_50' is the Mean Average Precision at IoU 0.50
        precision_proxy = results['map_50'].item()
        print(f"Precision (using mAP@.50):    {precision_proxy:.4f}")

        # 'mar_100' is the Max Average Recall for 100 detections
        recall_proxy = results['mar_100'].item()
        print(f"Recall (using mAR@100):      {recall_proxy:.4f}")

        # F1 = 2 * (Precision * Recall) / (Precision + Recall)
        if (precision_proxy + recall_proxy) > 0:
            f1_score = 2 * (precision_proxy * recall_proxy) / (precision_proxy + recall_proxy)
            print(f"F1 Score (approximated):     {f1_score:.4f}")
        else:
            print("F1 Score: N/A (Precision or Recall is zero)")
        # --- END FIX ---

        print(f"Area Under Curve (mAP):      {results['map']:.4f}")

    except Exception as e:
        print(f"  (Could not compute F1/Precision/Recall: {e})")

    print(f"\n--- Per-Class mAP (Accuracy) ---")

    class_ids = results['classes']
    map_per_class = results['map_per_class']

    if class_ids.dim() == 0:
        class_ids = class_ids.unsqueeze(0)
    if map_per_class.dim() == 0:
        map_per_class = map_per_class.unsqueeze(0)

    for class_id, map_val in zip(class_ids, map_per_class):
        print(f"  Class {class_id.item()}: mAP = {map_val.item():.4f}")


if __name__ == "__main__":
    evaluate()
