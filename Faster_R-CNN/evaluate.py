# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# File       : evaluate
# Time       ：2025/11/14 23:26
# Author     ：Jingyang Dai
# zId        ：z5553615
# Description：
"""
import time
from pathlib import Path

import torch
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou  # NEW: for matching predictions to GT

from src import config
from src.model import get_model
from src.DataProcessing.DataLoaders import data_loader_test


def load_trained_model():
    """
    Load Faster R-CNN with trained weights.
    Tries 'fasterrcnn_best.pth' first, then 'fasterrcnn_final.pth'.
    """
    model = get_model().to(config.DEVICE)

    ckpt_paths = [
        Path("fasterrcnn_best.pth"),
        Path("fasterrcnn_final.pth"),
    ]

    state_dict = None
    for ckpt in ckpt_paths:
        if ckpt.is_file():
            print(f"Loading weights from: {ckpt}")
            # 'weights_only' 可能在部分 torch 版本不支持，这里做兼容处理
            try:
                state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
            except TypeError:
                state_dict = torch.load(ckpt, map_location="cpu")
            break

    if state_dict is None:
        raise FileNotFoundError(
            "No checkpoint found. Please run train.py first to generate "
            "'fasterrcnn_best.pth' or 'fasterrcnn_final.pth'."
        )

    model.load_state_dict(state_dict)
    model.to(config.DEVICE)
    model.eval()
    return model


def match_predictions_to_targets(prediction, target, iou_thr=0.5, score_thr=0.5):
    """
    Match predictions to ground-truth boxes using IoU.

    Args:
        prediction (dict): {'boxes', 'scores', 'labels'}
        target (dict): {'boxes', 'labels'}
        iou_thr (float): IoU threshold for a valid match.
        score_thr (float): score threshold for predictions.

    Returns:
        matched_true (List[int]): matched ground-truth labels.
        matched_pred (List[int]): matched predicted labels.
    """
    boxes_pred = prediction["boxes"]
    scores_pred = prediction["scores"]
    labels_pred = prediction["labels"]

    boxes_gt = target["boxes"]
    labels_gt = target["labels"]

    if boxes_pred.numel() == 0 or boxes_gt.numel() == 0:
        return [], []

    # IoU matrix: [num_pred, num_gt]
    ious = box_iou(boxes_pred, boxes_gt)

    # Greedy matching: each GT can be matched at most once
    matched_true = []
    matched_pred = []

    num_gt = boxes_gt.size(0)
    gt_matched = torch.zeros(num_gt, dtype=torch.bool)

    # sort predictions by score (desc)
    scores_sorted, order = scores_pred.sort(descending=True)

    for idx in order:
        if scores_sorted[order == idx] < score_thr:
            # all remaining predictions have lower scores
            break

        iou_vals = ious[idx]
        max_iou, gt_idx = iou_vals.max(dim=0)

        if max_iou >= iou_thr and not gt_matched[gt_idx]:
            gt_matched[gt_idx] = True
            matched_true.append(int(labels_gt[gt_idx].item()))
            matched_pred.append(int(labels_pred[idx].item()))

    return matched_true, matched_pred


def compute_classification_metrics(y_true, y_pred, num_classes):
    """
    Compute overall and macro-averaged classification metrics from
    matched (y_true, y_pred) pairs.

    Args:
        y_true (List[int]): ground-truth labels (1..num_classes-1).
        y_pred (List[int]): predicted labels (1..num_classes-1).
        num_classes (int): total number of classes (including background if any).

    Returns:
        dict with accuracy, macro_precision, macro_recall, macro_f1.
    """
    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
        }

    y_true = torch.tensor(y_true, dtype=torch.long)
    y_pred = torch.tensor(y_pred, dtype=torch.long)

    # Confusion matrix: [num_classes, num_classes]
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # In our dataset, classes are 1..12 (0 is unused), so we skip index 0
    cls_ids = torch.arange(1, num_classes)
    sub_cm = cm[cls_ids][:, cls_ids]

    tp = sub_cm.diag()
    support = sub_cm.sum(dim=1)  # gt count per class
    pred_sum = sub_cm.sum(dim=0)  # pred count per class

    precision_per_class = torch.where(
        pred_sum > 0, tp.float() / pred_sum.float(), torch.zeros_like(tp, dtype=torch.float)
    )
    recall_per_class = torch.where(
        support > 0, tp.float() / support.float(), torch.zeros_like(tp, dtype=torch.float)
    )
    f1_per_class = torch.where(
        (precision_per_class + recall_per_class) > 0,
        2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class),
        torch.zeros_like(tp, dtype=torch.float),
    )

    macro_precision = precision_per_class.mean().item()
    macro_recall = recall_per_class.mean().item()
    macro_f1 = f1_per_class.mean().item()

    accuracy = tp.sum().float() / sub_cm.sum().float()

    return {
        "accuracy": accuracy.item(),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def evaluate():
    print(f"Starting evaluation on device: {config.DEVICE}")

    model = load_trained_model()

    # For detection mAP (keep it **on CPU** to save GPU memory)
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)

    all_true_cls = []
    all_pred_cls = []

    start_time = time.time()

    with torch.no_grad():
        for images, targets in tqdm(data_loader_test, desc="Evaluating"):
            # 1) Send images to device for model inference
            images = [img.to(config.DEVICE) for img in images]
            outputs = model(images)

            # 2) Move outputs & targets back to CPU for metric + classification stats
            outputs_cpu = [{k: v.cpu() for k, v in out.items()} for out in outputs]
            targets_cpu = [{k: v.cpu() for k, v in tgt.items()} for tgt in targets]

            # 3) Update detection metric
            metric.update(outputs_cpu, targets_cpu)

            # 4) Collect classification pairs (gt, pred) for each image
            for pred_dict, gt_dict in zip(outputs_cpu, targets_cpu):
                matched_true, matched_pred = match_predictions_to_targets(
                    pred_dict, gt_dict, iou_thr=0.5, score_thr=0.5
                )
                all_true_cls.extend(matched_true)
                all_pred_cls.extend(matched_pred)

    total_time = time.time() - start_time
    results_det = metric.compute()

    # ----- Print results -----

    print(f"\nTotal Test Time: {total_time:.2f}s")

    # Detection metrics
    print("\n--- DETECTION Metrics (mAP) ---")
    print(f"mAP (IoU 0.50:0.95):         {results_det['map']:.4f}")
    print(f"mAP (IoU = 0.50 / mAP@.50):  {results_det['map_50']:.4f}")
    print(f"mAP (IoU = 0.75 / mAP@.75):  {results_det['map_75']:.4f}")

    # “AUC-like” metric：用 mAP 作为 PR 曲线下的面积 proxy
    print(f"AUC-like metric (mAP):       {results_det['map']:.4f}")

    # Classification metrics from matched boxes
    cls_metrics = compute_classification_metrics(
        all_true_cls, all_pred_cls, num_classes=config.NUM_CLASSES
    )

    print("\n--- CLASSIFICATION Metrics (from matched detections) ---")
    print(f"Accuracy:                    {cls_metrics['accuracy']:.4f}")
    print(f"Macro Precision:             {cls_metrics['macro_precision']:.4f}")
    print(f"Macro Recall:                {cls_metrics['macro_recall']:.4f}")
    print(f"Macro F1 Score:              {cls_metrics['macro_f1']:.4f}")

    # Per-class mAP (可以在报告里当作“per-class AP/Accuracy”)
    print("\n--- Per-Class mAP ---")
    class_ids = results_det["classes"]
    map_per_class = results_det["map_per_class"]

    if class_ids.dim() == 0:
        class_ids = class_ids.unsqueeze(0)
    if map_per_class.dim() == 0:
        map_per_class = map_per_class.unsqueeze(0)

    for cid, ap in zip(class_ids, map_per_class):
        print(f"  Class {cid.item():2d}: mAP = {ap.item():.4f}")


if __name__ == "__main__":
    evaluate()
