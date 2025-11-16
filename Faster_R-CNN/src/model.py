# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# File       : model
# Time       ：2025/11/7 14:18
# Author     ：Jingyang Dai
# zId        ：z5553615
# Description：
"""
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights, \
    fasterrcnn_resnet50_fpn
from src import config


def get_model():
    """
    Loads a pre-trained Faster R-CNN model and modifies the
    classification head for our specific number of classes.
    """

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config.NUM_CLASSES)

    return model

# if __name__ == '__main__':
#     # A small test to ensure the model builds correctly
#     print("Building model...")
#     model = get_model()
#     print("Model built successfully.")
#     print(f"Model will predict for {config.NUM_CLASSES} classes.")
