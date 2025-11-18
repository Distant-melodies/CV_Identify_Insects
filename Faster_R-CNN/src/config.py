# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# File       : config
# Time       ：2025/11/7 13:46
# Author     ：Jingyang Dai
# zId        ：z5553615
# Description：
"""

import os
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


ARCHIVE_DIR = ROOT_DIR / 'archive'

TRAIN_DIR = ARCHIVE_DIR / 'train'
VALID_DIR = ARCHIVE_DIR / 'valid'
TEST_DIR = ARCHIVE_DIR / 'test'


NUM_CLASSES = 13
BATCH_SIZE = 2
NUM_WORKERS = 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_EPOCHS = 25


# Imbalanced training experiment settings
USE_IMBALANCED_TRAIN = True
MINORITY_CLASSES = [1, 3, 5]
# Random seed used for building the imbalanced subset
IMBALANCE_SEED = 98


# class-balanced WeightedRandomSample
USE_BALANCED_SAMPLER = True
