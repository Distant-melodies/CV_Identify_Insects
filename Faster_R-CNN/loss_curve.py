# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# File       : loss_curve
# Time       ：2025/11/15 19:14
# Author     ：Jingyang Dai
# zId        ：z5553615
# Description：
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path("../results")
train = np.load(results_dir / "train_losses.npy")
val = np.load(results_dir / "val_losses.npy")

plt.figure()
plt.plot(train, label="train")
plt.plot(val, label="valuation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(results_dir / "loss_curve.png", dpi=200)
plt.close()
