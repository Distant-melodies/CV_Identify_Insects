# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# File       : train_yolo.py
# Time       : 2026/9/24 19:45
# Author     : Brett Dai
# zId        : z5553615
"""

import os
from ultralytics import YOLO


def run_yolo_train():
  # 初始化预训练模型
  model = YOLO("yolov8m.pt")

  # 启动训练
  results = model.train(
      data="insect_data.yaml",
      epochs=100,
      # --- 提速关键配置 ---
      imgsz=800,  # 相比 1024 提速近 40%，且对小昆虫定位精度几乎不降
      batch=8,
      workers=4,  # 若在 Windows 下卡住/报错可改回 2
      cache="ram",  # 核心！将预处理数据缓存在内存中，彻底消灭磁盘 I/O 阻塞
      # -------------------
      device=0,
      optimizer="AdamW",
      lr0=1e-3,
      lrf=0.01,
      amp=True,  # 维持混合精度加速
      close_mosaic=10,
      # --- 减少冗余计算 ---
      plots=False,  # 训练中途不频繁绘制预测图，减少磁盘写入
      save_period=10,  # 每 10 轮保存一次权重，减少保存 checkpoint 开销
      project="runs/insects_yolo",
      name="exp_yolov8m_fast",
      save=True,
      exist_ok=True,
  )

  print("训练完成！最优模型权重保存在 runs/insects_yolo/exp_yolov8m_1024/weights/best.pt")


if __name__ == "__main__":
  run_yolo_train()