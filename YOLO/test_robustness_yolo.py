import os
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import yaml
from ultralytics import YOLO

# 1. 自适应路径配置
CURRENT_DIR = Path(__file__).resolve().parent  # 指向 .../YOLO
PROJECT_ROOT = CURRENT_DIR.parent  # 指向工程根目录 .../CV_Identify_Insects

# 权重路径（优先查找之前训练的最优模型权重）
YOLO_WEIGHTS = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "runs"
    / "insects_yolo"
    / "exp_yolov8m_fast"
    / "weights"
    / "best.pt"
)
if not YOLO_WEIGHTS.exists():
  # 备用容错路径
  alt_weights = (
      PROJECT_ROOT
      / "runs"
      / "insects_yolo"
      / "exp_yolov8m_fast"
      / "weights"
      / "best.pt"
  )
  if alt_weights.exists():
    YOLO_WEIGHTS = alt_weights

# 数据源路径：严格对应 archive/valid/images 和 archive/valid/labels
VAL_IMAGES_DIR = PROJECT_ROOT / "archive" / "valid" / "images"
VAL_LABELS_DIR = PROJECT_ROOT / "archive" / "valid" / "labels"

# 临时压测文件夹放在当前 YOLO 目录下
TEMP_BENCHMARK_DIR = CURRENT_DIR / "temp_distorted_benchmark"

# 读取 YOLO 目录下的 insect_data.yaml 获取已有的类别名称字典
INSECT_YAML_PATH = CURRENT_DIR / "insect_data.yaml"
if INSECT_YAML_PATH.exists():
  with open(INSECT_YAML_PATH, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)
    CLASS_NAMES = _cfg.get("names", {})
else:
  CLASS_NAMES = {i: f"insect_class_{i}" for i in range(12)}

# 扰动级别配置 (对齐现有测试基线)
BRIGHTNESS_FACTORS = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5]  # 光照骤降到过曝
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2]  # 高斯噪声标准差 sigma


# 2. 图像扰动退化算子 (复用现有核心逻辑)
def apply_distortion(img_bgr, factor=1.0, sigma=0.0):
  """将 BGR 图像转换为 Tensor，施加确定性光照与高斯噪声，并通过 clamp 截断"""
  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
  img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

  # 1. 光照退化
  if factor != 1.0:
    jitter = T.ColorJitter(brightness=(factor, factor))
    img_tensor = jitter(img_tensor)

  # 2. 弱光高斯噪声与截断
  if sigma > 0.0:
    noise = torch.randn_like(img_tensor) * sigma
    img_tensor = img_tensor + noise

  img_clamped = torch.clamp(img_tensor, 0.0, 1.0)
  distorted_bgr = (
      (img_clamped.permute(1, 2, 0).numpy() * 255)
      .astype(np.uint8)[:, :, ::-1]
      .copy()
  )
  return distorted_bgr


# 3. 构造微型临时验证子集 (修复 train 缺失问题)
def create_distorted_dataset(
    src_img_dir, src_lbl_dir, dst_dir, factor, sigma, max_samples=200
):
  """生成带扰动的临时验证集并配置规范的 YAML"""
  img_dst = dst_dir / "images"
  lbl_dst = dst_dir / "labels"

  if dst_dir.exists():
    shutil.rmtree(dst_dir)
  img_dst.mkdir(parents=True, exist_ok=True)
  lbl_dst.mkdir(parents=True, exist_ok=True)

  # 采样验证集图片
  img_paths = sorted(list(src_img_dir.glob("*.*")))[:max_samples]
  if not img_paths:
    raise FileNotFoundError(f"未在 {src_img_dir} 中找到任何测试图片！")

  for img_p in img_paths:
    raw_img = cv2.imread(str(img_p))
    if raw_img is None:
      continue

    # 施加光照/加噪退化
    distorted = apply_distortion(raw_img, factor=factor, sigma=sigma)
    cv2.imwrite(str(img_dst / img_p.name), distorted)

    # 拷贝对应的 label 文件
    lbl_p = src_lbl_dir / f"{img_p.stem}.txt"
    if lbl_p.exists():
      shutil.copy(str(lbl_p), str(lbl_dst / lbl_p.name))

  # 必须同时提供 train 和 val 键，否则 YOLO 会抛出 SyntaxError
  yaml_dict = {
      "path": str(dst_dir.resolve()),
      "train": "images",  # 指向 images，满足 YOLO 格式校验
      "val": "images",  # 评估时使用的验证路径
      "names": CLASS_NAMES,
  }

  yaml_path = dst_dir / "temp_data.yaml"
  with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(yaml_dict, f, allow_unicode=True, sort_keys=False)

  return yaml_path


# 4. 执行全流程扰动压测
def run_robustness_test():
  print("=" * 65)
  print("        启动 YOLOv8 目标检测鲁棒性扰动基准压测        ")
  print("=" * 65)
  print(f"当前工作根目录: {PROJECT_ROOT}")
  print(f"最优模型权重:   {YOLO_WEIGHTS}")
  print(f"验证集来源:     {VAL_IMAGES_DIR}")

  if not YOLO_WEIGHTS.exists():
    raise FileNotFoundError(f"未找到最优权重文件: {YOLO_WEIGHTS}")

  model = YOLO(str(YOLO_WEIGHTS))

  # 阶段 1: 光照退化压测 (固定 sigma=0)
  print("\n>>> 开始光照退化（Brightness Factor）基准压测...")
  brightness_results = []
  for factor in BRIGHTNESS_FACTORS:
    temp_yaml = create_distorted_dataset(
        VAL_IMAGES_DIR,
        VAL_LABELS_DIR,
        TEMP_BENCHMARK_DIR,
        factor=factor,
        sigma=0.0,
        max_samples=200,
    )
    metrics = model.val(
        data=str(temp_yaml), imgsz=800, conf=0.25, iou=0.6, verbose=False
    )
    map50 = metrics.box.map50 * 100
    recall = metrics.box.mr * 100
    brightness_results.append((factor, map50, recall))
    print(
        f"  [光照 f={factor:.1f}] -> mAP@0.50: {map50:.2f}%, Recall:"
        f" {recall:.2f}%"
    )

  # 阶段 2: 弱光高斯噪声压测 (固定 factor=1.0)
  print("\n>>> 开始高斯噪声退化（Gaussian Noise Sigma）基准压测...")
  noise_results = []
  for sigma in NOISE_LEVELS:
    temp_yaml = create_distorted_dataset(
        VAL_IMAGES_DIR,
        VAL_LABELS_DIR,
        TEMP_BENCHMARK_DIR,
        factor=1.0,
        sigma=sigma,
        max_samples=200,
    )
    metrics = model.val(
        data=str(temp_yaml), imgsz=800, conf=0.25, iou=0.6, verbose=False
    )
    map50 = metrics.box.map50 * 100
    recall = metrics.box.mr * 100
    noise_results.append((sigma, map50, recall))
    print(
        f"  [高斯噪声 sigma={sigma:.2f}] -> mAP@0.50: {map50:.2f}%, Recall:"
        f" {recall:.2f}%"
    )

  # 压测结束，清理临时生成的扰动图片目录
  if TEMP_BENCHMARK_DIR.exists():
    shutil.rmtree(TEMP_BENCHMARK_DIR)

  # 5. 打印并输出汇总 Markdown 报告
  report = """
### YOLOv8 扰动鲁棒性压测对照结果 (Benchmark Summary)

#### 1. 确定性光照退化响应 (ColorJitter Brightness)
| 光照倍率 (Factor) | 对应场景环境 | mAP@0.50 (%) | Recall (%) | 性能衰减幅度 |
| :--- | :--- | :--- | :--- | :--- |
"""
  base_map = brightness_results[3][1]  # f=1.0 为正常光照基准
  scene_desc = [
      "极限暗光 (夜间)",
      "弱光/阴影",
      "微弱昏暗",
      "正常基准光照",
      "轻度过曝",
      "强光过曝",
  ]
  for i, (f, m, r) in enumerate(brightness_results):
    diff = m - base_map
    report += f"| **{f:.1f}** | {scene_desc[i]} | {m:.2f}% | {r:.2f}% | {diff:+.2f}% |\n"

  report += """
#### 2. 弱光传感器高斯噪声响应 (Gaussian Noise & Clamp)
| 噪声强度 (Sigma) | 对应干扰等级 | mAP@0.50 (%) | Recall (%) | 性能衰减幅度 |
| :--- | :--- | :--- | :--- | :--- |
"""
  base_noise_map = noise_results[0][1]
  noise_desc = [
      "无噪声 (基准)",
      "轻微传感器散粒噪声",
      "中等高 ISO 噪声",
      "重度恶劣噪点",
  ]
  for i, (s, m, r) in enumerate(noise_results):
    diff = m - base_noise_map
    report += f"| **{s:.2f}** | {noise_desc[i]} | {m:.2f}% | {r:.2f}% | {diff:+.2f}% |\n"

  print("\n" + "=" * 65)
  print(report)

  report_save_path = CURRENT_DIR / "robustness_benchmark_report.md"
  with open(report_save_path, "w", encoding="utf-8") as f:
    f.write(report)
  print(f"[OK] 压测报告已成功保存至: {report_save_path}")


if __name__ == "__main__":
  run_robustness_test()