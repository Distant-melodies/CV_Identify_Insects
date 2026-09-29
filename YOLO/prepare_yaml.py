# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# File       : prepare_yaml.py
# Time       : 2026/9/24 19:23
# Author     : Brett Dai
# zId        : z5553615
"""

from pathlib import Path
import yaml


def generate_data_yaml(
    archive_dir="../archive", output_yaml="insect_data.yaml"
):
  archive_path = Path(archive_dir)

  # 扫描所有 txt 查找出现的 class_id
  all_classes = set()
  txt_files = list(archive_path.glob("**/*.txt"))

  if not txt_files:
    raise FileNotFoundError(f"在 {archive_dir} 及其子目录下未找到 .txt 文件！")

  for txt in txt_files:
    with open(txt, "r") as f:
      for line in f:
        parts = line.strip().split()
        if parts:
          all_classes.add(int(parts[0]))

  sorted_classes = sorted(list(all_classes))
  num_classes = len(sorted_classes)
  print(
      f"扫描完成：共识别到 {num_classes} 个昆虫类别，类别ID: {sorted_classes}"
  )

  # 构建默认名称字典（若有物种名单可在此替换具体昆虫名）
  class_names = {cid: f"insect_class_{cid}" for cid in sorted_classes}

  # 判断目录结构形态
  has_split_dirs = (archive_path / "images" / "train").exists()

  data_config = {
      "path": str(archive_path.resolve()),
      "train": (
          "images/train"
          if has_split_dirs
          else (
              "train" if (archive_path / "train").exists() else str(archive_path)
          )
      ),
      "val": (
          "images/val"
          if has_split_dirs
          else ("val" if (archive_path / "val").exists() else str(archive_path))
      ),
      "names": class_names,
  }

  with open(output_yaml, "w", encoding="utf-8") as f:
    yaml.dump(data_config, f, allow_unicode=True, sort_keys=False)

  print(f"配置文件已生成至: {output_yaml}")


if __name__ == "__main__":
  generate_data_yaml()