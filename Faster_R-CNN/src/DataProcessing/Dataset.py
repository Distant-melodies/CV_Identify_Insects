# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# File       : PestDataset
# Time       ：2025/11/7 13:31
# Author     ：Jingyang Dai
# zId        ：z5553615
# Description：
"""
import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from tqdm import tqdm


class PestDataset(Dataset):
    """
    Custom PyTorch Dataset for the AgroPest-12 dataset.
    (Optimized Version: Pre-loads all annotations into RAM.)
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Path to the 'train', 'valid', or 'test' directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        # --- OPTIMIZATION 1: PRE-LOAD ALL ANNOTATIONS ---

        # We removed the extra 'print' statement.
        # This description will now appear in the progress bar.
        desc_text = f"Pre-loading annotations for {os.path.basename(root_dir)}"
        self.all_annotations = []

        for img_name in tqdm(self.image_files, desc=desc_text):
            img_path = os.path.join(self.image_dir, img_name)
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(self.label_dir, label_name)

            with Image.open(img_path) as img:
                img_width, img_height = img.size

            boxes = []
            labels = []

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(float(parts[0]))
                            x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts[1:])

                            x_center_abs = x_center_norm * img_width
                            y_center_abs = y_center_norm * img_height
                            width_abs = width_norm * img_width
                            height_abs = height_norm * img_height

                            x_min = x_center_abs - (width_abs / 2)
                            y_min = y_center_abs - (height_abs / 2)
                            x_max = x_center_abs + (width_abs / 2)
                            y_max = y_center_abs + (height_abs / 2)

                            boxes.append([x_min, y_min, x_max, y_max])
                            labels.append(class_id + 1)

            self.all_annotations.append({
                'img_name': img_name,
                'boxes': boxes,
                'labels': labels
            })
        # We also removed the 'print("Pre-loading complete.")' line
        # --- END OPTIMIZATION 1 ---

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # --- OPTIMIZATION 2: __getitem__ IS NOW MUCH FASTER ---

        annotation = self.all_annotations[idx]

        img_name = annotation['img_name']
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        boxes = torch.as_tensor(annotation['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(annotation['labels'], dtype=torch.int64)

        if boxes.shape == torch.Size([0]):
            boxes = boxes.reshape(0, 4)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes.shape[0] > 0 else torch.tensor(0.0)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform:
            image = self.transform(image)

        return image, target


def get_transform(train):
    """Defines the transformations to be applied to the images."""
    transforms = []
    transforms.append(T.ToTensor())

    if train:
        pass

    return T.Compose(transforms)
