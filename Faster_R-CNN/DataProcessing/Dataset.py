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


class PestDataset(Dataset):
    """
    Custom PyTorch Dataset for the AgroPest-12 dataset.
    This class assumes labels are in YOLO format (.txt files).
    """

    def __init__(self, root_dir, transform=None):
        """

        :param root_dir: Path to the 'train', 'valid', or 'test' directory.
        :param transform: Optional transform to be applied on a sample.
        """

        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        self.transform = transform

        # Get all image filenames (assuming they are sorted and match)
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        # Open with PIL.Image.open for compatibility with torchvision transforms
        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size

        # Load label
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    # Parse YOLO format: class_id x_center y_center width height
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(float(parts[0]))
                        x_center_norm = float(parts[1])
                        y_center_norm = float(parts[2])
                        width_norm = float(parts[3])
                        height_norm = float(parts[4])

                        # Coordinate Conversion
                        # Convert normalized YOLO coords to [x_min, y_min, x_max, y_max] pixel coords
                        x_center_abs = x_center_norm * img_width
                        y_center_abs = y_center_norm * img_height
                        width_abs = width_norm * img_width
                        height_abs = height_norm * img_height

                        x_min = x_center_abs - (width_abs / 2)
                        y_min = y_center_abs - (height_abs / 2)
                        x_max = x_center_abs + (width_abs / 2)
                        y_max = y_center_abs + (height_abs / 2)

                        boxes.append([x_min, y_min, x_max, y_max])

                        # Map class_id (0-11) to labels (1-12)
                        # 0 is reserved for background
                        labels.append(class_id + 1)

        # Convert to torch Tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])

        # Calculate area (or 0 if no boxes)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes.shape[0] > 0 else torch.tensor(0.0)

        # Assume all instances are not crowds
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        # Pack the Target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # Apply transforms (e.g., ToTensor)
        if self.transform:
            image = self.transform(image)

        return image, target


def get_transform(train):
    """Defines the transformations to be applied to the images."""
    transforms = []
    # Converts PIL image to PyTorch Tensor
    transforms.append(T.ToTensor())

    if train:
        # Add simple data augmentation during training
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)
