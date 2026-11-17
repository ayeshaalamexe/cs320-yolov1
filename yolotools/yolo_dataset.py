"""Loader for the YOLO-format Penn-Fudan folder (images/<split>, labels/<split>).

Interface matches the team's YOLOv1 code: __getitem__ -> (image CHW float tensor 0-1, boxes (N,4) tensor
[xc,yc,w,h] normalised) and `collate_fn` (variable number of boxes per image). Also exposes
`ground_truth(i)` and `image_tensor(i)` for evaluate.py.
"""
import os
import numpy as np
from PIL import Image
from .augment import random_augment
from .boxes import xywh_to_xyxy, xyxy_to_xywh

try:
    import torch
    from torch.utils.data import Dataset as _Base
except ImportError:                                   # lets the module be imported / tested without torch
    torch = None
    _Base = object


def read_yolo_label(path):
    if not os.path.exists(path):
        return np.zeros(0, int), np.zeros((0, 4))
    rows = [ln.split() for ln in open(path) if ln.strip()]
    if not rows:
        return np.zeros(0, int), np.zeros((0, 4))
    arr = np.array(rows, dtype=np.float64)
    return arr[:, 0].astype(int), arr[:, 1:5]


class YoloFolderDataset(_Base):
    def __init__(self, root, split="train", img_size=448, augment=False, aug_kwargs=None):
        self.root, self.split, self.img_size, self.augment = root, split, img_size, augment
        self.aug_kwargs = aug_kwargs or {}
        d = os.path.join(root, "images", split)
        self.files = sorted(f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg")))

    def __len__(self):
        return len(self.files)

    def _raw(self, i):
        f = self.files[i]
        img = Image.open(os.path.join(self.root, "images", self.split, f)).convert("RGB")
        labels, xywh = read_yolo_label(os.path.join(self.root, "labels", self.split, os.path.splitext(f)[0] + ".txt"))
        return img, labels, xywh

    def sample(self, i, rng=None):
        """numpy sample: (uint8 HxWx3 image, boxes_xywh (N,4) float32, labels)."""
        img, labels, xywh = self._raw(i)
        boxes = xywh_to_xyxy(xywh)
        if self.augment:
            img, boxes, labels = random_augment(img, boxes, labels, rng, **self.aug_kwargs)
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        return np.asarray(img), xyxy_to_xywh(boxes).astype(np.float32), labels

    def ground_truth(self, i):
        _, labels, xywh = self._raw(i)
        return {"boxes": xywh_to_xyxy(xywh), "labels": labels, "difficult": np.zeros(len(labels), bool)}

    def image_tensor(self, i):
        img, _, _ = self._raw(i)
        arr = np.asarray(img.resize((self.img_size, self.img_size), Image.BILINEAR))
        return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, i):
        arr, boxes, _ = self.sample(i)
        return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0, torch.from_numpy(boxes.reshape(-1, 4))


def collate_fn(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]
