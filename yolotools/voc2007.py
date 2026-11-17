"""PASCAL VOC 2007: download, XML parsing, loader and YOLOv1 target encoder.

Coordinates: VOC boxes are 1-indexed inclusive pixels -> continuous edges (xmin-1, ymin-1, xmax, ymax),
then normalised to 0-1 by the image size. Images are resized straight to img_size x img_size (as in YOLOv1).
Training excludes `difficult` objects (as darknet's voc_label.py does); evaluation keeps them and flags
them so metrics.py ignores them (official VOC protocol).
"""
import os, subprocess, tarfile, urllib.request
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from .augment import random_augment
from .boxes import xyxy_to_xywh

try:
    import torch
    from torch.utils.data import Dataset as _Base
except ImportError:
    torch = None
    _Base = object

VOC_CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor")
CLASS_TO_ID = {c: i for i, c in enumerate(VOC_CLASSES)}
_OX = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/"
_PJ = "https://pjreddie.com/media/files/"
TARBALLS = {"trainval": "VOCtrainval_06-Nov-2007.tar", "test": "VOCtest_06-Nov-2007.tar"}
EXPECTED_IMAGES = {"trainval": 5011, "test": 4952, "train": 2501, "val": 2510}


def download_voc2007(dest="/content/data", parts=("trainval", "test")):
    """Download + extract VOC2007 (tries the Oxford host, then a mirror). Returns .../VOCdevkit/VOC2007."""
    os.makedirs(dest, exist_ok=True)
    devkit = os.path.join(dest, "VOCdevkit", "VOC2007")
    for part in parts:
        marker = os.path.join(devkit, "ImageSets", "Main", f"{part}.txt")
        if os.path.exists(marker):
            continue
        tar_path = os.path.join(dest, TARBALLS[part])
        for base in (_OX, _PJ):
            try:
                if subprocess.call(["wget", "-q", "-c", "--tries=3", "-O", tar_path, base + TARBALLS[part]]) == 0 \
                        and os.path.getsize(tar_path) > 1e8:
                    break
            except FileNotFoundError:
                urllib.request.urlretrieve(base + TARBALLS[part], tar_path)
                break
        else:
            raise RuntimeError(f"could not download {TARBALLS[part]}; download it manually into {dest}")
        with tarfile.open(tar_path) as t:
            t.extractall(dest)
        os.remove(tar_path)
    return devkit


def parse_voc_xml(path):
    """-> dict(filename, width, height, objects=[dict(name, label, difficult, truncated, box_edges)])."""
    r = ET.parse(path).getroot()
    size = r.find("size")
    out = {"filename": r.findtext("filename"), "width": int(size.findtext("width")),
           "height": int(size.findtext("height")), "objects": []}
    for o in r.findall("object"):
        bb = o.find("bndbox")
        xmin, ymin, xmax, ymax = (int(float(bb.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax"))
        out["objects"].append({"name": o.findtext("name").strip(), "label": CLASS_TO_ID[o.findtext("name").strip()],
                               "difficult": int(o.findtext("difficult") or 0),
                               "truncated": int(o.findtext("truncated") or 0),
                               "box_edges": (xmin - 1, ymin - 1, xmax, ymax)})
    return out


def read_image_ids(devkit, image_set):
    with open(os.path.join(devkit, "ImageSets", "Main", f"{image_set}.txt")) as f:
        return [ln.strip() for ln in f if ln.strip()]


def encode_yolov1_target(boxes_xywh, labels, S=7, C=20, collision="first"):
    """-> (S,S,5+C) float32 laid out as [x_off, y_off, w, h, obj, one-hot C] (same as the team's encode_targets).

    One object per cell (YOLOv1 limitation). collision='first' keeps the first object, 'largest' keeps the bigger one.
    """
    t = np.zeros((S, S, 5 + C), dtype=np.float32)
    for (xc, yc, w, h), c in zip(np.asarray(boxes_xywh).reshape(-1, 4), labels):
        j, i = min(int(xc * S), S - 1), min(int(yc * S), S - 1)
        if t[i, j, 4] == 1 and (collision == "first" or w * h <= t[i, j, 2] * t[i, j, 3]):
            continue
        t[i, j, :4] = (xc * S - j, yc * S - i, w, h)
        t[i, j, 4] = 1
        t[i, j, 5:] = 0
        t[i, j, 5 + int(c)] = 1
    return t


def decode_target(t, S=7):
    """Inverse of encode_yolov1_target (for visual checks) -> boxes_xywh (K,4), labels (K,)."""
    ii, jj = np.where(t[..., 4] == 1)
    boxes = np.array([[(j + t[i, j, 0]) / S, (i + t[i, j, 1]) / S, t[i, j, 2], t[i, j, 3]] for i, j in zip(ii, jj)]).reshape(-1, 4)
    return boxes, np.array([int(np.argmax(t[i, j, 5:])) for i, j in zip(ii, jj)], dtype=int)


class VOC2007Detection(_Base):
    """VOC2007 detection dataset.

    __getitem__ -> (image CHW float 0-1, target (S,S,5+C)).  For evaluation use `ground_truth(i)`
    (normalised xyxy + labels + difficult, unaugmented) and `image_tensor(i)`.
    """
    classes = VOC_CLASSES

    def __init__(self, devkit, image_set="trainval", img_size=448, S=7, augment=False,
                 keep_difficult=False, collision="first", aug_kwargs=None):
        self.devkit, self.image_set, self.img_size, self.S = devkit, image_set, img_size, S
        self.augment, self.keep_difficult, self.collision = augment, keep_difficult, collision
        self.aug_kwargs = aug_kwargs or {}
        self.ids = read_image_ids(devkit, image_set)
        self._ann_cache = {}

    def __len__(self):
        return len(self.ids)

    def annotation(self, i):
        if i not in self._ann_cache:
            self._ann_cache[i] = parse_voc_xml(os.path.join(self.devkit, "Annotations", self.ids[i] + ".xml"))
        return self._ann_cache[i]

    def _image(self, i):
        return Image.open(os.path.join(self.devkit, "JPEGImages", self.ids[i] + ".jpg")).convert("RGB")

    def ground_truth(self, i):
        a = self.annotation(i)
        scale = np.array([a["width"], a["height"], a["width"], a["height"]], float)
        e = np.array([o["box_edges"] for o in a["objects"]], float).reshape(-1, 4) / scale
        return {"boxes": e, "labels": np.array([o["label"] for o in a["objects"]], dtype=int),
                "difficult": np.array([bool(o["difficult"]) for o in a["objects"]], dtype=bool)}

    def sample(self, i, rng=None):
        """numpy sample: (uint8 img_size x img_size x 3, boxes_xywh (N,4), labels (N,))."""
        gt = self.ground_truth(i)
        keep = np.ones(len(gt["labels"]), bool) if self.keep_difficult else ~gt["difficult"]
        boxes, labels = gt["boxes"][keep], gt["labels"][keep]
        img = self._image(i)
        if self.augment:
            img, boxes, labels = random_augment(img, boxes, labels, rng, **self.aug_kwargs)
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        return np.asarray(img), xyxy_to_xywh(boxes).astype(np.float32), labels

    def image_tensor(self, i):
        arr = np.asarray(self._image(i).resize((self.img_size, self.img_size), Image.BILINEAR))
        return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, i):
        arr, boxes, labels = self.sample(i)
        target = encode_yolov1_target(boxes, labels, self.S, len(VOC_CLASSES), self.collision)
        return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0, torch.from_numpy(target)


def dataset_statistics(devkit, image_set):
    """Counts per class (objects / non-difficult objects / images containing the class)."""
    ids = read_image_ids(devkit, image_set)
    obj, nd, imgs = np.zeros(20, int), np.zeros(20, int), np.zeros(20, int)
    for i in ids:
        a = parse_voc_xml(os.path.join(devkit, "Annotations", i + ".xml"))
        seen = set()
        for o in a["objects"]:
            obj[o["label"]] += 1
            nd[o["label"]] += 1 - o["difficult"]
            seen.add(o["label"])
        for c in seen:
            imgs[c] += 1
    return {"images": len(ids), "objects": obj, "non_difficult": nd, "images_with_class": imgs}
