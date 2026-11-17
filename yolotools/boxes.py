"""Box utilities (NumPy only).

Conventions
-----------
xyxy : (x1, y1, x2, y2)   continuous edge coordinates (x2 > x1)
xywh : (xc, yc, w, h)     centre + size
IoU is computed on continuous coordinates. Because IoU is invariant to scaling each
axis separately, it gives identical values on pixel or on normalised (0-1) boxes.
"""
import numpy as np


def xywh_to_xyxy(b):
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                     b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], axis=1)


def xyxy_to_xywh(b):
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                     b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1)


def box_area(b):
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def iou_matrix(a, b):
    """Pairwise IoU between a (N,4) and b (M,4) xyxy boxes -> (N,M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def iou_one_to_many(box, boxes):
    return iou_matrix(np.asarray(box).reshape(1, 4), boxes)[0]
