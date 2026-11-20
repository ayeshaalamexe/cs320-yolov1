"""Non-Maximum Suppression and YOLOv1 output decoding (NumPy).

NMS rule (same as torchvision.ops.nms): visit boxes in descending score order, keep a
box, and suppress every remaining box whose IoU with it is STRICTLY greater than
`iou_thr`. Ties in score are broken by original index (stable sort).
"""
import numpy as np
from .boxes import iou_one_to_many, xywh_to_xyxy


def to_numpy(x):
    """Accept numpy arrays, lists or torch tensors."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def nms(boxes, scores, iou_thr=0.5, max_keep=None):
    """Greedy NMS. boxes (N,4) xyxy, scores (N,). Returns kept indices (descending score)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(boxes) != len(scores):
        raise ValueError("boxes and scores must have the same length")
    if not 0.0 <= iou_thr <= 1.0:
        raise ValueError("iou_thr must be in [0, 1]")
    n = len(boxes)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    suppressed = np.zeros(n, dtype=bool)
    keep = []
    for pos in range(n):
        i = order[pos]
        if suppressed[i]:
            continue
        keep.append(i)
        if max_keep is not None and len(keep) >= max_keep:
            break
        rest = order[pos + 1:]
        rest = rest[~suppressed[rest]]
        if len(rest):
            suppressed[rest[iou_one_to_many(boxes[i], boxes[rest]) > iou_thr]] = True
    return np.asarray(keep, dtype=np.int64)


def batched_nms(boxes, scores, labels, iou_thr=0.5, max_keep=None):
    """Class-aware NMS: boxes of different classes never suppress each other."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    kept = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        kept.append(idx[nms(boxes[idx], scores[idx], iou_thr)])
    kept = np.concatenate(kept)
    kept = kept[np.argsort(-scores[kept], kind="stable")]
    return kept[:max_keep] if max_keep is not None else kept


def decode_yolov1(pred, B=2, C=20, score_thr=0.05, clip=True):
    """Decode ONE image's raw YOLOv1 output (S,S,B*5+C) into candidate detections.

    Layout per cell: [x,y,w,h,conf]*B then C class probabilities, where x,y are offsets
    inside the cell and w,h are fractions of the whole image (as in the team's
    encode_targets / YOLOLoss). Score of box b for class c = conf_b * P(c | object).
    Returns boxes (K,4) xyxy in 0-1 image coordinates, scores (K,), labels (K,).
    """
    pred = to_numpy(pred).astype(np.float64)
    S = pred.shape[0]
    if pred.shape != (S, S, B * 5 + C):
        raise ValueError(f"expected (S,S,{B * 5 + C}), got {pred.shape}")
    rows, cols = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    cls_p = pred[..., B * 5:]
    all_boxes, all_scores, all_labels = [], [], []
    for b in range(B):
        x, y, w, h, conf = (pred[..., b * 5 + k] for k in range(5))
        xc, yc = (cols + x) / S, (rows + y) / S
        xyxy = xywh_to_xyxy(np.stack([xc, yc, np.abs(w), np.abs(h)], -1).reshape(-1, 4))
        scores = conf.reshape(-1, 1) * cls_p.reshape(-1, C)          # (S*S, C)
        cell, cls = np.where(scores > score_thr)
        all_boxes.append(xyxy[cell])
        all_scores.append(scores[cell, cls])
        all_labels.append(cls)
    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels).astype(np.int64)
    if clip:
        boxes = np.clip(boxes, 0.0, 1.0)
    return boxes, scores, labels


def postprocess_yolov1(pred, B=2, C=20, score_thr=0.05, iou_thr=0.5, max_det=100):
    """decode_yolov1 + class-aware NMS + top-k. Returns dict(boxes, scores, labels)."""
    boxes, scores, labels = decode_yolov1(pred, B, C, score_thr)
    keep = batched_nms(boxes, scores, labels, iou_thr, max_det)
    return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}
