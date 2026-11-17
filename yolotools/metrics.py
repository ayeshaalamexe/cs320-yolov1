"""Detection metrics: PASCAL-VOC style AP / mAP at a given IoU (default 0.5).

Matching follows the official VOC devkit
  * detections of a class are processed in descending score order (across all images);
  * each detection is compared with the ground truths of ITS image and class and takes
    the one with the highest IoU; if IoU >= iou_thr:
        - that GT is `difficult`           -> detection ignored (neither TP nor FP)
        - GT not yet matched               -> TP, GT is marked as matched
        - GT already matched               -> FP (duplicate detection)
    otherwise FP;
  * recall's denominator is the number of NON-difficult GTs.
AP is the VOC2007 11-point metric (use_07_metric=True, the official metric for VOC2007)
or the all-point area under the monotone precision envelope (VOC2010+ / use_07_metric=False).

Inputs
------
preds[i] = {"boxes": (N,4) xyxy, "scores": (N,), "labels": (N,)}
gts[i]   = {"boxes": (M,4) xyxy, "labels": (M,), "difficult": (M,) bool (optional)}
Boxes may be pixels or normalised coordinates, as long as preds and gts agree.
"""
import numpy as np
from .boxes import iou_one_to_many


def voc_ap(rec, prec, use_07_metric=True):
    rec, prec = np.asarray(rec, float), np.asarray(prec, float)
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            p = prec[rec >= t].max() if np.any(rec >= t) else 0.0
            ap += p / 11.0
        return float(ap)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def match_detections(preds, gts, num_classes, iou_thr=0.5):
    """Greedy VOC matching. Returns (records, npos).

    records: dict of arrays sorted by descending score -
        img, cls, score, status   (status: 1=TP, 0=FP, -1=ignored/difficult)
    npos: (num_classes,) count of non-difficult GTs.
    """
    gt_store, npos = {}, np.zeros(num_classes, dtype=np.int64)
    for i, g in enumerate(gts):
        boxes = np.asarray(g["boxes"], float).reshape(-1, 4)
        labels = np.asarray(g["labels"]).reshape(-1).astype(int)
        diff = np.asarray(g.get("difficult", np.zeros(len(boxes), bool))).reshape(-1).astype(bool)
        for c in np.unique(labels):
            m = labels == c
            gt_store[(i, int(c))] = [boxes[m], diff[m], np.zeros(int(m.sum()), bool)]
            npos[int(c)] += int((~diff[m]).sum())

    img, cls, score, box = [], [], [], []
    for i, p in enumerate(preds):
        b = np.asarray(p["boxes"], float).reshape(-1, 4)
        img.append(np.full(len(b), i)); box.append(b)
        cls.append(np.asarray(p["labels"]).reshape(-1).astype(int))
        score.append(np.asarray(p["scores"], float).reshape(-1))
    if not img:
        z = np.zeros(0)
        return {"img": z.astype(int), "cls": z.astype(int), "score": z, "status": z.astype(int)}, npos
    img, cls, score = np.concatenate(img), np.concatenate(cls), np.concatenate(score)
    box = np.concatenate(box)
    order = np.argsort(-score, kind="stable")
    img, cls, score, box = img[order], cls[order], score[order], box[order]

    status = np.zeros(len(score), dtype=np.int64)
    for k in range(len(score)):
        entry = gt_store.get((int(img[k]), int(cls[k])))
        if entry is None or len(entry[0]) == 0:
            continue                                   # FP
        gboxes, gdiff, gdone = entry
        ious = iou_one_to_many(box[k], gboxes)
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr:
            if gdiff[j]:
                status[k] = -1
            elif not gdone[j]:
                status[k] = 1
                gdone[j] = True
            # else: duplicate of an already matched GT -> stays 0 (FP)
    return {"img": img, "cls": cls, "score": score, "status": status}, npos


def _ap_for_class(status, npos_c, use_07_metric):
    """status: statuses of one class's detections in descending-score order."""
    if npos_c == 0:
        return np.nan, np.zeros(0), np.zeros(0)
    tp = np.cumsum(status == 1)           # ignored (-1) entries add to neither tp nor fp
    fp = np.cumsum(status == 0)
    rec = tp / float(npos_c)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    return voc_ap(rec, prec, use_07_metric), rec, prec


def evaluate_detections(preds, gts, num_classes, iou_thr=0.5, use_07_metric=True, class_names=None):
    """Per-class AP and mAP at one IoU threshold."""
    recs, npos = match_detections(preds, gts, num_classes, iou_thr)
    ap = np.full(num_classes, np.nan)
    curves, final_p, final_r = [], np.zeros(num_classes), np.zeros(num_classes)
    for c in range(num_classes):
        st = recs["status"][recs["cls"] == c]
        a, r, p = _ap_for_class(st, int(npos[c]), use_07_metric)
        ap[c] = a
        curves.append((r, p))
        if len(r):
            final_r[c], final_p[c] = r[-1], p[-1]
    valid = ~np.isnan(ap)
    return {"mAP": float(ap[valid].mean()) if valid.any() else 0.0, "ap": ap, "npos": npos,
            "curves": curves, "precision": final_p, "recall": final_r, "iou_thr": iou_thr,
            "metric": "VOC07-11pt" if use_07_metric else "all-point",
            "class_names": list(class_names) if class_names is not None else [str(i) for i in range(num_classes)]}


def map_over_iou_thresholds(preds, gts, num_classes, thresholds=None, use_07_metric=False):
    """mAP at several IoU thresholds (default 0.50:0.05:0.95) -> dict(thr -> mAP) + 'mean'."""
    thresholds = np.round(np.arange(0.5, 0.96, 0.05), 2) if thresholds is None else thresholds
    out = {float(t): evaluate_detections(preds, gts, num_classes, float(t), use_07_metric)["mAP"]
           for t in thresholds}
    out["mean"] = float(np.mean([v for k, v in out.items()]))
    return out


def precision_recall_at_threshold(preds, gts, num_classes, score_thr, iou_thr=0.5):
    """Micro-averaged TP/FP/FN, precision, recall, F1 using only detections with score >= score_thr.

    Valid because matching is greedy in descending score: dropping the low-score tail does not
    change how higher-scored detections are matched.
    """
    recs, npos = match_detections(preds, gts, num_classes, iou_thr)
    st = recs["status"][recs["score"] >= score_thr]
    tp, fp = int((st == 1).sum()), int((st == 0).sum())
    fn = int(npos.sum()) - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1, "score_thr": score_thr}


def sweep_score_thresholds(preds, gts, num_classes, thresholds, iou_thr=0.5):
    """Convenience for confidence-threshold studies: list of precision_recall_at_threshold."""
    return [precision_recall_at_threshold(preds, gts, num_classes, t, iou_thr) for t in thresholds]


def format_ap_table(result):
    lines = [f"{'class':<14}{'AP':>8}{'GT':>7}"]
    for n, a, k in zip(result["class_names"], result["ap"], result["npos"]):
        lines.append(f"{n:<14}{a * 100:>7.2f}%{int(k):>7}")
    lines.append(f"{'mAP@' + str(result['iou_thr']):<14}{result['mAP'] * 100:>7.2f}%   ({result['metric']})")
    return "\n".join(lines)
