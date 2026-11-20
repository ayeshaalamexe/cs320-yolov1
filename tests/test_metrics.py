import numpy as np
import pytest
from yolotools.metrics import (voc_ap, evaluate_detections, precision_recall_at_threshold,
                               map_over_iou_thresholds)


def P(boxes, scores, labels):
    return {"boxes": np.array(boxes, float).reshape(-1, 4), "scores": np.array(scores, float),
            "labels": np.array(labels)}


def G(boxes, labels, difficult=None):
    d = {"boxes": np.array(boxes, float).reshape(-1, 4), "labels": np.array(labels)}
    if difficult is not None:
        d["difficult"] = np.array(difficult, bool)
    return d


def test_perfect():
    gts = [G([[0, 0, 10, 10], [20, 20, 30, 30]], [0, 1]), G([[5, 5, 15, 15]], [0])]
    preds = [P([[0, 0, 10, 10], [20, 20, 30, 30]], [.9, .8], [0, 1]), P([[5, 5, 15, 15]], [.7], [0])]
    for use07 in (True, False):
        r = evaluate_detections(preds, gts, 2, 0.5, use07)
        assert r["mAP"] == pytest.approx(1.0)


def test_no_predictions_and_no_gt_class():
    gts = [G([[0, 0, 10, 10]], [0])]
    r = evaluate_detections([P([], [], [])], gts, 2)
    assert r["ap"][0] == 0.0 and np.isnan(r["ap"][1]) and r["mAP"] == 0.0   # class 1 has no GT: excluded


def test_hand_computed_ap():
    # 3 GT of class 0 in one image. Detections sorted by score: TP, FP, TP  -> rec .333,.333,.667 prec 1,.5,.667
    gts = [G([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]], [0, 0, 0])]
    preds = [P([[0, 0, 10, 10], [100, 100, 110, 110], [20, 0, 30, 10]], [.9, .8, .7], [0, 0, 0])]
    r_all = evaluate_detections(preds, gts, 1, 0.5, use_07_metric=False)
    # envelope: prec = [1, .667, .667] ; area = (1/3)*1 + (1/3)*(2/3) = 0.5556
    assert r_all["mAP"] == pytest.approx(1 / 3 + (1 / 3) * (2 / 3))
    r07 = evaluate_detections(preds, gts, 1, 0.5, use_07_metric=True)
    # recall thresholds 0..0.6 (7 pts) have max prec 1 for t<=.3 (4 pts: 0,.1,.2,.3), .667 for .4-.6 (3 pts), 0 after
    assert r07["mAP"] == pytest.approx((4 * 1.0 + 3 * (2 / 3)) / 11)


def test_duplicate_is_false_positive():
    gts = [G([[0, 0, 10, 10]], [0])]
    preds = [P([[0, 0, 10, 10], [0, 0, 10, 10]], [.9, .8], [0, 0])]
    r = evaluate_detections(preds, gts, 1, 0.5, False)
    assert r["precision"][0] == pytest.approx(0.5) and r["recall"][0] == pytest.approx(1.0)


def test_difficult_ignored():
    gts = [G([[0, 0, 10, 10], [20, 20, 30, 30]], [0, 0], difficult=[False, True])]
    # detecting the difficult one is neither TP nor FP; missing it costs no recall
    preds = [P([[0, 0, 10, 10], [20, 20, 30, 30]], [.9, .8], [0, 0])]
    r = evaluate_detections(preds, gts, 1, 0.5, False)
    assert r["npos"][0] == 1 and r["mAP"] == pytest.approx(1.0)
    preds2 = [P([[0, 0, 10, 10]], [.9], [0])]
    assert evaluate_detections(preds2, gts, 1, 0.5, False)["mAP"] == pytest.approx(1.0)


def test_wrong_class_and_wrong_image():
    gts = [G([[0, 0, 10, 10]], [0]), G([[0, 0, 10, 10]], [0])]
    preds = [P([[0, 0, 10, 10]], [.9], [1]), P([], [], [])]
    assert evaluate_detections(preds, gts, 2, 0.5)["ap"][0] == 0.0


def test_iou_threshold_boundary():
    gts = [G([[0, 0, 10, 10]], [0])]
    preds = [P([[0, 0, 10, 5]], [.9], [0])]            # IoU exactly 0.5 -> counts (>=)
    assert evaluate_detections(preds, gts, 1, 0.5, False)["mAP"] == pytest.approx(1.0)
    assert evaluate_detections(preds, gts, 1, 0.51, False)["mAP"] == 0.0


def test_normalised_equals_pixel_boxes():
    rng = np.random.default_rng(0)
    gts, preds = [], []
    for _ in range(5):
        xy = rng.uniform(0, 300, (4, 2)); wh = rng.uniform(20, 100, (4, 2))
        gb = np.concatenate([xy, xy + wh], 1)
        pb = gb + rng.normal(0, 6, gb.shape)
        gts.append(G(gb, rng.integers(0, 3, 4))); preds.append(P(pb, rng.uniform(0, 1, 4), rng.integers(0, 3, 4)))
    sx, sy = 1 / 640, 1 / 480
    scale = np.array([sx, sy, sx, sy])
    gts_n = [{**g, "boxes": g["boxes"] * scale} for g in gts]
    preds_n = [{**p, "boxes": p["boxes"] * scale} for p in preds]
    a = evaluate_detections(preds, gts, 3)["mAP"]
    b = evaluate_detections(preds_n, gts_n, 3)["mAP"]
    assert a == pytest.approx(b)


# ---- independent reference: classic py-faster-rcnn style voc_eval ---------------------------------
def ref_voc_eval(preds, gts, cls, ovthresh, use07):
    R = {i: dict(bbox=g["boxes"][g["labels"] == cls],
                 difficult=np.asarray(g.get("difficult", np.zeros(len(g["boxes"]), bool)))[g["labels"] == cls]
                 if len(g["boxes"]) else np.zeros(0, bool), det=None) for i, g in enumerate(gts)}
    npos = sum(int((~r["difficult"]).sum()) for r in R.values())
    for r in R.values():
        r["det"] = [False] * len(r["bbox"])
    ids, conf, BB = [], [], []
    for i, p in enumerate(preds):
        m = p["labels"] == cls
        for b, s in zip(p["boxes"][m], p["scores"][m]):
            ids.append(i); conf.append(s); BB.append(b)
    if not ids:
        return 0.0
    conf, BB = np.array(conf), np.array(BB)
    order = np.argsort(-conf, kind="stable")
    BB = BB[order]; ids = [ids[k] for k in order]
    nd = len(ids); tp = np.zeros(nd); fp = np.zeros(nd)
    for d in range(nd):
        r = R[ids[d]]; bb = BB[d]; ovmax = -np.inf; BBGT = r["bbox"]
        if len(BBGT):
            ixmin = np.maximum(BBGT[:, 0], bb[0]); iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2]); iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin, 0.); ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih
            uni = (bb[2] - bb[0]) * (bb[3] - bb[1]) + (BBGT[:, 2] - BBGT[:, 0]) * (BBGT[:, 3] - BBGT[:, 1]) - inters
            ov = inters / uni; ovmax = np.max(ov); j = np.argmax(ov)
        if ovmax >= ovthresh:
            if not r["difficult"][j]:
                if not r["det"][j]:
                    tp[d] = 1.; r["det"][j] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.
    fp = np.cumsum(fp); tp = np.cumsum(tp)
    rec = tp / float(npos); prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    if use07:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            p = 0 if np.sum(rec >= t) == 0 else np.max(prec[rec >= t])
            ap += p / 11.
        return ap
    mrec = np.concatenate(([0.], rec, [1.])); mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])


@pytest.mark.parametrize("seed", range(15))
@pytest.mark.parametrize("use07", [True, False])
def test_fuzz_against_reference(seed, use07):
    rng = np.random.default_rng(seed)
    C = 4
    gts, preds = [], []
    for _ in range(rng.integers(3, 12)):
        m = rng.integers(0, 6)
        xy = rng.uniform(0, 200, (m, 2)); wh = rng.uniform(15, 80, (m, 2))
        gb = np.concatenate([xy, xy + wh], 1)
        gl = rng.integers(0, C, m)
        gts.append(G(gb, gl, rng.random(m) < 0.2))
        k = rng.integers(0, 10)
        pick = rng.integers(0, max(m, 1), k)
        pb = (gb[pick] if m else np.zeros((k, 4))) + rng.normal(0, 8, (k, 4))
        pb[:, 2:] = np.maximum(pb[:, 2:], pb[:, :2] + 1)
        pl = np.where(rng.random(k) < 0.8, gl[pick] if m else 0, rng.integers(0, C, k))
        preds.append(P(pb, np.round(rng.uniform(0, 1, k), 3), pl))
    mine = evaluate_detections(preds, gts, C, 0.5, use07)
    for c in range(C):
        npos = sum(int(((g["labels"] == c) & ~g.get("difficult", np.zeros(len(g["boxes"]), bool))).sum()) for g in gts)
        if npos == 0:
            continue
        assert mine["ap"][c] == pytest.approx(ref_voc_eval(preds, gts, c, 0.5, use07), abs=1e-9)


def test_precision_recall_threshold_prefix_property():
    gts = [G([[0, 0, 10, 10], [20, 20, 30, 30]], [0, 0])]
    preds = [P([[0, 0, 10, 10], [50, 50, 60, 60], [20, 20, 30, 30]], [.9, .6, .3], [0, 0, 0])]
    r = precision_recall_at_threshold(preds, gts, 1, 0.5)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1)
    r = precision_recall_at_threshold(preds, gts, 1, 0.1)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 1, 0) and r["recall"] == 1.0
    r = precision_recall_at_threshold(preds, gts, 1, 0.95)
    assert r["tp"] == 0 and r["fn"] == 2 and r["precision"] == 0.0


def test_map_over_thresholds_monotone():
    gts = [G([[0, 0, 100, 100]], [0])]
    preds = [P([[3, 3, 100, 100]], [.9], [0])]       # IoU ~0.94
    out = map_over_iou_thresholds(preds, gts, 1)
    assert out[0.5] == 1.0 and out[0.95] == 0.0 and 0 < out["mean"] < 1
    vals = [out[k] for k in sorted(k for k in out if k != "mean")]
    assert vals == sorted(vals, reverse=True)
