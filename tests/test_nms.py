import numpy as np
import pytest
from yolotools.nms import nms, batched_nms, decode_yolov1, postprocess_yolov1


def naive_nms(boxes, scores, thr):
    """Independent O(n^2) reference using scalar IoU maths."""
    def iou(a, b):
        iw = min(a[2], b[2]) - max(a[0], b[0]); ih = min(a[3], b[3]) - max(a[1], b[1])
        if iw <= 0 or ih <= 0:
            return 0.0
        inter = iw * ih
        return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
    idx = sorted(range(len(boxes)), key=lambda i: (-scores[i], i))
    keep = []
    for i in idx:
        if all(iou(boxes[i], boxes[k]) <= thr for k in keep):
            keep.append(i)
    return keep


def test_empty():
    assert len(nms(np.zeros((0, 4)), np.zeros(0), 0.5)) == 0
    assert len(batched_nms(np.zeros((0, 4)), np.zeros(0), np.zeros(0), 0.5)) == 0


def test_hand_case():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], float)
    scores = np.array([0.9, 0.8, 0.7])
    # IoU(0,1) = 81/119 = 0.68
    assert nms(boxes, scores, 0.5).tolist() == [0, 2]
    assert nms(boxes, scores, 0.7).tolist() == [0, 1, 2]     # 0.68 <= 0.7 -> both kept
    assert nms(boxes, scores, 0.0).tolist() == [0, 2]


def test_order_is_by_score_not_index():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], float)
    assert nms(boxes, np.array([0.2, 0.9]), 0.5).tolist() == [1]


def test_iou_equal_to_threshold_is_kept():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 5]], float)   # IoU exactly 0.5
    assert nms(boxes, np.array([0.9, 0.8]), 0.5).tolist() == [0, 1]


def test_max_keep():
    boxes = np.array([[0, 0, 1, 1], [5, 5, 6, 6], [9, 9, 10, 10]], float)
    assert len(nms(boxes, np.array([.3, .2, .1]), 0.5, max_keep=2)) == 2


def test_class_aware():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], float)
    scores = np.array([0.9, 0.8])
    assert nms(boxes, scores, 0.5).tolist() == [0]
    assert batched_nms(boxes, scores, np.array([0, 1]), 0.5).tolist() == [0, 1]
    assert batched_nms(boxes, scores, np.array([1, 1]), 0.5).tolist() == [0]


def test_bad_args():
    with pytest.raises(ValueError):
        nms(np.zeros((2, 4)), np.zeros(3), 0.5)
    with pytest.raises(ValueError):
        nms(np.zeros((2, 4)), np.zeros(2), 1.5)


@pytest.mark.parametrize("seed", range(20))
def test_fuzz_against_naive(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(1, 80)
    xy = rng.uniform(0, 100, (n, 2)); wh = rng.uniform(1, 40, (n, 2))
    boxes = np.concatenate([xy, xy + wh], 1)
    scores = np.round(rng.uniform(0, 1, n), 2)            # rounding -> many score ties
    thr = float(rng.choice([0.1, 0.3, 0.5, 0.7]))
    assert nms(boxes, scores, thr).tolist() == naive_nms(boxes, scores, thr)


def test_decode_yolov1_roundtrip():
    S, B, C = 7, 2, 20
    pred = np.zeros((S, S, B * 5 + C))
    # object in row 3, col 2 : cell offsets .5,.5 ; w=.2 h=.4 ; conf .9 ; class 14 prob 1
    pred[3, 2, 0:5] = [0.5, 0.5, 0.2, 0.4, 0.9]
    pred[3, 2, 10 + 14] = 1.0
    boxes, scores, labels = decode_yolov1(pred, B, C, score_thr=0.5)
    assert len(boxes) == 1 and labels[0] == 14 and np.isclose(scores[0], 0.9)
    xc, yc = (2 + .5) / 7, (3 + .5) / 7
    assert np.allclose(boxes[0], [xc - .1, yc - .2, xc + .1, yc + .2])


def test_postprocess_suppresses_duplicate_boxes_of_same_cell():
    S, B, C = 7, 2, 20
    pred = np.zeros((S, S, B * 5 + C))
    pred[3, 3, 0:5] = [0.5, 0.5, 0.3, 0.3, 0.9]
    pred[3, 3, 5:10] = [0.52, 0.5, 0.3, 0.3, 0.8]      # near-identical 2nd box
    pred[3, 3, 10 + 6] = 1.0
    out = postprocess_yolov1(pred, B, C, score_thr=0.1, iou_thr=0.5)
    assert len(out["boxes"]) == 1 and np.isclose(out["scores"][0], 0.9)
    out2 = postprocess_yolov1(pred, B, C, score_thr=0.1, iou_thr=1.0)  # nothing suppressed
    assert len(out2["boxes"]) == 2


def test_decode_shape_error():
    with pytest.raises(ValueError):
        decode_yolov1(np.zeros((7, 7, 25)), B=2, C=20)
