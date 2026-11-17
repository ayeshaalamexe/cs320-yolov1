"""Glue: run a YOLOv1-style torch model over a dataset -> detections -> mAP.

Works with VOC2007Detection and YoloFolderDataset (both expose image_tensor(i) / ground_truth(i)).
The model must map (N,3,H,W) -> (N,S,S,B*5+C) like the team's YOLOv1 class.
"""
import time
import numpy as np
from .nms import postprocess_yolov1
from .metrics import evaluate_detections, format_ap_table


def predict_dataset(model, dataset, B=2, C=20, device="cuda", batch_size=16, score_thr=0.01,
                    nms_iou=0.5, max_det=100):
    import torch
    model.eval()
    preds, t_model, n = [], 0.0, len(dataset)
    with torch.no_grad():
        for s in range(0, n, batch_size):
            x = torch.stack([dataset.image_tensor(i) for i in range(s, min(n, s + batch_size))]).to(device)
            if device != "cpu":
                torch.cuda.synchronize()
            t0 = time.time()
            out = model(x).float().cpu().numpy()
            if device != "cpu":
                torch.cuda.synchronize()
            t_model += time.time() - t0
            preds += [postprocess_yolov1(o, B, C, score_thr, nms_iou, max_det) for o in out]
    return preds, {"model_ms_per_image": 1000 * t_model / max(n, 1)}


def evaluate_yolov1(model, dataset, B=2, C=20, device="cuda", batch_size=16, score_thr=0.01,
                    nms_iou=0.5, max_det=100, iou_thr=0.5, use_07_metric=True, class_names=None, verbose=True):
    preds, timing = predict_dataset(model, dataset, B, C, device, batch_size, score_thr, nms_iou, max_det)
    gts = [dataset.ground_truth(i) for i in range(len(dataset))]
    res = evaluate_detections(preds, gts, C, iou_thr, use_07_metric, class_names or getattr(dataset, "classes", None))
    res.update(timing)
    if verbose:
        print(format_ap_table(res))
        print(f"model forward: {timing['model_ms_per_image']:.1f} ms/image")
    return res, preds, gts
