# cs320-yolov1
CS-320 ANN and deep learning: YOLOv1 — reproduce, extend, ablate (pedestrian detection: original YOLO vs modern YOLO).

## Team and Contribution Log
- **Ayesha** — Paper lead: study and reproduce the original YOLOv1 baseline; ablation: baseline vs modern YOLO
- **Hassaan Adil**: Data Lead: dataset loading, annotation conversion, dataset splits, quality checks, statistics, and evaluation-data preparation; also VOC 2007 / NMS / mAP and augmentation ablation
- **Sara** — Experiments lead: result tables and plots, inference demo, report; ablation: confidence thresholds and input sizes

## Hassaan's components
| Component | Where |
|---|---|
| Penn-Fudan → YOLO dataset, split manifest, quality checks, stats | `notebooks/Hassaan_Adil_Data_Lead.ipynb`, `yolotools/penn_fudan.py`, `yolotools/quality.py`, `PennFudanYOLO/` |
| VOC 2007 loader (+ YOLOv1 7×7×25 target encoder, augmentation) | `yolotools/voc2007.py`, `yolotools/augment.py` |
| NMS + YOLOv1 output decoding | `yolotools/nms.py` |
| mAP@0.5 (VOC07 11-pt / all-point), P/R at threshold | `yolotools/metrics.py`, `yolotools/evaluate.py` |
| Notebook for VOC / NMS / mAP | `notebooks/Hassaan_Adil_Data_Lead.ipynb` |
| Augmentation ablation (Ultralytics, 3 seeds) | `notebooks/Hassaan_Adil_Data_Lead.ipynb`, `yolotools/ablation.py`, `results/` |
| Write-up | `docs/Hassaan_Data_Lead_Report.md` |

## Quick start (Colab)
1. Open a notebook from `notebooks/` in Colab and run all cells (they write the `yolotools` package themselves).
2. Dataset for training: `PennFudanYOLO/data.yaml` (`path:` is rewritten by the notebooks).
3. Tests (local or Colab): `pip install pytest pillow numpy pandas matplotlib && python -m pytest tests`

## Using the tools in your own notebook
```python
from yolotools.voc2007 import VOC2007Detection, download_voc2007
from yolotools.evaluate import evaluate_yolov1
devkit = download_voc2007('/content/data')
train = VOC2007Detection(devkit, 'trainval', 448, augment=True)      # (image 3x448x448, target 7x7x25)
test  = VOC2007Detection(devkit, 'test', 448)                        # augment=False for evaluation
res, preds, gts = evaluate_yolov1(model, test, B=2, C=20, device='cuda', score_thr=0.01, nms_iou=0.5)
```
Dataset notes: `PennFudanYOLO/README_dataset.md`. Do **not** tune hyper-parameters on the test split.

