# Augmentation ablation (RQ2) - Penn-Fudan, yolo11n.pt, 50 epochs, imgsz 640, batch 16, seeds [0, 1, 2]

All variants: same pretrained weights, data split (119/25/26 images), epochs, image size, batch size and seeds; only augmentation
arguments differ. Best checkpoint chosen on val; test evaluated once per run (conf=0.001, NMS IoU=0.6).

| Variant | n | test_map50 | test_map50_95 | test_precision | test_recall |
|---|---|---|---|---|---|
| default_aug | 3 | 0.974 ± 0.003 | 0.775 ± 0.012 | 0.955 ± 0.012 | 0.894 ± 0.015 |
| no_aug | 3 | 0.926 ± 0.012 | 0.631 ± 0.038 | 0.937 ± 0.012 | 0.840 ± 0.008 |

mean ± sample sd over 3 seeds. On test mAP@0.5, default augmentation scores 0.974 versus 0.926 without
augmentation (difference +0.048), which is larger than the seed-to-seed standard deviation (0.012).
Per-run numbers: aug_ablation_runs CSV. Figures: aug_ablation_bars.png, aug_ablation_curves.png, aug_ablation_predictions.png.

Interpretation notes (check them against the figures before writing the report):
* A COCO-pretrained detector already knows 'person', so Penn-Fudan can sit close to a performance ceiling and hide augmentation effects; the from-scratch run (MODEL = 'yolo11n.yaml') is the stricter test of RQ2.
* Compare val-loss curves: if no_aug's validation loss rises while training loss falls, that is over-fitting that augmentation is regularising.
* The test split is tiny (26 images); report seeds, not a single run.
