# Augmentation ablation (RQ2) - yolo11n, 50 epochs, imgsz 640, seeds [0, 1, 2]

* test_map50: no_aug 0.916, default_aug 0.974, difference +0.058 (bigger than the seed-to-seed noise -> real effect)
* test_map50_95: no_aug 0.635, default_aug 0.775, difference +0.140 (bigger than the seed-to-seed noise -> real effect)
* test_recall: no_aug 0.829, default_aug 0.894, difference +0.065 (bigger than the seed-to-seed noise -> real effect)

Mean +- std over seeds:

| variant     | test_map50     | test_map50_95   | test_precision   | test_recall    | val_map50      | val_map50_95   |
|:------------|:---------------|:----------------|:-----------------|:---------------|:---------------|:---------------|
| default_aug | 0.974 +- 0.003 | 0.775 +- 0.012  | 0.955 +- 0.012   | 0.894 +- 0.015 | 0.977 +- 0.013 | 0.801 +- 0.029 |
| no_aug      | 0.916 +- 0.015 | 0.635 +- 0.007  | 0.912 +- 0.032   | 0.829 +- 0.045 | 0.942 +- 0.026 | 0.666 +- 0.018 |

Note: yolo11n is COCO-pretrained and already knows `person`, and the test set has only 26 images, so the effect size should be read together with the seed standard deviation.
