# Data lead report section — Hassaan Adil (524115)
Project: *Pedestrian Detection: Original YOLO vs. Modern YOLO* (CS-320). Role per proposal: dataset loading, YOLO-format conversion, splits, quality checks; ablation: with vs without data augmentation. This week's extra items: VOC 2007 loader, NMS, mAP@0.5.

## 1. Penn-Fudan dataset preparation
**Source.** Official Penn-Fudan Pedestrian zip: 170 images (74 `FudanPed*`, 96 `PennPed*`), 423 annotated pedestrians, with PASCAL-style annotation text files and per-instance masks.

**Conversion.** Annotation boxes are 1-indexed inclusive pixels `(xmin,ymin,xmax,ymax)`. They are converted to continuous edges `(xmin-1, ymin-1, xmax, ymax)` and then to YOLO `0 xc yc w h`, normalised by each image's own width/height (class 0 = pedestrian). Images stay at original resolution; resizing belongs to the training code. Each annotation box is cross-checked against the box derived from the instance mask; count differences and low IoU (<0.70) are written to `conversion_log.txt`. Integrity checks: unreadable images, missing/orphan annotations and masks, annotation-vs-image size mismatch, degenerate or out-of-image boxes.

**Split.** Image-level 70/15/15 with seed 42 → 119 / 25 / 26 images (295 / 55 / 73 pedestrians). The manifest `split_manifest.csv` (filename, split, source set, size, #pedestrians, label source) is frozen and reused by every notebook so that all team results use identical splits. Limitation: an image-level split can place visually similar frames from one scene in different splits, and the test set is small (26 images), so metrics on it have high variance — hence multiple seeds in the ablation.

**Quality checks (10, automated, `quality_report.txt`).** (1) every image has a label file; (2) rows have 5 numeric fields; (3) class ids ∈ {0}; (4) normalised values in [0,1] and box inside the image; (5) positive width/height; (6) no image in more than one split; (7) #images = #labels per split; (8) per-split image/instance totals agree with the manifest and the published totals; (9) `data.yaml` directories exist and `nc=1`; (10) a random sample of saved labels, converted back to pixels, reproduces the original annotation boxes. The checks themselves are unit-tested: corrupted datasets (bad class id, box outside image, missing label, image in two splits) are correctly reported as FAIL.

**Deliverables.** `images/` + `labels/` × {train,val,test}, `data.yaml`, `split_manifest.csv`, `dataset_stats.csv`, `quality_report.txt`, `conversion_log.txt`, `README_dataset.md`, figures (`sample_grid.png`, `mask_instances.png`, `label_check_samples.png`, `dataset_stats_plots.png`).

## 2. Loaders
* `YoloFolderDataset` — reads the YOLO folder; returns `(3×448×448 image, [xc,yc,w,h] boxes)` compatible with the YOLOv1 code; optional box-aware augmentation.
* `VOC2007Detection` — VOC2007 trainval (5011) / test (4952); image `3×448×448` and target `7×7×25` (`[x,y,w,h,obj,20 classes]`, same layout as the YOLOv1 loss); `difficult` objects excluded for training and flagged for evaluation; at most one object per cell (YOLOv1 limitation; the share of objects lost is measured in notebook B).
* Augmentation (`yolotools/augment.py`), as in the YOLOv1 paper: HSV exposure/saturation jitter up to ×1.5, random scale/translation up to 20 %, horizontal flip; boxes are transformed with the image and dropped if <30 % remains visible. Geometry is verified against pixel positions in unit tests.

## 3. Non-Maximum Suppression
Greedy NMS: process boxes by descending score, keep a box, suppress remaining boxes with IoU **strictly greater** than the threshold; class-aware variant never suppresses across classes. YOLOv1 decoding: per cell and per box, score = confidence × class probability, threshold, then class-aware NMS and top-k. Implementation verified against a naive reference (fuzz tests with tied scores) and against `torchvision.ops.nms/batched_nms` (200 random problems, identical kept indices; notebook B).

## 4. mAP@0.5
Follows the VOC devkit protocol (greedy matching in descending score, duplicate = false positive, `difficult` ignored, recall over non-difficult GTs). Reports the VOC2007 11-point AP (the metric comparable with the YOLO paper) and the all-point AP; also mAP over IoU 0.50:0.95, per-class AP, PR curves, and precision/recall/F1 at a score threshold (useful for Sara's confidence-threshold study). Verified by hand-computed cases and a randomised comparison against an independent devkit-style implementation (both metrics, with difficult objects). End-to-end sanity on the real VOC test set with synthetic detections: GT-as-detections → 100 %; noisy detections lower; NMS increases mAP.

## 5. Ablation: data augmentation (RQ2)
Ultralytics YOLO, COCO-pretrained weights fine-tuned on Penn-Fudan. Variants: `no_aug` (all stochastic augmentations zero: HSV, flips, scale, translate, shear, rotation, perspective, mosaic, mixup, copy-paste, erasing) vs `default_aug`. Identical model, data, epochs, image size, batch size and seeds (0,1,2); best checkpoint selected on val; test evaluated once per run; mean ± sd reported (precision, recall, mAP@0.5, mAP@0.5:0.95, inference speed). Optional one-at-a-time variants (flip, HSV, geometric, mosaic) and an optional from-scratch run, which is the stricter test because a COCO-pretrained model already knows "person".

**Results.** `<!-- paste results/aug_ablation_report.md here after running notebook C -->`

**Discussion guidance.** State the mean difference and compare it with the seed-to-seed sd; mention the tiny test set; relate any gap to the validation-loss curves (over-fitting without augmentation); say explicitly if the pretrained model is at a performance ceiling.

## 6. Reproducibility
Seeds fixed (split seed 42; training seeds 0/1/2, `deterministic=True`); all code in `yolotools/`, 97 automated tests (`python -m pytest tests`); notebooks A–C regenerate every artefact in Colab.
