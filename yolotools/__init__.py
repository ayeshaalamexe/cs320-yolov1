"""CS-320 YOLOv1 project - Hassaan (Data lead) toolkit.

Modules
-------
boxes         box format conversions and IoU
penn_fudan    Penn-Fudan download, PASCAL->YOLO conversion, splits, dataset writer
quality       the 10 dataset quality checks + report
viz           dataset figures (sample grid, mask instances, label check, stats)
augment       box-aware augmentation (HSV, flip, scale+translate)
yolo_dataset  loader for the YOLO-format Penn-Fudan folder
voc2007       PASCAL VOC 2007 download + loader + YOLOv1 7x7xK target encoder
nms           NMS, class-aware NMS, YOLOv1 output decoding
metrics       VOC-style AP / mAP@IoU (07 11-point and all-point), P/R at threshold
evaluate      glue: run a YOLOv1-style model over a dataset and compute mAP
ablation      Ultralytics augmentation on/off ablation helpers
"""
__version__ = "1.0.0"
