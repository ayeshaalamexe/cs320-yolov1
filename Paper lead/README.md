[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ayeshaalamexe/cs320-yolov1/blob/main/Paper%20lead/paper_lead__code_file.ipynb)

# YOLOv1: Penn-Fudan and VOC07

| Dataset | Setting | Val mAP@0.5 | Test mAP@0.5 |
| Penn-Fudan | lr=1e-4, no aug | 0.164 | 0.114 |
| Penn-Fudan | lr=3e-5, no aug | 0.333 | 0.290 |
| Penn-Fudan | lr=3e-5, aug | 0.360 | 0.324 |
| VOC07 | 40 epochs | - | 0.333 (paper 0.634) |

Speed: ~6 ms/image (~150 FPS). Parameters: 243,179,867.
Single seed per setting.

Note: select Runtime -> Change runtime type -> T4 GPU in Colab before running.

Checkpoints: (Drive link yahan)