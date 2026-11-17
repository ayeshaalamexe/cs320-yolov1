# PennFudanYOLO (Member 1 handover)
- Source: official Penn-Fudan zip (https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip)
- Labels: YOLO detection format `0 xc yc w h` (normalised by each image's own size), class 0 = pedestrian
- Boxes: official PASCAL annotation boxes (1-indexed inclusive -> converted), cross-checked against mask instance boxes
- Split: image level 70/15/15, seed 42, see split_manifest.csv
- Images kept at original resolution (resizing is done by the training code; normalised labels are unaffected)
- Do NOT tune hyper-parameters on the test split
- Update `path:` in data.yaml to wherever this folder lives on your machine/Colab
