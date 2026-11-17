import os
import numpy as np
import pytest
from PIL import Image

ANN = '''# Compatible with PASCAL Annotation Version 1.00
Image filename : "PennFudanPed/PNGImages/{stem}.png"
Image size (X x Y x C) : {W} x {H} x 3
Database : "The Penn-Fudan-Pedestrian Database"
Objects with ground truth : {n} {{ {names} }}
# Top left pixel co-ordinates : (1, 1)
{objs}'''
OBJ = '''# Details for object {k} ("PASpersonWalking")
# Center point -- not available in this database
Original label for object {k} "PASpersonWalking" : "PennFudanPed"
Bounding box for object {k} "PASpersonWalking" (Xmin, Ymin) - (Xmax, Ymax) : ({x1}, {y1}) - ({x2}, {y2})
Pixel mask for object {k} "PASpersonWalking" : "PennFudanPed/PedMasks/{stem}_mask.png"
'''


@pytest.fixture(scope="session")
def fake_pf(tmp_path_factory):
    """Synthetic Penn-Fudan: 40 images, 1-4 rectangles each, official annotation text format."""
    root = tmp_path_factory.mktemp("pf") / "PennFudanPed"
    for d in ("PNGImages", "Annotation", "PedMasks"):
        os.makedirs(root / d)
    rng = np.random.default_rng(1)
    truth = {}
    for n in range(40):
        stem = ("FudanPed%05d" if n % 2 == 0 else "PennPed%05d") % (n + 1)
        W, H = int(rng.integers(200, 320)), int(rng.integers(200, 320))
        img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
        mask = np.zeros((H, W), np.uint8)
        k = int(rng.integers(1, 5)); objs = ""; boxes = []
        for j in range(1, k + 1):
            x1 = int(rng.integers(1, W - 80)); y1 = int(rng.integers(1, H - 100))
            x2 = x1 + int(rng.integers(30, 70)); y2 = y1 + int(rng.integers(50, 95))
            mask[y1 - 1:y2, x1 - 1:x2] = j                  # 1-indexed inclusive box -> mask pixels
            objs += OBJ.format(k=j, x1=x1, y1=y1, x2=x2, y2=y2, stem=stem)
            boxes.append((x1, y1, x2, y2))
        Image.fromarray(img).save(root / "PNGImages" / f"{stem}.png")
        Image.fromarray(mask).save(root / "PedMasks" / f"{stem}_mask.png")
        (root / "Annotation" / f"{stem}.txt").write_text(
            ANN.format(stem=stem, W=W, H=H, n=k, names=" ".join(['"PASpersonWalking"'] * k), objs=objs), encoding="latin-1")
        truth[f"{stem}.png"] = (W, H, boxes)
    return root, truth
