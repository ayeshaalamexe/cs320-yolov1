"""Box-aware augmentation for YOLOv1-style loaders (PIL + NumPy).

Follows the YOLOv1 paper: random scale + translation up to 20 %, HSV exposure/saturation jitter up to
x1.5, plus horizontal flip. Boxes are normalised xyxy (0-1) and are transformed together with the image.
"""
import numpy as np
from PIL import Image


def hsv_jitter(img, rng, hue=0.05, sat=1.5, val=1.5):
    hsv = np.array(img.convert("HSV"), dtype=np.float32)
    fs = sat ** rng.uniform(-1, 1)
    fv = val ** rng.uniform(-1, 1)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-hue, hue) * 255) % 256
    hsv[..., 1] = np.clip(hsv[..., 1] * fs, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * fv, 0, 255)
    return Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")


def hflip(img, boxes):
    out = boxes.copy()
    out[:, [0, 2]] = 1.0 - boxes[:, [2, 0]]
    return img.transpose(Image.FLIP_LEFT_RIGHT), out


def scale_translate(img, boxes, rng, max_scale=0.2, max_shift=0.2, min_visible=0.3, min_size=0.01):
    """Return (img, boxes, keep_mask). Boxes losing <min_visible of their area or smaller than min_size are dropped."""
    W, H = img.size
    s = rng.uniform(1 - max_scale, 1 + max_scale)
    tx, ty = rng.uniform(-max_shift, max_shift, 2)
    # output(x', y') samples input at (x'-(0.5+t)W)/s + 0.5W
    coeffs = (1 / s, 0, 0.5 * W - (0.5 + tx) * W / s, 0, 1 / s, 0.5 * H - (0.5 + ty) * H / s)
    img = img.transform((W, H), Image.AFFINE, coeffs, resample=Image.BILINEAR, fillcolor=(127, 127, 127))
    nb = boxes.copy()
    nb[:, [0, 2]] = (boxes[:, [0, 2]] - 0.5) * s + 0.5 + tx
    nb[:, [1, 3]] = (boxes[:, [1, 3]] - 0.5) * s + 0.5 + ty
    area0 = np.clip(nb[:, 2] - nb[:, 0], 0, None) * np.clip(nb[:, 3] - nb[:, 1], 0, None)
    nb = np.clip(nb, 0.0, 1.0)
    area1 = (nb[:, 2] - nb[:, 0]) * (nb[:, 3] - nb[:, 1])
    keep = (area1 >= min_visible * np.maximum(area0, 1e-12)) & ((nb[:, 2] - nb[:, 0]) >= min_size) & ((nb[:, 3] - nb[:, 1]) >= min_size)
    return img, nb, keep


def random_augment(img, boxes, labels, rng=None, p_flip=0.5, p_hsv=1.0, p_affine=1.0,
                   flip=True, hsv=True, affine=True):
    """img: PIL RGB; boxes (N,4) normalised xyxy; labels (N,). Each op can be switched off for ablations."""
    rng = rng or np.random.default_rng()
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    labels = np.asarray(labels).copy()
    if hsv and rng.random() < p_hsv:
        img = hsv_jitter(img, rng)
    if flip and rng.random() < p_flip:
        img, boxes = hflip(img, boxes)
    if affine and rng.random() < p_affine:
        img, boxes, keep = scale_translate(img, boxes, rng)
        boxes, labels = boxes[keep], labels[keep]
    return img, boxes, labels
