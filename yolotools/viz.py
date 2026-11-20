"""Dataset figures: sample grid, mask instances, label check from SAVED labels, statistics plots."""
import csv, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from .penn_fudan import mask_instance_edges, yolo_to_edges
from .yolo_dataset import read_yolo_label

COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#008080", "#9a6324", "#000075"]


def _draw(ax, edges, color_offset=0):
    for k, (x1, y1, x2, y2) in enumerate(edges):
        c = COLORS[(k + color_offset) % len(COLORS)]
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=2, ec=c, fc="none"))
        ax.text(x1, y1 - 2, "pedestrian", color="white", fontsize=7, weight="bold", bbox=dict(fc=c, ec="none", pad=1))


def plot_sample_grid(root, n=6, seed=0, save=None):
    files = sorted(os.listdir(os.path.join(root, "PNGImages")))
    pick = np.random.default_rng(seed).choice(len(files), n, replace=False)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4.5 * ((n + 1) // 2), 8))
    for ax, i in zip(axes.flat, pick):
        ax.imshow(Image.open(os.path.join(root, "PNGImages", files[i]))); ax.set_title(files[i], fontsize=9); ax.axis("off")
    plt.tight_layout()
    if save: fig.savefig(save, dpi=110, bbox_inches="tight")
    return fig


def plot_mask_instances(root, stem="FudanPed00013", save=None):
    img = Image.open(os.path.join(root, "PNGImages", stem + ".png"))
    mask = np.array(Image.open(os.path.join(root, "PedMasks", stem + "_mask.png")))
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].imshow(img); ax[0].set_title("Image")
    ax[1].imshow(mask, cmap="gray"); ax[1].set_title("Raw mask (values 0,1,2,... look near-black)")
    ax[2].imshow(np.ma.masked_where(mask == 0, mask), cmap="tab10"); ax[2].set_title("Each ID = one pedestrian (mask-derived boxes)")
    _draw(ax[2], mask_instance_edges(os.path.join(root, "PedMasks", stem + "_mask.png")))
    for a in ax: a.axis("off")
    plt.tight_layout()
    if save: fig.savefig(save, dpi=110, bbox_inches="tight")
    return fig


def plot_label_check(out_dir, n=6, seed=0, save=None):
    """Draw boxes read back from the saved YOLO label files (what the models will actually see)."""
    items = [(s, f) for s in ("train", "val", "test") for f in sorted(os.listdir(os.path.join(out_dir, "images", s)))]
    pick = [items[i] for i in np.random.default_rng(seed).choice(len(items), n, replace=False)]
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4.5 * ((n + 1) // 2), 8))
    for ax, (s, f) in zip(axes.flat, pick):
        img = Image.open(os.path.join(out_dir, "images", s, f)); W, H = img.size
        _, xywh = read_yolo_label(os.path.join(out_dir, "labels", s, os.path.splitext(f)[0] + ".txt"))
        ax.imshow(img); _draw(ax, yolo_to_edges(xywh, W, H))
        ax.set_title(f"[{s}] {f} ({len(xywh)} boxes from saved labels)", fontsize=8); ax.axis("off")
    plt.tight_layout()
    if save: fig.savefig(save, dpi=110, bbox_inches="tight")
    return fig


def plot_statistics(out_dir, save=None):
    rows = list(csv.DictReader(open(os.path.join(out_dir, "split_manifest.csv"))))
    per_split = {s: [] for s in ("train", "val", "test")}
    hs, ws, ars = [], [], []
    for r in rows:
        _, xywh = read_yolo_label(os.path.join(out_dir, "labels", r["split"], os.path.splitext(r["filename"])[0] + ".txt"))
        per_split[r["split"]].append(len(xywh))
        if len(xywh):
            hs += list(xywh[:, 3]); ws += list(xywh[:, 2])
            ars += list((xywh[:, 2] * int(r["width"])) / (xywh[:, 3] * int(r["height"])))
    fig, ax = plt.subplots(1, 4, figsize=(19, 4))
    ax[0].bar(list(per_split), [len(v) for v in per_split.values()], color="#4363d8"); ax[0].set_title("Images per split")
    mx = max(max(v) for v in per_split.values() if v)
    for s, v in per_split.items():
        ax[1].hist(v, bins=np.arange(0.5, mx + 1.5), alpha=.55, label=s, density=True)
    ax[1].set_title("Pedestrians per image"); ax[1].legend()
    ax[2].hist(hs, bins=30, color="#3cb44b"); ax[2].set_title("Box height / image height")
    ax[3].hist(ars, bins=30, color="#f58231"); ax[3].set_title("Box aspect ratio (w/h, pixels)")
    plt.tight_layout()
    if save: fig.savefig(save, dpi=110, bbox_inches="tight")
    return fig, {"box_h_median": float(np.median(hs)), "box_w_median": float(np.median(ws)),
                 "aspect_median": float(np.median(ars)), "small_boxes_h<0.2": int(np.sum(np.array(hs) < 0.2))}
