"""Penn-Fudan Pedestrian -> YOLO-format dataset (download, parse, convert, split, write).

Official PASCAL annotation boxes are 1-indexed INCLUSIVE pixel coordinates. They are converted to
continuous edges  (xmin-1, ymin-1, xmax, ymax)  and then to normalised YOLO `class xc yc w h`.
Every box is cross-checked against the box derived from the instance mask.
"""
import csv, json, os, re, shutil, urllib.request, zipfile
import numpy as np
from PIL import Image
from .boxes import iou_matrix, xyxy_to_xywh

URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
SEED = 42
RATIOS = (0.70, 0.15, 0.15)
EXPECTED = {"images": 170, "instances": 423}          # published Penn-Fudan totals
MIN_MASK_IOU_WARN = 0.70

_BOX_RE = re.compile(r"Bounding box for object\s+\d+.*?:\s*\((\d+),\s*(\d+)\)\s*-\s*\((\d+),\s*(\d+)\)")
_SIZE_RE = re.compile(r"Image size \(X x Y x C\)\s*:\s*(\d+)\s*x\s*(\d+)\s*x\s*(\d+)")


def download_penn_fudan(dest="/content", url=URL):
    """Download + unzip; returns the dataset root (folder containing PNGImages/)."""
    os.makedirs(dest, exist_ok=True)
    root = os.path.join(dest, "PennFudanPed")
    if os.path.isdir(os.path.join(root, "PNGImages")):
        return root
    zpath = os.path.join(dest, "PennFudanPed.zip")
    urllib.request.urlretrieve(url, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
    return root


def parse_annotation(path):
    """-> dict(width, height, boxes=[(xmin,ymin,xmax,ymax)] in 1-indexed inclusive pixels)."""
    txt = open(path, encoding="latin-1").read()
    m = _SIZE_RE.search(txt)
    if not m:
        raise ValueError(f"no image size in {path}")
    boxes = [tuple(int(v) for v in g) for g in _BOX_RE.findall(txt)]
    return {"width": int(m.group(1)), "height": int(m.group(2)), "boxes": boxes}


def annotation_to_edges(b):
    xmin, ymin, xmax, ymax = b
    return (xmin - 1, ymin - 1, xmax, ymax)


def mask_instance_edges(mask_path):
    """Instance boxes from the mask (0 = background, 1..N = pedestrians) as continuous edges."""
    mask = np.array(Image.open(mask_path))
    out = []
    for i in np.unique(mask):
        if i == 0:
            continue
        ys, xs = np.where(mask == i)
        out.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    return np.array(out, dtype=np.float64).reshape(-1, 4)


def edges_to_yolo(edges, W, H):
    xc, yc, w, h = xyxy_to_xywh(np.asarray(edges, float).reshape(-1, 4)).T
    return np.stack([xc / W, yc / H, w / W, h / H], axis=1)


def yolo_to_edges(yolo, W, H):
    yolo = np.asarray(yolo, float).reshape(-1, 4)
    return np.stack([(yolo[:, 0] - yolo[:, 2] / 2) * W, (yolo[:, 1] - yolo[:, 3] / 2) * H,
                     (yolo[:, 0] + yolo[:, 2] / 2) * W, (yolo[:, 1] + yolo[:, 3] / 2) * H], axis=1)


def source_set(name):
    return "Penn" if name.startswith("PennPed") else "Fudan"


def build_records(root):
    """Parse + validate everything. Returns (records, issues, warnings)."""
    d_img, d_ann, d_msk = (os.path.join(root, d) for d in ("PNGImages", "Annotation", "PedMasks"))
    imgs = sorted(f for f in os.listdir(d_img) if f.lower().endswith(".png"))
    anns = {os.path.splitext(f)[0] for f in os.listdir(d_ann) if f.endswith(".txt")}
    msks = {f[:-len("_mask.png")] for f in os.listdir(d_msk) if f.endswith("_mask.png")}
    stems = {os.path.splitext(f)[0] for f in imgs}
    issues = {"unreadable_images": [], "missing_masks": [], "missing_annotations": [], "orphan_masks": [],
              "orphan_annotations": [], "size_mismatch": [], "invalid_boxes": [], "conversion_failures": []}
    warnings = []
    issues["missing_annotations"] = sorted(stems - anns)
    issues["missing_masks"] = sorted(stems - msks)
    issues["orphan_annotations"] = sorted(anns - stems)
    issues["orphan_masks"] = sorted(msks - stems)
    records = []
    for f in imgs:
        stem = os.path.splitext(f)[0]
        if stem in issues["missing_annotations"]:
            continue
        try:
            with Image.open(os.path.join(d_img, f)) as im:
                im.verify()
            with Image.open(os.path.join(d_img, f)) as im:
                W, H = im.size
        except Exception as e:                                  # noqa: BLE001
            issues["unreadable_images"].append(f"{f}: {e}")
            continue
        try:
            ann = parse_annotation(os.path.join(d_ann, stem + ".txt"))
        except Exception as e:                                  # noqa: BLE001
            issues["conversion_failures"].append(f"{f}: {e}")
            continue
        if (ann["width"], ann["height"]) != (W, H):
            issues["size_mismatch"].append(f"{f}: annotation {ann['width']}x{ann['height']} vs image {W}x{H}")
        edges = np.array([annotation_to_edges(b) for b in ann["boxes"]], float).reshape(-1, 4)
        bad = [i for i, e in enumerate(edges) if e[2] <= e[0] or e[3] <= e[1]]
        if bad:
            issues["invalid_boxes"].append(f"{f}: degenerate box index {bad}")
            edges = np.delete(edges, bad, axis=0)
        clipped = np.clip(edges, 0, [W, H, W, H])
        if len(edges) and not np.allclose(clipped, edges):
            warnings.append(f"{f}: box extended outside image and was clipped")
        edges = clipped
        miou = None
        if stem not in issues["missing_masks"]:
            mb = mask_instance_edges(os.path.join(d_msk, stem + "_mask.png"))
            if len(mb) != len(edges):
                warnings.append(f"{f}: {len(edges)} annotation boxes vs {len(mb)} mask instances")
            if len(mb) and len(edges):
                best = iou_matrix(edges, mb).max(axis=1)
                miou = float(best.min())
                if miou < MIN_MASK_IOU_WARN:
                    warnings.append(f"{f}: lowest annotation-vs-mask IoU {miou:.2f}")
        records.append({"filename": f, "source_set": source_set(f), "width": W, "height": H,
                        "edges": edges, "min_mask_iou": miou})
    return records, issues, warnings


def make_split(filenames, seed=SEED, ratios=RATIOS):
    """Image-level random split. Floor for train/val, remainder to test (170 -> 119/25/26)."""
    names = sorted(filenames)
    perm = np.random.RandomState(seed).permutation(len(names))
    n_tr = int(np.floor(ratios[0] * len(names) + 1e-9))   # epsilon: 0.7*170 == 118.99999...
    n_va = int(np.floor(ratios[1] * len(names) + 1e-9))
    split = {}
    for rank, i in enumerate(perm):
        split[names[i]] = "train" if rank < n_tr else ("val" if rank < n_tr + n_va else "test")
    return split


def load_manifest_split(path):
    with open(path, newline="") as f:
        return {r["filename"]: r["split"] for r in csv.DictReader(f)}


DATASET_README = """# PennFudanYOLO (Member 1 handover)
- Source: official Penn-Fudan zip ({url})
- Labels: YOLO detection format `0 xc yc w h` (normalised by each image's own size), class 0 = pedestrian
- Boxes: official PASCAL annotation boxes (1-indexed inclusive -> converted), cross-checked against mask instance boxes
- Split: image level 70/15/15, seed {seed}, see split_manifest.csv
- Images kept at original resolution (resizing is done by the training code; normalised labels are unaffected)
- Do NOT tune hyper-parameters on the test split
- Update `path:` in data.yaml to wherever this folder lives on your machine/Colab
"""


def write_dataset(root, out_dir, records, split, seed=SEED, issues=None, warnings=None):
    """Write images/, labels/, split_manifest.csv, dataset_stats.csv, data.yaml, README_dataset.md, conversion_log.txt."""
    if os.path.isdir(out_dir):
        for sub in ("images", "labels"):
            shutil.rmtree(os.path.join(out_dir, sub), ignore_errors=True)
    for s in ("train", "val", "test"):
        os.makedirs(os.path.join(out_dir, "images", s), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", s), exist_ok=True)
    rows = []
    for r in records:
        s = split[r["filename"]]
        shutil.copy2(os.path.join(root, "PNGImages", r["filename"]), os.path.join(out_dir, "images", s, r["filename"]))
        yolo = edges_to_yolo(r["edges"], r["width"], r["height"])
        with open(os.path.join(out_dir, "labels", s, os.path.splitext(r["filename"])[0] + ".txt"), "w") as f:
            for xc, yc, w, h in yolo:
                f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        rows.append({"filename": r["filename"], "split": s, "source_set": r["source_set"], "width": r["width"],
                     "height": r["height"], "n_pedestrians": len(yolo), "label_source": "annotation"})
    rows.sort(key=lambda x: x["filename"])
    with open(os.path.join(out_dir, "split_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    stats = [("Total", len(rows), sum(r["n_pedestrians"] for r in rows))]
    for s in ("train", "val", "test"):
        sel = [r for r in rows if r["split"] == s]
        stats.append((s.capitalize(), len(sel), sum(r["n_pedestrians"] for r in sel)))
    with open(os.path.join(out_dir, "dataset_stats.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Metric", "Images", "Pedestrian instances"]); w.writerows(stats)
    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        f.write("# Auto-generated by Member 1 notebook (Penn-Fudan, single class)\n"
                f"path: {os.path.abspath(out_dir)}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                "nc: 1\nnames:\n  0: pedestrian\n")
    with open(os.path.join(out_dir, "README_dataset.md"), "w") as f:
        f.write(DATASET_README.format(url=URL, seed=seed))
    with open(os.path.join(out_dir, "conversion_log.txt"), "w") as f:
        f.write("ISSUES\n" + json.dumps(issues or {}, indent=2) + "\n\nWARNINGS\n" + "\n".join(warnings or []) + "\n")
    return rows, stats


def prepare(root, out_dir, manifest_path=None, seed=SEED):
    """Full pipeline. If manifest_path is given its split is reused (keeps team results comparable)."""
    records, issues, warnings = build_records(root)
    names = [r["filename"] for r in records]
    if manifest_path and os.path.exists(manifest_path):
        split = load_manifest_split(manifest_path)
        missing = set(names) - set(split)
        if missing:
            raise ValueError(f"manifest lacks {len(missing)} images, e.g. {sorted(missing)[:3]}")
        split = {n: split[n] for n in names}
    else:
        split = make_split(names, seed)
    rows, stats = write_dataset(root, out_dir, records, split, seed, issues, warnings)
    return records, split, issues, warnings, stats
