"""The 10 dataset quality checks -> quality_report.txt (PASS/FAIL per check)."""
import csv, os
import numpy as np
from PIL import Image
from .penn_fudan import parse_annotation, annotation_to_edges, yolo_to_edges

SPLITS = ("train", "val", "test")


def _read_labels(path):
    rows = []
    with open(path) as f:
        for ln in f:
            if ln.strip():
                rows.append(ln.split())
    return rows


def run_quality_checks(out_dir, root=None, expected_images=None, expected_instances=None, n_sample=40, seed=0):
    """Return list of (ok, number, title, detail). `root` (original Penn-Fudan) enables check 10."""
    res = []
    def add(i, title, problems):
        res.append((not problems, i, title, "all good" if not problems else f"{len(problems)} problem(s): {problems[:5]}"))

    imgs = {s: sorted(os.listdir(os.path.join(out_dir, "images", s))) for s in SPLITS}
    labs = {s: sorted(os.listdir(os.path.join(out_dir, "labels", s))) for s in SPLITS}
    # 1
    p = [f"{s}/{f}" for s in SPLITS for f in imgs[s] if os.path.splitext(f)[0] + ".txt" not in set(labs[s])]
    add(1, "Every image has a label file", p)
    # load all label rows once
    parsed = {}
    for s in SPLITS:
        for f in labs[s]:
            parsed[(s, f)] = _read_labels(os.path.join(out_dir, "labels", s, f))
    # 2
    p = [f"{s}/{f}" for (s, f), rows in parsed.items() for r in rows
         if len(r) != 5 or not all(_isfloat(v) for v in r)]
    add(2, "Label rows are valid YOLO (5 fields)", p)
    ok_rows = {k: [list(map(float, r)) for r in v if len(r) == 5 and all(_isfloat(x) for x in r)] for k, v in parsed.items()}
    # 3
    p = [f"{s}/{f}" for (s, f), rows in ok_rows.items() for r in rows if r[0] != 0 or r[0] != int(r[0])]
    add(3, "Class IDs valid (only 0)", p)
    # 4
    p = [f"{s}/{f}" for (s, f), rows in ok_rows.items() for r in rows
         if not (0 <= r[1] <= 1 and 0 <= r[2] <= 1 and 0 < r[3] <= 1 and 0 < r[4] <= 1
                 and r[1] - r[3] / 2 >= -1e-6 and r[1] + r[3] / 2 <= 1 + 1e-6
                 and r[2] - r[4] / 2 >= -1e-6 and r[2] + r[4] / 2 <= 1 + 1e-6)]
    add(4, "Normalised values within 0-1 (boxes inside image)", p)
    # 5
    p = [f"{s}/{f}" for (s, f), rows in ok_rows.items() for r in rows if r[3] <= 0 or r[4] <= 0]
    add(5, "Positive box width/height", p)
    # 6
    seen, p = {}, []
    for s in SPLITS:
        for f in imgs[s]:
            if f in seen:
                p.append(f"{f} in {seen[f]} and {s}")
            seen[f] = s
    add(6, "No image in more than one split", p)
    # 7
    p = [f"{s}: {len(imgs[s])} images vs {len(labs[s])} labels" for s in SPLITS if len(imgs[s]) != len(labs[s])]
    add(7, "Image count == label count per split", p)
    # 8
    p = []
    man = os.path.join(out_dir, "split_manifest.csv")
    if os.path.exists(man):
        rows = list(csv.DictReader(open(man)))
        for s in SPLITS:
            sel = [r for r in rows if r["split"] == s]
            n_inst = sum(len(v) for (ss, _), v in ok_rows.items() if ss == s)
            if len(sel) != len(imgs[s]):
                p.append(f"{s}: manifest {len(sel)} images vs folder {len(imgs[s])}")
            if sum(int(r["n_pedestrians"]) for r in sel) != n_inst:
                p.append(f"{s}: manifest instances != label rows ({n_inst})")
        if expected_images is not None and len(rows) != expected_images:
            p.append(f"total images {len(rows)} != expected {expected_images}")
        if expected_instances is not None and sum(int(r["n_pedestrians"]) for r in rows) != expected_instances:
            p.append("total instances != expected")
    else:
        p.append("split_manifest.csv missing")
    add(8, "Each split has the expected files / instance total", p)
    # 9
    p = []
    yml = os.path.join(out_dir, "data.yaml")
    if not os.path.exists(yml):
        p.append("data.yaml missing")
    else:
        cfg = {}
        for ln in open(yml):
            if ":" in ln and not ln.lstrip().startswith("#") and not ln.startswith(" "):
                k, v = ln.split(":", 1); cfg[k.strip()] = v.strip()
        base = cfg.get("path", out_dir)
        if not os.path.isdir(base):                      # data.yaml may point to Colab path; fall back to folder
            base = out_dir
        for k in ("train", "val", "test"):
            if not os.path.isdir(os.path.join(base, cfg.get(k, "?"))):
                p.append(f"{k} dir missing")
        if cfg.get("nc") != "1":
            p.append("nc != 1")
    add(9, "data.yaml points to valid directories", p)
    # 10
    p = []
    if root is None:
        res.append((True, 10, "Sampled saved labels match original boxes", "skipped (no original root given)"))
    else:
        rng = np.random.default_rng(seed)
        keys = sorted(ok_rows)
        for k in [keys[i] for i in rng.choice(len(keys), min(n_sample, len(keys)), replace=False)]:
            s, f = k
            stem = os.path.splitext(f)[0]
            with Image.open(os.path.join(out_dir, "images", s, stem + ".png")) as im:
                W, H = im.size
            ann = parse_annotation(os.path.join(root, "Annotation", stem + ".txt"))
            orig = np.array([annotation_to_edges(b) for b in ann["boxes"]], float).reshape(-1, 4)
            orig = np.clip(orig, 0, [W, H, W, H])
            saved = yolo_to_edges(np.array([r[1:] for r in ok_rows[k]]).reshape(-1, 4), W, H)
            if orig.shape != saved.shape or not np.allclose(orig, saved, atol=0.05):
                p.append(stem)
        add(10, "Sampled saved labels match original boxes", p)
    return res


def _isfloat(v):
    try:
        float(v); return True
    except ValueError:
        return False


def write_report(results, path):
    lines = [f"{'PASS' if ok else 'FAIL'}  {i}. {t}  -> {d}" for ok, i, t, d in results]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)
