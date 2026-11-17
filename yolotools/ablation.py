"""Augmentation ON vs OFF ablation with Ultralytics YOLO on Penn-Fudan (RQ2).

Design
------
* Same model weights, data, epochs, image size, batch, optimiser settings and seeds for every variant;
  ONLY the augmentation hyper-parameters differ.
* `default_aug`  = Ultralytics defaults (HSV, flip, scale, translate, mosaic, ...).
* `no_aug`       = every stochastic augmentation set to zero (images are only letter-boxed/resized, which is
                   preprocessing, not augmentation).
* Several seeds per variant (tiny dataset -> high variance). Test split evaluated ONCE per run with the
  checkpoint chosen on the VALIDATION split (Ultralytics best.pt).
"""
import csv, json, os
import numpy as np

# every stochastic-augmentation argument Ultralytics knows about, set to "off"
AUG_OFF = dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
               perspective=0.0, flipud=0.0, fliplr=0.0, bgr=0.0, mosaic=0.0, mixup=0.0, cutmix=0.0,
               copy_paste=0.0, erasing=0.0, close_mosaic=0)
# optional single-component variants ("one at a time")
EXTRA_VARIANTS = {
    "flip_only": dict(AUG_OFF, fliplr=0.5),
    "hsv_only": dict(AUG_OFF, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4),
    "geometric_only": dict(AUG_OFF, translate=0.1, scale=0.5),
    "mosaic_only": dict(AUG_OFF, mosaic=1.0),
}
VARIANTS = {"no_aug": AUG_OFF, "default_aug": {}}


def supported(overrides):
    """Drop arguments the installed Ultralytics version does not know (e.g. cutmix on older releases)."""
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
        return {k: v for k, v in overrides.items() if k in DEFAULT_CFG_DICT}
    except Exception:                                      # noqa: BLE001
        return dict(overrides)


def run_one(variant, overrides, seed, data_yaml, model_name="yolo11n.pt", epochs=50, imgsz=640, batch=16,
            device=0, project="runs/aug_ablation", workers=2):
    from ultralytics import YOLO
    name = f"{variant}_seed{seed}"
    model = YOLO(model_name)
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, seed=seed, deterministic=True,
                device=device, project=project, name=name, exist_ok=True, plots=False, verbose=False,
                workers=workers, patience=epochs, **supported(overrides))
    best = YOLO(os.path.join(project, name, "weights", "best.pt"))
    out = {"variant": variant, "seed": seed, "model": model_name, "epochs": epochs, "imgsz": imgsz}
    for split in ("val", "test"):
        m = best.val(data=data_yaml, split=split, imgsz=imgsz, batch=batch, device=device, conf=0.001,
                     iou=0.6, plots=False, verbose=False)
        out.update({f"{split}_precision": float(m.box.mp), f"{split}_recall": float(m.box.mr),
                    f"{split}_map50": float(m.box.map50), f"{split}_map50_95": float(m.box.map)})
        if split == "test":
            out.update({f"speed_{k}_ms": float(v) for k, v in m.speed.items()})
    return out


def run_ablation(data_yaml, variants=None, seeds=(0, 1, 2), out_csv="aug_ablation_runs.csv", **kw):
    """Run every variant x seed, appending to out_csv after each run (resumable if Colab disconnects)."""
    variants = variants or VARIANTS
    done = set()
    rows = []
    if os.path.exists(out_csv):
        rows = list(csv.DictReader(open(out_csv)))
        done = {(r["variant"], int(r["seed"])) for r in rows}
    for v, ov in variants.items():
        for s in seeds:
            if (v, s) in done:
                print(f"skip {v} seed {s} (already in {out_csv})"); continue
            print(f"=== {v} | seed {s} ===")
            r = run_one(v, ov, s, data_yaml, **kw)
            rows.append(r)
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=sorted({k for x in rows for k in x}))
                w.writeheader(); w.writerows(rows)
    return rows


def summarise(rows, metrics=("test_map50", "test_map50_95", "test_precision", "test_recall",
                             "val_map50", "val_map50_95")):
    """-> {variant: {metric: (mean, std, n)}} (sample std, ddof=1 when n>1)."""
    out = {}
    for v in sorted({r["variant"] for r in rows}):
        sel = [r for r in rows if r["variant"] == v]
        out[v] = {m: (float(np.mean([float(r[m]) for r in sel])),
                      float(np.std([float(r[m]) for r in sel], ddof=1)) if len(sel) > 1 else 0.0, len(sel)) for m in metrics}
    return out


def markdown_table(summary, metrics=("test_map50", "test_map50_95", "test_precision", "test_recall")):
    head = "| Variant | n | " + " | ".join(metrics) + " |\n|---|---|" + "---|" * len(metrics)
    lines = [head]
    for v, d in summary.items():
        lines.append(f"| {v} | {d[metrics[0]][2]} | " + " | ".join(f"{d[m][0]:.3f} ± {d[m][1]:.3f}" for m in metrics) + " |")
    return "\n".join(lines)
