#!/usr/bin/env python
"""CLI: download Penn-Fudan, convert to YOLO format, split, run quality checks, make figures.

    python scripts/prepare_penn_fudan.py --out PennFudanYOLO [--manifest PennFudanYOLO/split_manifest.csv]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
from yolotools import penn_fudan as pf, quality, viz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="existing PennFudanPed folder (else downloaded to --download-dir)")
    ap.add_argument("--download-dir", default="/content")
    ap.add_argument("--out", default="PennFudanYOLO")
    ap.add_argument("--manifest", default=None, help="reuse the split of an existing split_manifest.csv")
    ap.add_argument("--seed", type=int, default=pf.SEED)
    a = ap.parse_args()
    root = a.root or pf.download_penn_fudan(a.download_dir)
    records, split, issues, warnings, stats = pf.prepare(root, a.out, a.manifest, a.seed)
    res = quality.run_quality_checks(a.out, root, pf.EXPECTED["images"], pf.EXPECTED["instances"])
    print(quality.write_report(res, os.path.join(a.out, "quality_report.txt")))
    print("stats:", stats, "| warnings:", len(warnings))
    viz.plot_sample_grid(root, save=os.path.join(a.out, "sample_grid.png"))
    viz.plot_label_check(a.out, save=os.path.join(a.out, "label_check_samples.png"))
    viz.plot_statistics(a.out, save=os.path.join(a.out, "dataset_stats_plots.png"))
    sys.exit(0 if all(r[0] for r in res) else 1)


if __name__ == "__main__":
    main()
