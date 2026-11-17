import os, subprocess, sys
import matplotlib; matplotlib.use("Agg")
from yolotools import viz, ablation
from yolotools.penn_fudan import prepare


def test_viz_and_cli(fake_pf, tmp_path):
    root, _ = fake_pf
    out = tmp_path / "ds"
    r = subprocess.run([sys.executable, "scripts/prepare_penn_fudan.py", "--root", str(root), "--out", str(out)],
                       capture_output=True, text=True)
    # real-count asserts (170/423) cannot hold for the 40-image fixture -> check 8 is expected to fail only there
    assert "PASS  1." in r.stdout and "PASS  10." in r.stdout and "FAIL  8." in r.stdout, r.stdout + r.stderr
    for f in ("sample_grid.png", "label_check_samples.png", "dataset_stats_plots.png", "quality_report.txt",
              "conversion_log.txt", "dataset_stats.csv", "split_manifest.csv", "data.yaml", "README_dataset.md"):
        assert (out / f).exists(), f
    viz.plot_mask_instances(str(root), "FudanPed00001", save=str(tmp_path / "m.png"))
    assert (tmp_path / "m.png").exists()


def test_ablation_summary_helpers():
    rows = [{"variant": "a", "seed": s, **{m: v for m in ("test_map50", "test_map50_95", "test_precision", "test_recall", "val_map50", "val_map50_95")}}
            for s, v in ((0, 0.8), (1, 0.9))]
    s = ablation.summarise(rows)
    assert abs(s["a"]["test_map50"][0] - 0.85) < 1e-9 and s["a"]["test_map50"][2] == 2
    assert "0.850" in ablation.markdown_table(s)
    assert set(ablation.AUG_OFF) >= {"mosaic", "fliplr", "hsv_h", "scale", "translate", "mixup"}
    assert ablation.supported({"foo_bar": 1}) in ({}, {"foo_bar": 1})   # no ultralytics here -> passthrough
