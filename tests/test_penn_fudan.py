import csv, os, shutil
import numpy as np
import pytest
from yolotools.penn_fudan import (parse_annotation, annotation_to_edges, edges_to_yolo, yolo_to_edges,
                                  build_records, make_split, prepare, load_manifest_split)
from yolotools.quality import run_quality_checks, write_report
from yolotools.yolo_dataset import YoloFolderDataset, read_yolo_label


def test_parse_annotation(fake_pf):
    root, truth = fake_pf
    stem = "FudanPed00001"
    a = parse_annotation(root / "Annotation" / f"{stem}.txt")
    W, H, boxes = truth[f"{stem}.png"]
    assert (a["width"], a["height"]) == (W, H) and a["boxes"] == boxes


def test_conversion_math():
    # 1-indexed inclusive (1,1)-(10,20) on a 100x200 image covers pixels 0..9 x 0..19 -> edges (0,0,10,20)
    e = annotation_to_edges((1, 1, 10, 20))
    assert e == (0, 0, 10, 20)
    y = edges_to_yolo([e], 100, 200)[0]
    assert np.allclose(y, [0.05, 0.05, 0.10, 0.10])
    assert np.allclose(yolo_to_edges([y], 100, 200)[0], e)


def test_split_sizes_and_determinism():
    names = [f"f{i:03d}.png" for i in range(170)]
    s1, s2 = make_split(names), make_split(names)
    assert s1 == s2
    counts = {k: sum(v == k for v in s1.values()) for k in ("train", "val", "test")}
    assert counts == {"train": 119, "val": 25, "test": 26}            # matches the team's manifest sizes
    assert make_split(names, seed=1) != s1


def test_full_pipeline(fake_pf, tmp_path):
    root, truth = fake_pf
    out = tmp_path / "PennFudanYOLO"
    records, split, issues, warnings, stats = prepare(str(root), str(out))
    assert all(not v for v in issues.values())
    # overlapping synthetic boxes are occluded in the mask -> only IoU warnings are acceptable
    assert all("IoU" in w for w in warnings)
    assert len(records) == 40 and sum(r["min_mask_iou"] >= 0.999 for r in records) >= 30
    n_inst = sum(len(b) for _, _, b in truth.values())
    assert stats[0][1:] == (40, n_inst)
    assert {s: len(os.listdir(out / "images" / s)) for s in ("train", "val", "test")} == {"train": 28, "val": 6, "test": 6}
    res = run_quality_checks(str(out), str(root), expected_images=40, expected_instances=n_inst)
    assert len(res) == 10 and all(ok for ok, *_ in res), res
    txt = write_report(res, str(out / "quality_report.txt"))
    assert txt.count("PASS") == 10
    # label file content
    f = sorted(os.listdir(out / "labels" / "train"))[0]
    cls, xywh = read_yolo_label(str(out / "labels" / "train" / f))
    assert set(cls) == {0} and ((xywh > 0) & (xywh <= 1)).all()


def test_manifest_is_reused(fake_pf, tmp_path):
    root, _ = fake_pf
    a = tmp_path / "a"; prepare(str(root), str(a))
    # flip a split in the manifest and make sure prepare() honours it
    rows = list(csv.DictReader(open(a / "split_manifest.csv")))
    rows[0]["split"] = "test" if rows[0]["split"] != "test" else "train"
    with open(tmp_path / "m.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    b = tmp_path / "b"; _, split, *_ = prepare(str(root), str(b), manifest_path=str(tmp_path / "m.csv"))
    assert split[rows[0]["filename"]] == rows[0]["split"]


def test_quality_checks_detect_corruption(fake_pf, tmp_path):
    root, _ = fake_pf
    out = tmp_path / "ds"; prepare(str(root), str(out))
    lab = sorted((out / "labels" / "train").iterdir())
    # bad class id, out-of-range box, missing label, duplicate image across splits
    lab[0].write_text("3 0.5 0.5 0.2 0.2\n")
    lab[1].write_text("0 0.99 0.5 0.5 0.2\n")
    lab[2].unlink()
    img = sorted((out / "images" / "val").iterdir())[0]
    shutil.copy(img, out / "images" / "test" / img.name)
    res = {i: ok for ok, i, *_ in run_quality_checks(str(out), str(root), n_sample=200)}
    assert not res[1] and not res[3] and not res[4] and not res[6] and not res[7]


def test_dataset_loader_with_and_without_aug(fake_pf, tmp_path):
    root, _ = fake_pf
    out = tmp_path / "ds"; prepare(str(root), str(out))
    plain = YoloFolderDataset(str(out), "train", 128)
    arr, boxes, labels = plain.sample(0)
    assert arr.shape == (128, 128, 3) and boxes.shape[1] == 4 and len(boxes) == len(plain.ground_truth(0)["boxes"])
    aug = YoloFolderDataset(str(out), "train", 128, augment=True)
    rng = np.random.default_rng(0)
    for i in range(len(aug)):
        a, b, l = aug.sample(i, rng)
        assert a.shape == (128, 128, 3) and len(b) == len(l)
        if len(b):
            assert ((b[:, :2] - b[:, 2:] / 2) >= -1e-6).all() and ((b[:, :2] + b[:, 2:] / 2) <= 1 + 1e-6).all()
