import os
import numpy as np
import pytest
from PIL import Image
from yolotools.augment import hflip, scale_translate, hsv_jitter, random_augment
from yolotools.voc2007 import (parse_voc_xml, encode_yolov1_target, decode_target, VOC2007Detection,
                               dataset_statistics, VOC_CLASSES)


def white_box_image(W=200, H=160, box=(60, 40, 120, 110)):
    a = np.zeros((H, W, 3), np.uint8); x1, y1, x2, y2 = box
    a[y1:y2, x1:x2] = 255
    return Image.fromarray(a), np.array([[x1 / W, y1 / H, x2 / W, y2 / H]])


def white_extent(img):
    a = np.asarray(img.convert("L")) > 200
    ys, xs = np.where(a)
    return xs.min() / img.size[0], ys.min() / img.size[1], (xs.max() + 1) / img.size[0], (ys.max() + 1) / img.size[1]


def test_hflip_matches_pixels():
    img, b = white_box_image()
    fi, fb = hflip(img, b)
    assert np.allclose(white_extent(fi), fb[0], atol=0.01)


@pytest.mark.parametrize("seed", range(10))
def test_scale_translate_matches_pixels(seed):
    img, b = white_box_image(box=(70, 50, 110, 100))        # central box, stays mostly inside
    ni, nb, keep = scale_translate(img, b, np.random.default_rng(seed), min_visible=0.0)
    if keep[0]:
        got = white_extent(ni)
        exp = np.clip(nb[0], 0, 1)
        assert np.allclose(got, exp, atol=0.02), (got, exp)


def test_box_dropped_when_pushed_out():
    img, _ = white_box_image()
    b = np.array([[0.0, 0.0, 0.05, 0.05]])
    dropped = 0
    for s in range(200):
        _, _, keep = scale_translate(img, b, np.random.default_rng(s))
        dropped += (~keep[0])
    assert dropped > 20


def test_hsv_preserves_shape_and_changes_pixels():
    img, _ = white_box_image()
    arr = np.random.default_rng(0).integers(0, 255, (50, 60, 3), dtype=np.uint8)
    out = hsv_jitter(Image.fromarray(arr), np.random.default_rng(0))
    assert out.size == (60, 50) and not np.array_equal(np.asarray(out), arr)


def test_random_augment_switches():
    img, b = white_box_image()
    i2, b2, l2 = random_augment(img, b, np.array([3]), np.random.default_rng(0), flip=False, hsv=False, affine=False)
    assert np.array_equal(np.asarray(i2), np.asarray(img)) and np.allclose(b2, b) and l2.tolist() == [3]


def test_encode_decode_target_roundtrip_and_collision():
    boxes = np.array([[0.5, 0.5, 0.2, 0.3], [0.52, 0.51, 0.4, 0.4], [0.1, 0.9, 0.1, 0.1]])
    t = encode_yolov1_target(boxes, [14, 7, 0], S=7, C=20)
    assert t.shape == (7, 7, 25) and t[..., 4].sum() == 2                  # 2nd box collides with 1st
    db, dl = decode_target(t)
    assert sorted(dl.tolist()) == [0, 14]
    first = db[dl.tolist().index(14)]
    assert np.allclose(first, boxes[0], atol=1e-6)
    t2 = encode_yolov1_target(boxes, [14, 7, 0], S=7, C=20, collision="largest")
    assert decode_target(t2)[1].tolist().count(7) == 1                     # bigger box wins


XML = """<annotation><folder>VOC2007</folder><filename>{id}.jpg</filename><size><width>{W}</width><height>{H}</height><depth>3</depth></size>
<segmented>0</segmented>{objs}</annotation>"""
OBJ = """<object><name>{n}</name><pose>Left</pose><truncated>0</truncated><difficult>{d}</difficult>
<bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>"""


@pytest.fixture(scope="module")
def fake_voc(tmp_path_factory):
    dk = tmp_path_factory.mktemp("voc") / "VOCdevkit" / "VOC2007"
    for d in ("Annotations", "JPEGImages", "ImageSets/Main"):
        os.makedirs(dk / d)
    spec = {"000001": (300, 200, [("dog", 0, 10, 20, 150, 120), ("person", 1, 160, 30, 290, 190)]),
            "000002": (240, 240, [("car", 0, 1, 1, 240, 240)]),
            "000003": (320, 180, [("aeroplane", 0, 50, 50, 100, 90), ("aeroplane", 0, 60, 55, 110, 95)])}
    for id_, (W, H, objs) in spec.items():
        Image.fromarray(np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)).save(dk / "JPEGImages" / f"{id_}.jpg")
        (dk / "Annotations" / f"{id_}.xml").write_text(XML.format(id=id_, W=W, H=H, objs="".join(
            OBJ.format(n=n, d=d, x1=a, y1=b, x2=c, y2=e) for n, d, a, b, c, e in objs)))
    (dk / "ImageSets/Main/trainval.txt").write_text("\n".join(spec) + "\n")
    return dk


def test_parse_voc_xml(fake_voc):
    a = parse_voc_xml(fake_voc / "Annotations" / "000001.xml")
    assert (a["width"], a["height"]) == (300, 200) and len(a["objects"]) == 2
    o = a["objects"][0]
    assert o["name"] == "dog" and o["label"] == VOC_CLASSES.index("dog") and o["box_edges"] == (9, 19, 150, 120)
    assert a["objects"][1]["difficult"] == 1


def test_voc_dataset_numpy_paths(fake_voc):
    ds = VOC2007Detection(str(fake_voc), "trainval", img_size=96)
    assert len(ds) == 3
    gt = ds.ground_truth(0)
    assert gt["difficult"].tolist() == [False, True] and np.allclose(gt["boxes"][0], [9 / 300, 19 / 200, 150 / 300, 120 / 200])
    arr, boxes, labels = ds.sample(0)                       # difficult person dropped for training
    assert arr.shape == (96, 96, 3) and len(boxes) == 1 and labels.tolist() == [VOC_CLASSES.index("dog")]
    ds_k = VOC2007Detection(str(fake_voc), "trainval", img_size=96, keep_difficult=True)
    assert len(ds_k.sample(0)[1]) == 2
    full = ds.ground_truth(1)["boxes"][0]
    assert np.allclose(full, [0, 0, 1, 1])
    t = encode_yolov1_target(boxes, labels, 7, 20)
    assert t[..., 4].sum() == 1
    aug = VOC2007Detection(str(fake_voc), "trainval", img_size=96, augment=True)
    for i in range(3):
        a, b, l = aug.sample(i, np.random.default_rng(i)); assert a.shape == (96, 96, 3) and len(b) == len(l)


def test_dataset_statistics(fake_voc):
    s = dataset_statistics(str(fake_voc), "trainval")
    assert s["images"] == 3 and s["objects"].sum() == 5 and s["non_difficult"].sum() == 4
    assert s["images_with_class"][VOC_CLASSES.index("aeroplane")] == 1
