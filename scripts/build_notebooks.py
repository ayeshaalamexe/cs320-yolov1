"""Build the consolidated Hassaan Adil Data Lead Colab notebook."""
import json, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_COLAB = "/content/cs320-yolov1"


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True)}


def module_cells(mods):
    cells = [md("### Project modules\nThe cells below write the `yolotools` package to disk (same files that live in the GitHub repo under `yolotools/`). "
                "Run them once; nothing needs to be edited."),
             code(f"!mkdir -p {REPO_COLAB}/yolotools")]
    for m in ["__init__"] + mods:
        src = open(os.path.join(HERE, "yolotools", m + ".py")).read()
        cells.append(code(f"%%writefile {REPO_COLAB}/yolotools/{m}.py\n{src}"))
    cells.append(code(f"import sys\nsys.path.insert(0, '{REPO_COLAB}')\nimport yolotools\nprint('yolotools', yolotools.__version__, 'ready')"))
    return cells


BUILT = {}

def save(name, cells):
    # Keep the existing consolidated notebook content and replace generated A/B/C sections
    # with the current source definitions at the end of the build.
    BUILT[name] = cells
    print("prepared", name, len(cells), "cells")


SETUP = '''
import os, sys, json, shutil, random
import numpy as np
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED); np.random.seed(SEED)
try:                                   # Drive is used to keep results if Colab disconnects
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE = '/content/drive/MyDrive/yolo_project'
    os.makedirs(DRIVE, exist_ok=True)
except Exception:
    DRIVE = None
print('Drive folder:', DRIVE)
'''

# =============================================================== Notebook A: Penn-Fudan data lead
A = [md("""
# CS-320 · YOLOv1 project · Hassaan (Data lead) — Part A: Penn-Fudan dataset
**Scope (from the project proposal):** dataset loading, YOLO-format conversion, splits, quality checks, statistics, split manifest, `data.yaml`.

| Step | What this notebook does | Output |
|---|---|---|
| 1 | Download + integrity-check Penn-Fudan (170 images) | `PennFudanPed/` |
| 2 | Inspect images, raw annotation text, instance masks | `sample_grid.png`, `mask_instances.png` |
| 3 | Convert PASCAL boxes → normalised YOLO boxes, cross-check against mask boxes | `labels/`, `conversion_log.txt` |
| 4 | 70/15/15 image-level split (seed 42; reuses the team manifest so Ayesha's/Sara's numbers stay comparable) | `split_manifest.csv` |
| 5 | 10 automated quality checks | `quality_report.txt` |
| 6 | Statistics + figures | `dataset_stats.csv`, `dataset_stats_plots.png`, `label_check_samples.png` |
| 7 | Dataset loader (with / without augmentation) for the YOLOv1 code | `yolotools.yolo_dataset` |
| 8 | Package for GitHub + a full zip for teammates | Drive |
"""), md("## 0 · Setup"), code(SETUP)]
A += module_cells(["boxes", "penn_fudan", "quality", "viz", "augment", "yolo_dataset"])
A += [
 md("## 1 · Download and integrity check"),
 code('''
from yolotools import penn_fudan as pf
ROOT = pf.download_penn_fudan('/content')
for d in ['PNGImages', 'Annotation', 'PedMasks']:
    print(f"{d:10s}", len(os.listdir(f'{ROOT}/{d}')), 'files')
assert len(os.listdir(f'{ROOT}/PNGImages')) == pf.EXPECTED['images'] == 170
'''),
 md("## 2 · Inspect the raw data\nPenn-Fudan has two sources: `FudanPed*` (Fudan University campus) and `PennPed*` (University of Pennsylvania). The annotation text files use **1-indexed inclusive** pixel boxes; the masks encode every pedestrian with its own integer ID (0 = background)."),
 code('''
print(open(f'{ROOT}/Annotation/FudanPed00001.txt', encoding='latin-1').read())
from yolotools import viz
viz.plot_sample_grid(ROOT, n=6, save='/content/sample_grid.png'); plt.show()
viz.plot_mask_instances(ROOT, 'FudanPed00013', save='/content/mask_instances.png'); plt.show()
'''),
 md("""
## 3 · Convert to YOLO format
For each image: parse the official annotation boxes `(xmin,ymin,xmax,ymax)` (1-indexed, inclusive) → continuous edges `(xmin-1, ymin-1, xmax, ymax)` → normalised `class xc yc w h` (class 0 = pedestrian).  
Every annotation box is **cross-checked against the box derived from the instance mask**; low agreement or a different count is logged as a warning.

**Split.** The team's existing `split_manifest.csv` (image-level 70/15/15, seed 42, 119/25/26 images) is reused when found, because Ayesha's YOLOv1 notebook already trains on it — regenerating would change the splits and make results incomparable. If it cannot be found, an identical-recipe split is generated with seed 42.
"""),
 code('''
# look for the team manifest (GitHub repo first, then a file uploaded next to the notebook)
!rm -rf /content/team_repo && git clone -q --depth 1 https://github.com/ayeshaalamexe/cs320-yolov1.git /content/team_repo 2>&1 | tail -1
cands = ['/content/team_repo/PennFudanYOLO/split_manifest.csv', '/content/split_manifest.csv']
MANIFEST = next((c for c in cands if os.path.exists(c)), None)
print('manifest:', MANIFEST or 'none found -> generating a new seed-42 split')

OUT = '/content/PennFudanYOLO'
records, split, issues, warnings, stats = pf.prepare(ROOT, OUT, manifest_path=MANIFEST, seed=SEED)
print(json.dumps(issues, indent=2))
print('warnings:', len(warnings)); [print(' -', w) for w in warnings[:15]]
'''),
 code('''
import pandas as pd
miou = pd.Series([r['min_mask_iou'] for r in records if r['min_mask_iou'] is not None])
print('lowest annotation-vs-mask IoU per image:\\n', miou.describe().round(3).to_string())
pd.read_csv(f'{OUT}/dataset_stats.csv')
'''),
 md("## 4 · Split manifest\n`split_manifest.csv` lists every image with its split, source set (Fudan/Penn), size and pedestrian count. Splits are by **image**; note that near-identical frames from one scene can land in different splits — a known limitation of image-level splitting on this small dataset."),
 code('''
man = pd.read_csv(f'{OUT}/split_manifest.csv')
print(man.groupby(['split', 'source_set']).size().unstack(fill_value=0))
assert man.filename.is_unique                       # no image twice
man.head()
'''),
 md("## 5 · Quality checks (10 automated checks)"),
 code('''
from yolotools import quality
res = quality.run_quality_checks(OUT, ROOT, expected_images=pf.EXPECTED['images'], expected_instances=pf.EXPECTED['instances'])
print(quality.write_report(res, f'{OUT}/quality_report.txt'))
assert all(r[0] for r in res), 'a quality check FAILED - read the line above'
'''),
 md("## 6 · Statistics and visual label check\nThe label-check figure draws boxes **read back from the saved label files**, i.e. what the models will actually be trained on."),
 code('''
fig, extra = viz.plot_statistics(OUT, save=f'{OUT}/dataset_stats_plots.png'); plt.show()
print(extra)
viz.plot_label_check(OUT, n=6, seed=1, save=f'{OUT}/label_check_samples.png'); plt.show()
shutil.copy('/content/sample_grid.png', OUT); shutil.copy('/content/mask_instances.png', OUT)
'''),
 md("""
## 7 · Dataset loader for the YOLOv1 code
`YoloFolderDataset` returns `(image, boxes)` exactly like the team's `PennFudanDataset` (image `3×448×448` in 0-1, boxes `[xc,yc,w,h]` normalised) so it plugs into the existing `collate_fn` and `encode_targets`. `augment=True` applies box-aware HSV jitter, horizontal flip and random scale/translate (YOLOv1-paper style) — this is the switch for the augmentation ablation on the custom YOLOv1.
"""),
 code('''
import torch
from torch.utils.data import DataLoader
from yolotools.yolo_dataset import YoloFolderDataset, collate_fn
plain = YoloFolderDataset(OUT, 'train', 448)
aug   = YoloFolderDataset(OUT, 'train', 448, augment=True)
loader = DataLoader(aug, batch_size=8, shuffle=True, collate_fn=collate_fn)
x, b = next(iter(loader)); print('batch', tuple(x.shape), '| boxes in img0', tuple(b[0].shape))

import matplotlib.patches as patches
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for k in range(5):
    for row, ds in enumerate((plain, aug)):
        arr, bx, _ = ds.sample(k * 7)
        ax = axes[row, k]; ax.imshow(arr); ax.axis('off'); ax.set_title(('plain' if row == 0 else 'augmented') + f' #{k*7}')
        for xc, yc, w, h in bx:
            ax.add_patch(patches.Rectangle(((xc - w/2) * 448, (yc - h/2) * 448), w * 448, h * 448, lw=2, ec='lime', fc='none'))
plt.tight_layout(); plt.show()
'''),
 md("## 8 · Package for GitHub and for teammates"),
 code('''
GH = '/content/github_ready/PennFudanYOLO'           # small files only (images are re-creatable by this notebook)
shutil.rmtree('/content/github_ready', ignore_errors=True); os.makedirs(GH)
for f in ['split_manifest.csv', 'dataset_stats.csv', 'data.yaml', 'README_dataset.md', 'quality_report.txt', 'conversion_log.txt',
          'sample_grid.png', 'mask_instances.png', 'label_check_samples.png', 'dataset_stats_plots.png']:
    shutil.copy(f'{OUT}/{f}', GH)
shutil.copytree(f'{OUT}/labels', f'{GH}/labels')
shutil.make_archive('/content/PennFudanYOLO_full', 'zip', '/content', 'PennFudanYOLO')   # images + labels for Ayesha / Sara
if DRIVE:
    shutil.copy('/content/PennFudanYOLO_full.zip', DRIVE)
    shutil.copytree('/content/github_ready', f'{DRIVE}/github_ready', dirs_exist_ok=True)
    print('saved to Drive:', DRIVE)
print(os.path.getsize('/content/PennFudanYOLO_full.zip') // 1024, 'KB full zip')
print(open(f'{GH}/data.yaml').read())
'''),
 md("**Done with Part A.** Copy `github_ready/PennFudanYOLO/*` into the repo's `PennFudanYOLO/` folder (see `docs/GITHUB_PUSH_GUIDE.md`). Share `PennFudanYOLO_full.zip` with teammates; in Colab they only need to unzip it and use `data.yaml`.")]
save("Hassaan_A_PennFudan_Data_Lead.ipynb", A)

# =============================================================== Notebook B: VOC2007 + NMS + mAP
B = [md("""
# CS-320 · YOLOv1 project · Hassaan — Part B: VOC 2007 loader · NMS · mAP@0.5
**This week's task (due 20 Nov):** *VOC 2007 loader* and *mAP@0.5 and NMS*, used by Ayesha's YOLOv1 reproduction (7×7×30 output, 448 px).

| Section | Content |
|---|---|
| 1 | Download VOC2007 (trainval 5011 + test 4952 images), integrity + class statistics |
| 2 | `VOC2007Detection` loader → image `3×448×448` + target `7×7×25` (`[x,y,w,h,obj,20 classes]`), difficult handling, augmentation, visual check of the encoder |
| 3 | NMS (plain + class-aware) + YOLOv1 output decoder — verified against `torchvision.ops` |
| 4 | mAP@0.5 (VOC07 11-point and all-point), per-class AP, P/R at a score threshold, mAP@[.5:.95] — verified with hand-computed cases |
| 5 | End-to-end sanity on the real VOC test set with synthetic detections (no model needed) |
| 6 | How to evaluate the trained YOLOv1 with one call |
"""), md("## 0 · Setup"), code(SETUP)]
B += module_cells(["boxes", "augment", "voc2007", "nms", "metrics", "evaluate"])
B += [
 md("## 1 · Download VOC 2007 and check it"),
 code('''
from yolotools import voc2007 as voc
DEVKIT = voc.download_voc2007('/content/data')
for s in ['trainval', 'test']:
    n = len(voc.read_image_ids(DEVKIT, s))
    print(f'{s:9s} {n} images (expected {voc.EXPECTED_IMAGES[s]})')
    assert n == voc.EXPECTED_IMAGES[s]
'''),
 code('''
import pandas as pd
st = {s: voc.dataset_statistics(DEVKIT, s) for s in ['trainval', 'test']}
df = pd.DataFrame({'class': voc.VOC_CLASSES,
                   'trainval_obj': st['trainval']['objects'], 'trainval_imgs': st['trainval']['images_with_class'],
                   'test_obj': st['test']['objects'], 'test_imgs': st['test']['images_with_class'],
                   'test_non_difficult': st['test']['non_difficult']})
print(df.to_string(index=False)); print(df.drop(columns='class').sum())
df.set_index('class')[['trainval_obj', 'test_obj']].plot.bar(figsize=(12, 3.5), title='VOC2007 objects per class'); plt.show()
'''),
 md("""
## 2 · The loader
* `VOC2007Detection(devkit, 'trainval'|'test', img_size=448, S=7, augment=..., keep_difficult=...)`
* `__getitem__` → `(image 3×448×448 float 0-1, target 7×7×25)`; target layout per cell `[x_off, y_off, w, h, obj, one-hot(20)]` — identical to Ayesha's `encode_targets` (so the same loss works with `C=20`; the network output is `7×7×(2·5+20)=7×7×30`).
* Boxes: VOC's 1-indexed inclusive pixels → continuous edges, normalised by image size. Images are resized directly to 448×448 (as in the paper).
* Training excludes `difficult` objects (darknet convention). `ground_truth(i)` returns **all** objects with a `difficult` flag for evaluation.
* One object per cell is a YOLOv1 limitation; on a collision the first object is kept (`collision='largest'` keeps the bigger one).
* `augment=True`: HSV jitter (×1.5), horizontal flip, random scale/translate (±20 %), all box-aware. Evaluation sets must use `augment=False`.
"""),
 code('''
import torch, matplotlib.patches as patches
from torch.utils.data import DataLoader
tv_plain = voc.VOC2007Detection(DEVKIT, 'trainval', 448)
tv_aug   = voc.VOC2007Detection(DEVKIT, 'trainval', 448, augment=True)
x, t = next(iter(DataLoader(tv_aug, batch_size=8, shuffle=True)))
print('image batch', tuple(x.shape), x.dtype, '| target batch', tuple(t.shape), '| objects in batch', int(t[..., 4].sum()))

def show(ds, ids, title):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, i in zip(axes.flat, ids):
        arr, bx, lb = ds.sample(i)
        tgt = voc.encode_yolov1_target(bx, lb, 7, 20)
        db, dl = voc.decode_target(tgt)                       # draw what the TARGET encodes (checks the encoder)
        ax.imshow(arr); ax.axis('off')
        for (xc, yc, w, h), c in zip(db, dl):
            ax.add_patch(patches.Rectangle(((xc - w/2) * 448, (yc - h/2) * 448), w * 448, h * 448, lw=2, ec='lime', fc='none'))
            ax.text((xc - w/2) * 448, (yc - h/2) * 448 - 3, voc.VOC_CLASSES[c], color='black', fontsize=9, bbox=dict(fc='lime', ec='none', pad=1))
        ax.set_title(f'{ds.ids[i]}  ({len(lb)} objs, {len(db)} in target)')
    fig.suptitle(title); plt.tight_layout(); plt.show()
show(tv_plain, range(8), 'No augmentation - boxes decoded from the 7x7x25 target')
show(tv_aug, [0, 1, 2, 3, 0, 1, 2, 3], 'Augmentation ON - the same 4 images twice (different random draws)')
'''),
 code('''
# how many objects does the one-object-per-cell rule lose? (useful number for the report)
lost = tot = 0
for i in range(0, len(tv_plain), 10):
    _, bx, lb = tv_plain.sample(i)
    tot += len(lb); lost += len(lb) - int(voc.encode_yolov1_target(bx, lb, 7, 20)[..., 4].sum())
print(f'sampled {tot} training objects: {lost} ({100*lost/tot:.1f}%) cannot be encoded (two centres in one 7x7 cell)')
'''),
 md("""
## 3 · Non-Maximum Suppression
`nms(boxes, scores, iou_thr)`: visit boxes by descending score, keep one, suppress every remaining box with IoU **> thr**. `batched_nms` does this per class. `postprocess_yolov1(pred)` decodes one `7×7×30` output (score = box-confidence × class-probability), thresholds and applies class-aware NMS.
"""),
 code('''
from yolotools.nms import nms, batched_nms, postprocess_yolov1, decode_yolov1
rng = np.random.default_rng(0)
# visual demo: 3 true objects, each detected 4-6 times with jitter
truth = np.array([[20, 20, 100, 120], [130, 40, 200, 140], [60, 150, 150, 230]], float)
boxes, scores = [], []
for t in truth:
    for _ in range(rng.integers(4, 7)):
        boxes.append(t + rng.normal(0, 5, 4)); scores.append(rng.uniform(0.3, 0.99))
boxes, scores = np.array(boxes), np.array(scores)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, thr in zip(axes, [None, 0.3, 0.5, 0.7]):
    keep = np.arange(len(boxes)) if thr is None else nms(boxes, scores, thr)
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=2, ec='C3', fc='none')); ax.text(x1, y1, f'{scores[i]:.2f}', color='C3', fontsize=8)
    ax.set_xlim(0, 240); ax.set_ylim(250, 0); ax.set_title('raw: %d boxes' % len(boxes) if thr is None else f'NMS IoU>{thr}: {len(keep)} kept')
plt.show()
'''),
 code('''
# independent verification against torchvision (200 random problems, exact same kept indices)
import torch, torchvision
bad = 0
for seed in range(200):
    g = np.random.default_rng(seed); n = int(g.integers(1, 120))
    xy = g.uniform(0, 100, (n, 2)); wh = g.uniform(1, 40, (n, 2)); b = np.concatenate([xy, xy + wh], 1); s = g.uniform(0, 1, n)
    lab = g.integers(0, 5, n); thr = float(g.choice([0.2, 0.4, 0.5, 0.7]))
    tb, ts, tl = torch.tensor(b), torch.tensor(s), torch.tensor(lab)
    bad += nms(b, s, thr).tolist() != torchvision.ops.nms(tb, ts, thr).tolist()
    bad += batched_nms(b, s, lab, thr).tolist() != torchvision.ops.batched_nms(tb, ts, tl, thr).tolist()
print('mismatches vs torchvision:', bad); assert bad == 0
'''),
 md("""
## 4 · mAP@0.5
`evaluate_detections(preds, gts, num_classes, iou_thr=0.5, use_07_metric=True)` follows the VOC devkit: detections sorted by score; each takes the max-IoU ground truth of its image and class; `IoU ≥ thr` and unmatched → TP, already matched → FP (duplicate), matched to a `difficult` GT → ignored; recall's denominator counts non-difficult GTs only. `use_07_metric=True` is the official VOC2007 **11-point** AP (the number comparable with the YOLO paper's 63.4 %); `False` gives the all-point area.
"""),
 code('''
from yolotools.metrics import (evaluate_detections, map_over_iou_thresholds, precision_recall_at_threshold,
                               sweep_score_thresholds, format_ap_table)
P = lambda b, s, l: dict(boxes=np.array(b, float).reshape(-1, 4), scores=np.array(s, float), labels=np.array(l))
G = lambda b, l, d=None: dict(boxes=np.array(b, float).reshape(-1, 4), labels=np.array(l), **({} if d is None else dict(difficult=np.array(d))))
# hand-computed case: 3 GT; detections ranked TP, FP, TP
gts = [G([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]], [0, 0, 0])]
preds = [P([[0, 0, 10, 10], [100, 100, 110, 110], [20, 0, 30, 10]], [.9, .8, .7], [0, 0, 0])]
a = evaluate_detections(preds, gts, 1, 0.5, use_07_metric=False)['mAP']; b = evaluate_detections(preds, gts, 1, 0.5, use_07_metric=True)['mAP']
print('all-point AP', a, '(hand: 1/3 + 1/3*2/3 =', 1/3 + 2/9, ')   11-point AP', b, '(hand:', (4 + 3*2/3)/11, ')')
assert abs(a - (1/3 + 2/9)) < 1e-12 and abs(b - (4 + 2) / 11) < 1e-12
# duplicate -> FP ; difficult -> ignored
assert evaluate_detections([P([[0,0,10,10]]*2, [.9, .8], [0, 0])], [G([[0,0,10,10]], [0])], 1, .5, False)['precision'][0] == 0.5
assert evaluate_detections([P([[0,0,10,10], [20,20,30,30]], [.9, .8], [0, 0])], [G([[0,0,10,10], [20,20,30,30]], [0, 0], [False, True])], 1, .5, False)['mAP'] == 1.0
print('metric unit checks passed')
'''),
 md("## 5 · End-to-end sanity on the real VOC test set (synthetic detections, no model)\nBuild detections from the ground truth: every object is detected with prob. 0.8 with box noise, ~50 % get a duplicate, and ~2 random false positives per image are added. Expected behaviour: ideal detections → mAP = 100 %; noisy → lower; **NMS removes the duplicates and raises mAP**."),
 code('''
test = voc.VOC2007Detection(DEVKIT, 'test', 448)
gts = [test.ground_truth(i) for i in range(len(test))]
rng = np.random.default_rng(0)

def fake_detections(gts, p_det=0.8, sigma=0.04, p_dup=0.5, n_fp=2):
    out = []
    for g in gts:
        B, S, L = [], [], []
        for b, l in zip(g['boxes'], g['labels']):
            if rng.random() > p_det: continue
            wh = np.array([b[2]-b[0], b[3]-b[1]] * 2)
            for k in range(1 + (rng.random() < p_dup)):
                B.append(b + rng.normal(0, sigma, 4) * wh); S.append(rng.uniform(.5, 1) if k == 0 else rng.uniform(.3, .9)); L.append(l)
        for _ in range(n_fp):
            c = rng.uniform(0.1, 0.8, 2); B.append(np.r_[c, c + rng.uniform(.05, .3, 2)]); S.append(rng.uniform(0, .6)); L.append(rng.integers(0, 20))
        B = np.clip(np.array(B).reshape(-1, 4), 0, 1)
        out.append(dict(boxes=B, scores=np.array(S), labels=np.array(L)))
    return out

perfect = [dict(boxes=g['boxes'], scores=np.ones(len(g['labels'])), labels=g['labels']) for g in gts]
r = evaluate_detections(perfect, gts, 20, 0.5, True, voc.VOC_CLASSES); print('GT used as detections -> mAP', r['mAP']); assert abs(r['mAP'] - 1) < 1e-9
noisy = fake_detections(gts)
after = [dict(zip(('boxes', 'scores', 'labels'), (d['boxes'][k], d['scores'][k], d['labels'][k]))) for d in noisy for k in [batched_nms(d['boxes'], d['scores'], d['labels'], 0.5)]]
for name, d in [('noisy, no NMS', noisy), ('noisy + NMS(0.5)', after)]:
    print(f"{name:18s} mAP@0.5  07-11pt {evaluate_detections(d, gts, 20, .5, True)['mAP']*100:5.2f}%   all-point {evaluate_detections(d, gts, 20, .5, False)['mAP']*100:5.2f}%   #dets {sum(len(x['scores']) for x in d)}")
'''),
 code('''
res = evaluate_detections(after, gts, 20, 0.5, True, voc.VOC_CLASSES); print(format_ap_table(res))
fig, ax = plt.subplots(1, 3, figsize=(18, 4))
ax[0].bar(voc.VOC_CLASSES, res['ap']); ax[0].tick_params(axis='x', rotation=90); ax[0].set_title('AP per class (VOC07 11-pt)')
r_, p_ = res['curves'][voc.CLASS_TO_ID['person']]; ax[1].plot(r_, p_); ax[1].set_xlabel('recall'); ax[1].set_ylabel('precision'); ax[1].set_title('PR curve - person'); ax[1].set_ylim(0, 1.05)
m = map_over_iou_thresholds(after, gts, 20); ks = [k for k in m if k != 'mean']; ax[2].plot(ks, [m[k] for k in ks], 'o-'); ax[2].set_title(f"mAP vs IoU threshold (mean .5:.95 = {m['mean']:.3f})"); ax[2].set_xlabel('IoU')
plt.tight_layout(); plt.show()
sw = pd.DataFrame(sweep_score_thresholds(after, gts, 20, [0.0, 0.1, 0.3, 0.5, 0.7, 0.9])); print(sw.round(3).to_string(index=False))
'''),
 md("""
## 6 · Evaluating the trained YOLOv1 (for Ayesha's model)
`evaluate_yolov1` runs the model over a dataset, decodes `7×7×30` outputs, applies NMS and returns mAP. Use a **low** score threshold (0.01) for mAP, and the dataset with `augment=False`.

```python
from yolotools.evaluate import evaluate_yolov1
voc_test = voc.VOC2007Detection(DEVKIT, 'test', 448)                 # augment=False
res, preds, gts = evaluate_yolov1(model, voc_test, B=2, C=20, device='cuda',
                                  score_thr=0.01, nms_iou=0.5, iou_thr=0.5, use_07_metric=True)
print(res['mAP'])
```
For Penn-Fudan (1 class): `YoloFolderDataset(OUT, 'test')` with `C=1` (the model must output `7×7×(2·5+1)`), `use_07_metric=False`.  
The guarded cell below runs only if a variable named `model` exists in the notebook.
"""),
 code('''
if 'model' in globals():
    from yolotools.evaluate import evaluate_yolov1
    res, preds, gts_ = evaluate_yolov1(model, voc.VOC2007Detection(DEVKIT, 'test', 448), B=2, C=20, device='cuda')
else:
    print("no `model` defined here - paste Ayesha's YOLOv1 class + checkpoint loading above to evaluate it")
'''),
 md("**Done with Part B.** Everything above is also covered by the 97 automated tests in `tests/` (`python -m pytest tests`), including a randomised cross-check of the AP code against a classic devkit-style reference implementation.")]
save("Hassaan_B_VOC2007_NMS_mAP.ipynb", B)

# =============================================================== Notebook C: augmentation ablation
C = [md("""
# CS-320 · YOLOv1 project · Hassaan — Part C: Augmentation ablation (RQ2)
**Question:** *How does data augmentation affect performance when training on Penn-Fudan?*  
**Design.** Fine-tune the same Ultralytics YOLO model on the same data with identical settings and seeds; the **only** difference is the augmentation hyper-parameters:

* `no_aug` — HSV, flips, scale, translate, shear, rotation, perspective, mosaic, mixup, copy-paste, erasing all set to 0
* `default_aug` — Ultralytics defaults

Each variant is run with 3 seeds (26 test images → single runs are very noisy). The checkpoint is chosen on **val**; the **test** split is evaluated once per run. Never tune on test.
"""), md("## 0 · Setup"), code(SETUP),
 code("!pip install -q ultralytics\nimport torch, ultralytics\nprint('ultralytics', ultralytics.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE - enable Runtime > Change runtime type > GPU')")]
C += module_cells(["boxes", "penn_fudan", "ablation"])
C += [
 md("## 1 · Get the dataset (from Part A)\nOrder tried: folder already in `/content` → `PennFudanYOLO_full.zip` on Drive → rebuild from the original download using the team manifest."),
 code('''
import glob, subprocess
OUT = '/content/PennFudanYOLO'
def n_images(p): return sum(len(glob.glob(f'{p}/images/{s}/*.png')) for s in ('train', 'val', 'test'))
if not (os.path.isdir(OUT) and n_images(OUT) == 170):
    zp = f'{DRIVE}/PennFudanYOLO_full.zip' if DRIVE else None
    if zp and os.path.exists(zp):
        shutil.unpack_archive(zp, '/content'); print('unzipped from Drive')
    else:
        from yolotools import penn_fudan as pf
        subprocess.run('rm -rf /content/team_repo && git clone -q --depth 1 https://github.com/ayeshaalamexe/cs320-yolov1.git /content/team_repo', shell=True)
        man = '/content/team_repo/PennFudanYOLO/split_manifest.csv'
        pf.prepare(pf.download_penn_fudan('/content'), OUT, man if os.path.exists(man) else None); print('rebuilt dataset')
assert n_images(OUT) == 170, n_images(OUT)
# make data.yaml point to this machine's folder
yml = open(f'{OUT}/data.yaml').read().splitlines()
yml = [f'path: {OUT}' if l.startswith('path:') else l for l in yml]
open(f'{OUT}/data.yaml', 'w').write('\\n'.join(yml) + '\\n'); DATA = f'{OUT}/data.yaml'
print(open(DATA).read())
'''),
 md("## 2 · Configuration\nKeep these identical for every variant. 50 epochs ≈ a few minutes per run on a Colab T4."),
 code('''
from yolotools import ablation
MODEL    = 'yolo11n.pt'         # COCO-pretrained, fine-tuned (matches the proposal). Use the .yaml name for a from-scratch run.
EPOCHS, IMGSZ, BATCH = 50, 640, 16
SEEDS    = (0, 1, 2)
RUN_EXTRA = False               # True also runs flip_only / hsv_only / geometric_only / mosaic_only ("one at a time")
VARIANTS = dict(ablation.VARIANTS)
if RUN_EXTRA: VARIANTS.update(ablation.EXTRA_VARIANTS)
RES_DIR  = f'{DRIVE}/ablation' if DRIVE else '/content/ablation'; os.makedirs(RES_DIR, exist_ok=True)
CSV      = f'{RES_DIR}/aug_ablation_runs_{MODEL.split(".")[0]}.csv'
print({k: ablation.supported(v) for k, v in VARIANTS.items()})
'''),
 md("### Smoke test (1 epoch, 1 run) - catches setup problems in ~1 minute"),
 code('''
_ = ablation.run_one('smoke', ablation.AUG_OFF, 0, DATA, MODEL, epochs=1, imgsz=IMGSZ, batch=BATCH, project='/content/runs/smoke')
print({k: round(v, 3) if isinstance(v, float) else v for k, v in _.items()})
'''),
 md("## 3 · Run the ablation\nResults are appended to a CSV on Drive after every run, so if Colab disconnects just re-run this cell — finished runs are skipped."),
 code('''
rows = ablation.run_ablation(DATA, VARIANTS, SEEDS, out_csv=CSV, model_name=MODEL, epochs=EPOCHS, imgsz=IMGSZ,
                             batch=BATCH, project='/content/runs/aug_ablation')
import pandas as pd
df = pd.read_csv(CSV); df
'''),
 md("## 4 · Results"),
 code('''
summary = ablation.summarise(df.to_dict('records'))
table = ablation.markdown_table(summary); print(table)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for a, m in zip(ax, ['test_map50', 'test_map50_95']):
    names = list(summary); a.bar(names, [summary[v][m][0] for v in names], yerr=[summary[v][m][1] for v in names], capsize=6, color=['#999', '#4363d8', '#3cb44b', '#f58231', '#911eb4', '#008080'][:len(names)])
    for k, v in enumerate(names):
        pts = df[df.variant == v][m]; a.scatter([k] * len(pts), pts, color='k', zorder=3, s=18)
    a.set_title(m + ' (bar = mean ± sd, dots = seeds)'); a.set_ylim(max(0, df[m].min() - .05), min(1, df[m].max() + .03)); a.tick_params(axis='x', rotation=20)
plt.tight_layout(); plt.savefig(f'{RES_DIR}/aug_ablation_bars.png', dpi=120); plt.show()
'''),
 code('''
# learning curves (val mAP50 per epoch, seed 0) - shows over-fitting / convergence differences
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for v in VARIANTS:
    p = f'/content/runs/aug_ablation/{v}_seed0/results.csv'
    if not os.path.exists(p): continue
    r = pd.read_csv(p); r.columns = [c.strip() for c in r.columns]
    ax[0].plot(r['epoch'], r['metrics/mAP50(B)'], label=v); ax[1].plot(r['epoch'], r['val/box_loss'], label=v)
ax[0].set_title('val mAP50 vs epoch'); ax[1].set_title('val box loss vs epoch'); ax[0].legend(); ax[0].set_xlabel('epoch'); ax[1].set_xlabel('epoch')
plt.tight_layout(); plt.savefig(f'{RES_DIR}/aug_ablation_curves.png', dpi=120); plt.show()
'''),
 code('''
# qualitative: same 4 test images, predictions of the best checkpoint of each variant (seed 0)
from ultralytics import YOLO
from PIL import Image
test_imgs = sorted(glob.glob(f'{OUT}/images/test/*.png'))[:4]
fig, axes = plt.subplots(len(VARIANTS), 4, figsize=(18, 4.2 * len(VARIANTS)), squeeze=False)
for r_, v in enumerate(VARIANTS):
    w = f'/content/runs/aug_ablation/{v}_seed0/weights/best.pt'
    if not os.path.exists(w): continue
    m = YOLO(w)
    for c_, p in enumerate(test_imgs):
        axes[r_, c_].imshow(m.predict(p, imgsz=IMGSZ, conf=0.25, verbose=False)[0].plot()[..., ::-1]); axes[r_, c_].axis('off'); axes[r_, c_].set_title(f'{v} | {os.path.basename(p)}', fontsize=9)
plt.tight_layout(); plt.savefig(f'{RES_DIR}/aug_ablation_predictions.png', dpi=110); plt.show()
'''),
 md("## 5 · Auto-generated report text\nNumbers are filled in from the CSV. The comparison sentence is deliberately cautious: with 26 test images / 73 pedestrians and 3 seeds, a difference smaller than the seed-to-seed spread is **not** evidence of an effect."),
 code('''
m = 'test_map50'; a, b = summary['default_aug'][m], summary['no_aug'][m]
diff, noise = a[0] - b[0], max(a[1], b[1])
verdict = ('larger than' if abs(diff) > 2 * noise else 'comparable to' if abs(diff) > noise else 'within') + ' the seed-to-seed standard deviation'
text = f"""# Augmentation ablation (RQ2) - Penn-Fudan, {MODEL}, {EPOCHS} epochs, imgsz {IMGSZ}, batch {BATCH}, seeds {list(SEEDS)}

All variants: same pretrained weights, data split (119/25/26 images), epochs, image size, batch size and seeds; only augmentation
arguments differ. Best checkpoint chosen on val; test evaluated once per run (conf=0.001, NMS IoU=0.6).

{table}

mean ± sample sd over {a[2]} seeds. On test mAP@0.5, default augmentation scores {a[0]:.3f} versus {b[0]:.3f} without
augmentation (difference {diff:+.3f}), which is {verdict} ({noise:.3f}).
Per-run numbers: aug_ablation_runs CSV. Figures: aug_ablation_bars.png, aug_ablation_curves.png, aug_ablation_predictions.png.

Interpretation notes (check them against the figures before writing the report):
* A COCO-pretrained detector already knows 'person', so Penn-Fudan can sit close to a performance ceiling and hide augmentation effects; the from-scratch run (MODEL = 'yolo11n.yaml') is the stricter test of RQ2.
* Compare val-loss curves: if no_aug's validation loss rises while training loss falls, that is over-fitting that augmentation is regularising.
* The test split is tiny (26 images); report seeds, not a single run.
"""
open(f'{RES_DIR}/aug_ablation_report.md', 'w').write(text); print(text)
'''),
 md("**Optional, stricter run.** Set `MODEL = 'yolo11n.yaml'` (from scratch), `EPOCHS = 150`, re-run sections 2-5 — this writes a separate CSV and shows augmentation's effect without the COCO prior.\n\nCopy `ablation/*.md/png/csv` into the repo's `results/` folder.")]
save("Hassaan_C_Augmentation_Ablation.ipynb", C)


# Write all three generated sections into the single deliverable.
master_path = os.path.join(HERE, "notebooks", "Hassaan_Adil_Data_Lead.ipynb")
with open(master_path, encoding="utf-8") as f:
    master = json.load(f)
markers = ("# Notebook A", "# Notebook B", "# Notebook C")
base = master["cells"]
for i, cell in enumerate(base):
    source = "".join(cell.get("source", []))
    if any(source.startswith(marker) for marker in markers):
        base = base[:i]
        break
for old_name, title in [("Hassaan_A_PennFudan_Data_Lead.ipynb", "Notebook A: Penn-Fudan data pipeline"),
                        ("Hassaan_B_VOC2007_NMS_mAP.ipynb", "Notebook B: VOC 2007, NMS, and mAP"),
                        ("Hassaan_C_Augmentation_Ablation.ipynb", "Notebook C: augmentation ablation")]:
    base.append(md("---\n\n# " + title + "\n\nConsolidated source section.\n"))
    base.extend(BUILT[old_name])
master["cells"] = base
with open(master_path, "w", encoding="utf-8") as f:
    json.dump(master, f, ensure_ascii=False, indent=1)
print("wrote", os.path.relpath(master_path, HERE), len(base), "cells")
