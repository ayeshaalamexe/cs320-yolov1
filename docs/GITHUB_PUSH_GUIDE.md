# How to push Hassaan's work to GitHub

1. In Colab run notebook **A**, then **B**, then **C**. Results land in `My Drive/yolo_project/` (`github_ready/`, `ablation/`).
2. On your computer:
```bash
git clone https://github.com/ayeshaalamexe/cs320-yolov1.git && cd cs320-yolov1
git checkout -b hassaan/data-lead
# unzip hassaan_deliverables.zip over the repo folder (adds yolotools/ tests/ scripts/ notebooks/ docs/ README.md requirements.txt .gitignore)
# replace PennFudanYOLO/ metadata + labels with Drive/github_ready/PennFudanYOLO/*  (keep the team's split_manifest.csv unchanged)
mkdir -p results && cp <Drive>/ablation/* results/
python -m pytest tests -q          # expect all green
git add -A && git status           # check: no .pt / .zip / images/ files
git commit -m "Data lead: Penn-Fudan pipeline, VOC2007 loader, NMS, mAP, augmentation ablation"
git push -u origin hassaan/data-lead
```
3. Open a pull request (or push to `main` if the team does that). Split into several commits if you like, e.g.
   `Add Penn-Fudan conversion + quality checks`, `Add VOC2007 loader`, `Add NMS and mAP with tests`, `Add augmentation ablation + results`.
4. Re-save the executed notebooks (File > Save a copy in GitHub, or download .ipynb) so outputs are visible in the repo.
5. Tell Ayesha/Sara: the loaders, `postprocess_yolov1` and `evaluate_yolov1` are ready to use (see README).
