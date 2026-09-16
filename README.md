# YOLO26 OBB with a third output: boxes, classes, quality

Stock YOLO26 OBB supervises two things you annotate: the oriented **box** and the
**class**. This repo adds a third, **quality** — one scalar per object, predicted
per anchor, trained from the labels, and returned next to every detection.

The third output is built the same way the head builds the outputs it already
has. No new architecture block, no second network, no extra backbone: it is one
more branch of the existing detection head, reading the same P3/P4/P5 features
and flowing through the same decode, top-k, export and NMS path as the box and
the class scores.

## What was added

| Piece | File | What it does |
|---|---|---|
| Head | `obbq/head.py` | `OBB26Quality` — adds a `cv5` branch to `OBB26` |
| Loss | `obbq/loss.py` | `OBBQualityLoss` — adds a 5th term, `qual_loss` |
| Data | `obbq/data.py` | Quality-aware label parsing and dataset |
| Wiring | `obbq/model.py` | Model, trainer, validator, predictor |
| Config | `obbq/cfg/yolo26-obb-quality.yaml` | Stock `yolo26-obb.yaml` with head args `[nc, 1, 1]` |

### 1. The head branch

`ultralytics.nn.modules.head.OBB26` already has a template for "one more
per-anchor output": its angle branch, `cv4`. Per feature level it is

```python
nn.Sequential(Conv(x, c, 3), Conv(c, c, 3), nn.Conv2d(c, n_out, 1))
```

`OBB26Quality` builds `cv5` with exactly that pattern and plugs it into the
head's own extension points:

* `one2many` / `one2one` gain a `quality_head` entry, so the dual-head
  end-to-end training path drives it like every other branch, and the stock
  `fuse()` strips the unused copy automatically (the `one2one_` name prefix is
  all it keys on).
* `forward_head` concatenates the per-level logits into `preds["quality"]`,
  shape `(batch, nq, anchors)` — the same layout as `preds["angle"]`.
* `_inference` appends `quality.sigmoid()` to the decoded tensor.

The inference tensor per anchor is therefore

```
[ x, y, w, h,  cls_0..cls_{nc-1},  angle,  quality ]
```

and after the head's `postprocess` (YOLO26 is NMS-free) each detection is

```
[ x, y, w, h,  conf,  class_id,  angle,  quality ]
```

Because quality is appended *after* the existing channels, `Detect.postprocess`
gathers it with the same top-k indices as everything else, and the conf-filter
that stands in for NMS on end-to-end models is column-agnostic, so it passes
through untouched.

### 2. The label format

One extra column on the standard DOTA-style OBB line:

```
cls x1 y1 x2 y2 x3 y3 x4 y4 quality
```

Coordinates stay normalized to `[0, 1]`; `quality` is a scalar in `[0, 1]`. A
9-column line (plain OBB) is still accepted and defaults to `quality = 1.0`, so
an existing OBB dataset trains without being touched.

```
1 0.531934 0.410224 0.331685 0.683337 0.263790 0.633556 0.464039 0.360443 0.538006
                                                                       ^^^^^^^^^^
                                                                        quality
```

Migrate an existing dataset with `tools/add_quality_column.py`, or generate a
synthetic one with `tools/make_dataset.py`.

### 3. Carrying quality through augmentation

Quality is stored as a **second column of `cls`**, so `cls` is `(n, 2)`:
`[class_index, quality]`.

This is the part that makes the change small. Mosaic, mixup, cutmix,
copy-paste, random perspective and letterbox all drop, clip, reorder and
concatenate instances — and every one of them filters `cls` by the same indices
it uses for the boxes. Riding in `cls` means quality follows its own object
through all of them with **zero changes to any transform**. `collate_fn`
already concatenates `cls` along dim 0, so batching is free too.

Only two seams needed attention, both handled by subclasses:

* `verify_image_label_quality` parses the 10th column before the polygon is
  reshaped (`obbq/data.py`).
* `QualityFormat` restores the `(0, 2)` shape on images with no objects, where
  stock `Format` emits `zeros(0, 1)`.

The loss splits the two columns again via `OBBQualityLoss.split_cls`.

### 4. The loss

`OBBQualityLoss` extends `v8OBBLoss` with a fifth term. Quality is carried as a
trailing column of the target tensor so it survives the small-box filtering and
per-image padding in lockstep with the boxes, then:

```python
target_quality = gt_quality.squeeze(-1).gather(1, target_gt_idx)   # assigner's match
qual_loss = self.bce(pred_quality.squeeze(-1), target_quality)[fg_mask]
return (qual_loss * weight).sum() / target_scores_sum
```

An anchor's quality target is the quality of the ground-truth object the
task-aligned assigner matched it to. So the third output is supervised on
exactly the positives the first two are, with the same alignment weighting and
the same `BCEWithLogitsLoss` the class output already uses (soft targets — the
optimum of `BCE(sigmoid(z), q)` is `sigmoid(z) = q`).

Reported losses become `box_loss cls_loss l1_loss angle_loss qual_loss`.

## Usage

```bash
pip install ultralytics

# 1. a dataset in the quality-aware format
python tools/make_dataset.py --root datasets/obb-quality --train 240 --val 48

# 2. train
python train_quality.py --data datasets/obb-quality/data.yaml --epochs 80 --imgsz 320

# 3. predict, with all three outputs
python predict_quality.py --weights runs/obb/trial/weights/best.pt \
    --source datasets/obb-quality/images/val
```

From Python:

```python
import obbq

obbq.register()                          # enables the `quality` loss gain
model = obbq.YOLOQuality(obbq.model_cfg("n"))
model.train(data="datasets/obb-quality/data.yaml", epochs=80, imgsz=320, quality=1.0)

result = model.predict("image.jpg")[0]
result.obb.xywhr        # output 1: oriented boxes
result.obb.cls          # output 2: classes
result.quality          # output 3: quality, one value per detection
```

Validation prints the stock box/class metrics plus `quality: MAE …, corr …`,
and `results.csv` gains `metrics/quality_mae` and `metrics/quality_corr`.

`obbq.model_cfg(scale)` accepts `n`, `s`, `m`, `l`, `x`.

## Trial result

100 epochs on the synthetic set from `tools/make_dataset.py` (240 train / 48 val
images, 2 classes, `yolo26n` scale, imgsz 320, CPU, ~30 min), where an object's
annotated quality is its contrast against the background:

```
               Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                 all         48        115      0.929      0.869      0.937      0.775
                rect         38         55      0.932      0.909      0.951      0.860
                 bar         33         60      0.926      0.830      0.922      0.690
quality: MAE 0.0767, corr 0.8812, over 101 matched object(s)
```

The third output tracks its label: mean absolute error 0.077 on a 0-1 scale and
a correlation of 0.88 against the annotated quality of matched objects, while
box and class accuracy train normally alongside it. `qual_loss` falls
monotonically with the other four terms (see `runs/obb/trial/results.csv`).

Inference returns all three per detection:

```
val_0002.jpg: 6 detection(s)
  box=( 241.0,  42.6,  33.9,  42.2,-0.28rad)  class=rect  conf=0.954  quality=0.780
  box=( 178.3, 151.1,  29.0, 120.6,+0.43rad)  class=bar   conf=0.948  quality=0.533
  box=( 121.1, 150.0,  34.5, 117.4,+0.66rad)  class=bar   conf=0.887  quality=0.124
```

## Tests

```bash
python -m pytest tests/test_quality.py -v
```

Covers label parsing (10-column, 9-column default, out-of-range rejection), the
head's output layout and YAML arity check, the `cls` split, and that the quality
term is minimized where the prediction equals the annotation.

## Notes and limits

* **Scope.** Built and tested against ultralytics 8.4.153, YOLO26 OBB in its
  default end-to-end (NMS-free) mode. The legacy NMS path assumes the angle is
  the *last* column of a prediction row (`x[:, -1:]` in
  `ultralytics/utils/nms.py`), which the appended quality column would break, so
  keep `nms=False` (the YOLO26 default).
* **Head binding.** `parse_model` resolves head classes by name in
  `ultralytics.nn.tasks` and compares against them by identity, so
  `OBBQualityModel` binds the name `OBB26` to `OBB26Quality` for the duration of
  model construction only. Stock OBB models built before or after are unaffected.
* **Albumentations.** If `albumentations` is installed, ultralytics' wrapper
  reshapes `cls` with `cls[i].reshape(-1, 1)`, which would flatten the packed
  `(n, 2)` array. It is not installed here; with it installed, that one line
  needs the same treatment as the two seams above.
* **`nq`.** The head takes the number of quality channels as its third YAML
  argument, so `[nc, 1, 3]` would predict three per-object attributes instead of
  one. Only `nq = 1` is wired through the loss and the label parser.
* **Label cache.** `QualityOBBDataset` tags its cache hash, so a plain-OBB
  `labels.cache` is never silently reused for quality-aware labels.
