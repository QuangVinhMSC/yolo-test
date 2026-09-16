"""Independently score the defect output against the label files on disk.

This deliberately avoids the training-time validator: it reads the ground truth
straight from the label `.txt` files, runs the predictor, matches each annotated
object to its best-overlapping detection, and prints a confusion matrix. It
exists to check the validator's own numbers, so it shares no code with them.

Example:
    python tools/eval_defect.py --weights runs/obb/defect/weights/best.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.ops import xyxyxyxy2xywhr

import obbq
from obbq.defect import YOLODefect

IOU_MATCH = 0.5


def load_labels(path: Path, w: int, h: int):
    """Return (xywhr boxes in pixels, object classes, defect classes) from one label file."""
    rows = [r.split() for r in path.read_text().strip().splitlines() if r.strip()]
    if not rows:
        return torch.zeros((0, 5)), np.zeros(0, int), np.zeros(0, int)
    poly = torch.tensor([[float(v) for v in r[1:9]] for r in rows]).view(-1, 4, 2)
    poly[..., 0] *= w
    poly[..., 1] *= h
    cls = np.array([int(float(r[0])) for r in rows])
    defect = np.array([int(float(r[9])) for r in rows])
    return xyxyxyxy2xywhr(poly), cls, defect


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="runs/obb/defect/weights/best.pt")
    p.add_argument("--images", default="datasets/obb-defect/images/val")
    p.add_argument("--labels", default="datasets/obb-defect/labels/val")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    obbq.register()
    model = YOLODefect(a.weights)
    names = None
    true_all, pred_all, unmatched = [], [], 0

    for img_path in sorted(Path(a.images).glob("*.jpg")):
        (result,) = model.predict(str(img_path), imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)
        names = names or result.defect_names
        h, w = result.orig_shape
        gt_boxes, gt_cls, gt_defect = load_labels(Path(a.labels) / f"{img_path.stem}.txt", w, h)
        if not len(gt_boxes):
            continue
        if not len(result.obb):
            unmatched += len(gt_boxes)
            continue

        iou = batch_probiou(gt_boxes, result.obb.xywhr.cpu())
        best = iou.argmax(dim=1)
        ok = (iou.gather(1, best[:, None]).squeeze(1) > IOU_MATCH) & (
            result.obb.cls.cpu()[best] == torch.tensor(gt_cls, dtype=torch.float)
        )
        unmatched += int((~ok).sum())
        true_all.extend(gt_defect[ok.numpy()].tolist())
        pred_all.extend(result.defect.cpu()[best][ok].tolist())

    true, pred = np.array(true_all), np.array(pred_all)
    nd = len(names) if names else int(max(true.max(), pred.max()) + 1)
    label = [names.get(i, str(i)) if names else str(i) for i in range(nd)]

    print(f"\nmatched {len(true)} object(s), {unmatched} unmatched (no detection above IoU {IOU_MATCH})")
    print(f"defect accuracy: {(true == pred).mean():.4f}\n")
    print("confusion matrix (rows = annotated, cols = predicted)")
    print(" " * 12 + "".join(f"{n:>10s}" for n in label))
    for i in range(nd):
        row = [int(((true == i) & (pred == j)).sum()) for j in range(nd)]
        print(f"{label[i]:>12s}" + "".join(f"{v:>10d}" for v in row))


if __name__ == "__main__":
    main()
