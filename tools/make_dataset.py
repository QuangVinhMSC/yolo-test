"""Generate a small synthetic OBB dataset with a per-object quality label.

Label format (one line per object, the standard DOTA-style OBB line plus one
trailing value):

    cls x1 y1 x2 y2 x3 y3 x4 y4 quality

All polygon coordinates are normalized to [0, 1] exactly like the stock
ultralytics OBB format; `quality` is an extra scalar in [0, 1].

In this toy dataset the quality of an object is its contrast against the
background, so it is something a network can actually learn from pixels.
"""

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["rect", "bar"]


def rotated_box(cx, cy, w, h, angle):
    """Return the 4 corners of a rotated box as a (4, 2) float array."""
    c, s = math.cos(angle), math.sin(angle)
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    rot = np.array([[c, -s], [s, c]])
    return corners @ rot.T + np.array([cx, cy])


def make_sample(rng, size=320):
    """Draw a random image and return it together with its label rows."""
    bg = rng.integers(20, 60)
    img = np.full((size, size, 3), bg, dtype=np.uint8)
    img = cv2.add(img, rng.integers(0, 12, (size, size, 3), dtype=np.uint8))

    rows = []
    for _ in range(rng.integers(1, 5)):
        cls = int(rng.integers(0, len(CLASSES)))
        w = rng.uniform(40, 90) if cls == 0 else rng.uniform(70, 130)
        h = w * (rng.uniform(0.6, 1.0) if cls == 0 else rng.uniform(0.18, 0.35))
        cx = rng.uniform(w, size - w)
        cy = rng.uniform(w, size - w)
        angle = rng.uniform(0, math.pi)

        # Quality == contrast of the object against the background.
        quality = float(rng.uniform(0.05, 0.95))
        fill = int(np.clip(bg + 15 + quality * 190, 0, 255))

        pts = rotated_box(cx, cy, w, h, angle)
        cv2.fillPoly(img, [pts.astype(np.int32)], (fill, fill, fill))

        coords = " ".join(f"{v / size:.6f}" for v in pts.reshape(-1))
        rows.append(f"{cls} {coords} {quality:.6f}")
    return img, rows


def build_split(root: Path, split: str, count: int, seed: int, size: int):
    rng = np.random.default_rng(seed)
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        img, rows = make_sample(rng, size)
        cv2.imwrite(str(img_dir / f"{split}_{i:04d}.jpg"), img)
        (lbl_dir / f"{split}_{i:04d}.txt").write_text("\n".join(rows) + "\n")
    print(f"{split}: {count} images -> {img_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="datasets/obb-quality")
    p.add_argument("--train", type=int, default=64)
    p.add_argument("--val", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    root = Path(a.root).resolve()
    build_split(root, "train", a.train, a.seed, a.imgsz)
    build_split(root, "val", a.val, a.seed + 1, a.imgsz)

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (root / "data.yaml").write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\n\nnames:\n{names}\n"
    )
    print(f"data.yaml -> {root / 'data.yaml'}")


if __name__ == "__main__":
    random.seed(0)
    main()
