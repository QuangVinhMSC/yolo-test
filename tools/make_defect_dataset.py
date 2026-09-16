"""Generate a synthetic OBB dataset with a per-object defect class.

Label format (one line per object, the standard DOTA-style OBB line plus one
trailing integer)::

    cls x1 y1 x2 y2 x3 y3 x4 y4 defect_class

Polygon coordinates are normalized to [0, 1] exactly like the stock ultralytics
OBB format; `defect_class` is an index into the dataset YAML's `defect_names`.

The defect class is independent of the object class: each object is a `rect` or
a `bar`, and carries one of four surface defects drawn onto it, so a network has
to read the object's interior rather than its shape to get the defect right.
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["rect", "bar"]
DEFECTS = ["ok", "scratch", "hole", "crack"]


def rotated_box(cx, cy, w, h, angle):
    """Return the 4 corners of a rotated box as a (4, 2) float array."""
    c, s = math.cos(angle), math.sin(angle)
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return corners @ np.array([[c, -s], [s, c]]).T + np.array([cx, cy])


def draw_defect(img, pts, cx, cy, w, h, angle, defect, rng):
    """Draw the visual signature of `defect` onto the object at (cx, cy)."""
    if defect == 0:  # ok -- nothing added
        return
    c, s = math.cos(angle), math.sin(angle)
    long_axis, short_axis = np.array([c, s]), np.array([-s, c])
    half = max(w, h) / 2 * 0.7
    centre = np.array([cx, cy])

    if defect == 1:  # scratch -- a bright thin line along the object
        off = short_axis * rng.uniform(-0.25, 0.25) * min(w, h)
        p0 = (centre + off - long_axis * half).astype(np.int32)
        p1 = (centre + off + long_axis * half).astype(np.int32)
        cv2.line(img, tuple(p0), tuple(p1), (245, 245, 245), 3)
    elif defect == 2:  # hole -- a dark disc
        r = int(max(5, min(w, h) * 0.30))
        at = centre + long_axis * rng.uniform(-0.4, 0.4) * w
        cv2.circle(img, tuple(at.astype(np.int32)), r, (12, 12, 12), -1)
    else:  # crack -- a dark jagged polyline across the object
        n = 5
        ts = np.linspace(-half, half, n)
        jitter = rng.uniform(-1, 1, n) * min(w, h) * 0.18
        path = np.array([centre + long_axis * t + short_axis * j for t, j in zip(ts, jitter)])
        cv2.polylines(img, [path.astype(np.int32)], False, (18, 18, 18), 3)


def make_sample(rng, size=320):
    """Draw a random image and return it together with its label rows."""
    bg = int(rng.integers(20, 55))
    img = np.full((size, size, 3), bg, dtype=np.uint8)
    img = cv2.add(img, rng.integers(0, 10, (size, size, 3), dtype=np.uint8))

    rows, placed = [], []
    for _ in range(int(rng.integers(1, 4))):
        cls = int(rng.integers(0, len(CLASSES)))
        defect = int(rng.integers(0, len(DEFECTS)))
        w = rng.uniform(60, 100) if cls == 0 else rng.uniform(90, 140)
        h = w * (rng.uniform(0.7, 1.0) if cls == 0 else rng.uniform(0.30, 0.45))
        angle = rng.uniform(0, math.pi)

        # Keep objects apart, so one object's fill never hides another's defect mark
        for _attempt in range(30):
            cx, cy = rng.uniform(w / 2, size - w / 2), rng.uniform(w / 2, size - w / 2)
            if all(math.hypot(cx - px, cy - py) > (w + pw) / 2 for px, py, pw in placed):
                break
        else:
            continue
        placed.append((cx, cy, w))

        pts = rotated_box(cx, cy, w, h, angle)
        cv2.fillPoly(img, [pts.astype(np.int32)], (130, 130, 130))  # uniform fill: only the defect varies
        draw_defect(img, pts, cx, cy, w, h, angle, defect, rng)

        coords = " ".join(f"{v / size:.6f}" for v in pts.reshape(-1))
        rows.append(f"{cls} {coords} {defect}")
    return img, rows


def build_split(root: Path, split: str, count: int, seed: int, size: int):
    """Write one split's images and labels."""
    rng = np.random.default_rng(seed)
    img_dir, lbl_dir = root / "images" / split, root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        img, rows = make_sample(rng, size)
        cv2.imwrite(str(img_dir / f"{split}_{i:04d}.jpg"), img)
        (lbl_dir / f"{split}_{i:04d}.txt").write_text("\n".join(rows) + "\n")
    print(f"{split}: {count} images -> {img_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="datasets/obb-defect")
    p.add_argument("--train", type=int, default=400)
    p.add_argument("--val", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    root = Path(a.root).resolve()
    build_split(root, "train", a.train, a.seed, a.imgsz)
    build_split(root, "val", a.val, a.seed + 1, a.imgsz)

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    defects = "\n".join(f"  {i}: {n}" for i, n in enumerate(DEFECTS))
    (root / "data.yaml").write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\n\nnames:\n{names}\n\ndefect_names:\n{defects}\n"
    )
    print(f"data.yaml -> {root / 'data.yaml'}")


if __name__ == "__main__":
    main()
