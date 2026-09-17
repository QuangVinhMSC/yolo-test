"""Split a flat GUI-labeled dataset (images/ + labels/ + data.yaml) into a
train/val list, without moving or duplicating any files.

`tools/label_defect_gui.py` writes everything flat into `images/` and
`labels/` so it can resume labeling at any time. This script only adds
`train.txt` / `val.txt` (one image path per line) next to them and points
`data.yaml`'s `train`/`val` fields at those lists, so re-running it after
labeling more images just recomputes the split.

Usage:
    python tools/split_labeled_dataset.py --root labeled_dataset --val-frac 0.15
"""

import argparse
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="labeled_dataset")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    root = Path(a.root)
    images = sorted((root / "images").glob("*.*"))
    if not images:
        raise SystemExit(f"no images found under {root / 'images'}")

    rng = random.Random(a.seed)
    rng.shuffle(images)
    n_val = max(1, round(len(images) * a.val_frac))
    val, train = images[:n_val], images[n_val:]

    (root / "train.txt").write_text("\n".join(f"./images/{p.name}" for p in sorted(train)) + "\n")
    (root / "val.txt").write_text("\n".join(f"./images/{p.name}" for p in sorted(val)) + "\n")

    yaml_path = root / "data.yaml"
    lines = yaml_path.read_text().splitlines()
    out = []
    for line in lines:
        if line.startswith("train:"):
            out.append("train: train.txt")
        elif line.startswith("val:"):
            out.append("val: val.txt")
        else:
            out.append(line)
    yaml_path.write_text("\n".join(out) + "\n")

    print(f"train: {len(train)} images -> {root / 'train.txt'}")
    print(f"val:   {len(val)} images -> {root / 'val.txt'}")
    print(f"data.yaml updated -> {yaml_path}")


if __name__ == "__main__":
    main()
