"""Remove dataset samples that contribute no box and no class.

Ultralytics silently drops an image/label pair whose labels fail verification --
a polygon corner outside the image, a class index beyond the dataset's classes, a
malformed row -- and it drops the *whole* sample, not just the bad row. It also
treats a missing or empty label file as a background image. Either way the sample
trains nothing, but it stays on disk, and the counts in the results table quietly
stop matching the dataset.

This removes those samples. It does not reimplement the checks: it calls the same
verification function the loader uses, so what it removes is exactly what the
loader would have ignored.

Examples:
    python tools/prune_dataset.py --root datasets/obb-defect --format defect
    python tools/prune_dataset.py --root datasets/obb-defect --format defect --apply
"""

import argparse
from pathlib import Path

from ultralytics.data.utils import img2label_paths
from ultralytics.utils import YAML

from obbq.data import check_quality, make_check_defect, verify_image_label_extra

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def checker_for(fmt: str, data: dict):
    """Return the extra-column validator for a label format."""
    if fmt == "quality":
        return check_quality
    names = data.get("defect_names")
    if not names:
        raise SystemExit("--format defect needs 'defect_names' in the dataset YAML")
    return make_check_defect(len(names))


def scan(root: Path, split: str, num_cls: int, check_extra):
    """Return (kept, rejected) samples for one split, rejected as (image, label, reason)."""
    img_dir = root / "images" / split
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    labels = img2label_paths([str(p) for p in images])
    kept, rejected = 0, []
    for img, lbl in zip(images, labels):
        result = verify_image_label_extra((str(img), lbl, "", False, num_cls, 0, 0, False, check_extra))
        _, box, _, _, _, nm, _, ne, nc, msg = result
        if nc:  # corrupt: the loader ignores the whole sample
            rejected.append((img, Path(lbl), msg.strip().split("ignoring corrupt image/label: ")[-1]))
        elif nm or ne or box is None or not len(box):  # missing or empty labels: no box, no class
            rejected.append((img, Path(lbl), "no objects (missing or empty label file)"))
        else:
            kept += 1
    return kept, rejected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="datasets/obb-defect")
    p.add_argument("--format", choices=("quality", "defect"), default="defect")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--apply", action="store_true", help="delete the samples; without it, only report")
    a = p.parse_args()

    root = Path(a.root).resolve()
    data = YAML.load(root / "data.yaml")
    check_extra = checker_for(a.format, data)
    num_cls = len(data["names"])

    total_removed = 0
    for split in a.splits:
        if not (root / "images" / split).is_dir():
            continue
        kept, rejected = scan(root, split, num_cls, check_extra)
        print(f"\n{split}: {kept} usable, {len(rejected)} to remove")
        for img, _, reason in rejected:
            print(f"    {img.name}  ({reason})")
        if a.apply:
            for img, lbl, _ in rejected:
                img.unlink(missing_ok=True)
                lbl.unlink(missing_ok=True)
            for cache in (root / "labels").glob("*.cache"):
                cache.unlink()  # the cache is keyed on the file list, which just changed
        total_removed += len(rejected)

    if not a.apply:
        print(f"\n{total_removed} sample(s) would be removed. Re-run with --apply to delete them.")
    else:
        print(f"\nremoved {total_removed} sample(s), and cleared the label caches")


if __name__ == "__main__":
    main()
