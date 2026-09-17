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

It handles both layouts in this repo: the generators' `images/<split>/` tree,
and the flat `images/` + `labels/` tree that `tools/label_defect_gui.py` writes
(where the splits are `train.txt` / `val.txt` lists). In the flat case the list
files are rewritten so they never point at a removed image.

Examples:
    python tools/prune_dataset.py --root datasets/obb-defect --format defect
    python tools/prune_dataset.py --root labeled_dataset --format defect --apply
"""

import argparse
import os
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


def image_dirs(root: Path, splits: list[str]) -> list[tuple[str, Path]]:
    """Return the (name, directory) pairs to scan, for either dataset layout."""
    split_dirs = [(s, root / "images" / s) for s in splits if (root / "images" / s).is_dir()]
    if split_dirs:
        return split_dirs
    if (root / "images").is_dir():
        return [("images", root / "images")]  # flat layout, as written by label_defect_gui.py
    raise SystemExit(f"no images/ directory under {root}")


def scan(img_dir: Path, num_cls: int, check_extra):
    """Return (kept, rejected) samples for one directory, rejected as (image, label, reason)."""
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


def find_orphan_labels(root: Path, splits: list[str]) -> list[Path]:
    """Return label files that have no matching image, in either layout."""
    orphans = []
    for lbl_dir in [root / "labels" / s for s in splits] + [root / "labels"]:
        if not lbl_dir.is_dir():
            continue
        img_dir = Path(str(lbl_dir).replace(f"{os.sep}labels", f"{os.sep}images", 1))
        if not img_dir.is_dir():
            continue
        stems = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
        orphans += [p for p in sorted(lbl_dir.glob("*.txt")) if p.stem not in stems]
    return orphans


def rewrite_split_lists(root: Path, removed: set[str]) -> list[tuple[Path, int, int]]:
    """Drop removed images from any train.txt / val.txt list files, so none dangle."""
    rewritten = []
    for list_file in sorted(root.glob("*.txt")):
        lines = [ln for ln in list_file.read_text().splitlines() if ln.strip()]
        keep = [ln for ln in lines if Path(ln).name not in removed]
        if len(keep) != len(lines):
            list_file.write_text("\n".join(keep) + "\n" if keep else "")
            rewritten.append((list_file, len(lines) - len(keep), len(keep)))
    return rewritten


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

    total_removed, removed_names = 0, set()
    for name, img_dir in image_dirs(root, a.splits):
        kept, rejected = scan(img_dir, num_cls, check_extra)
        print(f"\n{name}: {kept} usable, {len(rejected)} to remove")
        for img, _, reason in rejected:
            print(f"    {img.name}  ({reason})")
        if a.apply:
            for img, lbl, _ in rejected:
                img.unlink(missing_ok=True)
                lbl.unlink(missing_ok=True)
                removed_names.add(img.name)
        total_removed += len(rejected)

    orphans = find_orphan_labels(root, a.splits)
    if orphans:
        print(f"\norphan labels with no image: {len(orphans)}")
        for lbl in orphans:
            print(f"    {lbl.name}")
        if a.apply:
            for lbl in orphans:
                lbl.unlink(missing_ok=True)

    if not a.apply:
        print(f"\n{total_removed + len(orphans)} file(s) would be removed. Re-run with --apply to delete them.")
        return

    for cache in root.rglob("*.cache"):
        cache.unlink()  # the label cache is keyed on the file list, which just changed
    rewritten = rewrite_split_lists(root, removed_names)
    print(f"\nremoved {total_removed} sample(s) and {len(orphans)} orphan label(s); cleared the label caches")
    for path, dropped, left in rewritten:
        print(f"    {path.name}: dropped {dropped} entr{'y' if dropped == 1 else 'ies'}, {left} left")
        if not left:
            print(f"    WARNING: {path.name} is now empty -- re-run tools/split_labeled_dataset.py to re-split")


if __name__ == "__main__":
    main()
