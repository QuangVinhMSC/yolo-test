"""Quality-aware OBB label format and dataset.

Label format -- the stock DOTA-style OBB line with one extra trailing value::

    cls x1 y1 x2 y2 x3 y3 x4 y4 quality

Coordinates stay normalized to [0, 1] as usual; `quality` is a scalar in
[0, 1]. A line without the extra value is accepted and defaults to quality 1.0,
so a plain OBB dataset still trains.

Quality is carried through the pipeline as a **second column of `cls`**. Every
augmentation that drops, clips, mixes or reorders instances already filters
`cls` by the same indices as the boxes, so the quality of an object follows its
box through mosaic, perspective, flips and copy-paste without touching any of
those transforms. The loss splits the two columns again.
"""

from __future__ import annotations

import os
from copy import copy
from itertools import repeat

import numpy as np
import torch
from ultralytics.data.augment import Format
from ultralytics.data.build import get_split_fraction
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_image
from ultralytics.utils import colorstr
from ultralytics.utils.instance import Instances  # noqa: F401  (kept for parity with ultralytics imports)
from ultralytics.utils.ops import segments2boxes

CLS_COLUMNS = 2  # cls carries [class index, quality]
POLYGON_COLUMNS = 9  # cls + 4 corner points
QUALITY_COLUMNS = 10  # cls + 4 corner points + quality
DEFAULT_QUALITY = 1.0


def verify_image_label_quality(args: tuple) -> list:
    """Verify one image-label pair in the quality-aware OBB format.

    Mirrors `ultralytics.data.utils.verify_image_label` for oriented boxes, with the trailing quality value split off
    before the polygon is parsed. Quality is returned in place of the keypoints slot of the stock result tuple.
    """
    im_file, lb_file, prefix, _keypoint, num_cls, _nkpt, _ndim, single_cls = args
    # Number (missing, found, empty, corrupt), message, segments, quality
    nm, nf, ne, nc, msg, segments, quality = 0, 0, 0, 0, "", [], None
    try:
        # Verify images
        msg, shape = check_image(im_file)
        msg = f"{prefix}{msg}" if msg else ""

        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file, encoding="utf-8") as f:
                rows = [x.split() for x in f.read().strip().splitlines() if len(x)]
            if nl := len(rows):
                widths = {len(x) for x in rows}
                assert widths <= {POLYGON_COLUMNS, QUALITY_COLUMNS}, (
                    f"labels require {POLYGON_COLUMNS} columns (plain OBB) or {QUALITY_COLUMNS} columns "
                    f"(OBB + quality), {sorted(widths)} columns detected"
                )
                classes = np.array([x[0] for x in rows], dtype=np.float32)
                quality = np.array(
                    [x[POLYGON_COLUMNS] if len(x) == QUALITY_COLUMNS else DEFAULT_QUALITY for x in rows],
                    dtype=np.float32,
                ).reshape(-1, 1)
                segments = [np.array(x[1:POLYGON_COLUMNS], dtype=np.float32).reshape(-1, 2) for x in rows]
                lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)

                points = np.concatenate(segments, 0)
                # Coordinate points check with 1% tolerance
                assert points.max() <= 1.01, f"non-normalized or out of bounds coordinates {points[points > 1.01]}"
                assert points.min() >= -0.01, f"negative coordinate {points[points < -0.01]}"
                assert quality.min() >= -0.01 and quality.max() <= 1.01, (
                    f"quality must be in [0, 1], got {quality.min():.4g} to {quality.max():.4g}"
                )
                quality = quality.clip(0.0, 1.0)

                max_cls = 0 if single_cls else lb[:, 0].max()
                assert max_cls < num_cls, (
                    f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                    f"Possible class labels are 0-{num_cls - 1}"
                )
                assert lb[:, 0].min() >= 0, f"negative class label {lb[:, 0].min()}"

                rows_arr = np.array([[float(v) for v in (x + [DEFAULT_QUALITY])[:QUALITY_COLUMNS]] for x in rows])
                _, i = np.unique(rows_arr, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    i = np.sort(i)
                    lb, quality = lb[i], quality[i]
                    segments = [segments[x] for x in i]
                    msg = f"{prefix}{im_file}: {nl - len(i)} duplicate labels removed"
                if single_cls:
                    lb[:, 0] = 0
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
                quality = np.zeros((0, 1), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
            quality = np.zeros((0, 1), dtype=np.float32)
        return im_file, lb[:, :5], shape, segments, quality, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}{im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, None, nm, nf, ne, nc, msg]


class QualityFormat(Format):
    """Format transform that keeps the packed `[class, quality]` shape of `cls` for empty images."""

    def apply_instances(self, labels: dict, params: dict | None = None) -> dict:
        """Format instances, restoring the quality column on images that contain no objects."""
        labels = super().apply_instances(labels, params)
        cls = labels.get("cls")
        if cls is not None and cls.shape[-1] != CLS_COLUMNS:  # stock Format emits zeros(nl, 1) when nl == 0
            labels["cls"] = torch.zeros((cls.shape[0], CLS_COLUMNS), dtype=cls.dtype)
        return labels


class QualityOBBDataset(YOLODataset):
    """YOLO OBB dataset whose labels carry a per-object quality value.

    `cls` is (n, 2): column 0 is the class index, column 1 is the quality. Everything else behaves like the stock
    OBB dataset.

    Examples:
        >>> dataset = QualityOBBDataset(img_path="images/train", data={"names": {0: "rect"}}, task="obb")
    """

    format_class = QualityFormat

    def get_cache_hash(self) -> str:
        """Tag the cache so a plain-OBB label cache is never reused for quality-aware labels, or vice versa."""
        return f"{super().get_cache_hash()}-quality"

    def verify_args(self) -> tuple:
        """Return the quality-aware verification function and its argument iterable."""
        return verify_image_label_quality, zip(
            self.im_files,
            self.label_files,
            repeat(self.prefix),
            repeat(self.use_keypoints),
            repeat(len(self.data["names"])),
            repeat(0),
            repeat(0),
            repeat(self.single_cls),
        )

    def result_to_label(self, result: list) -> tuple[dict | None, int, int, int, int, str]:
        """Pack the verified class and quality into a single `cls` array of shape (n, 2)."""
        im_file, lb, shape, segments, quality, nm_f, nf_f, ne_f, nc_f, msg = result
        label = (
            {
                "im_file": im_file,
                "shape": shape,
                "cls": np.concatenate((lb[:, 0:1], quality), 1),  # n, 2 -> [class, quality]
                "bboxes": lb[:, 1:],  # n, 4
                "segments": segments,
                "keypoints": None,
                "normalized": True,
                "bbox_format": "xywh",
            }
            if im_file
            else None
        )
        return label, nm_f, nf_f, ne_f, nc_f, msg


def build_quality_dataset(
    cfg,
    img_path: str,
    batch: int,
    data: dict,
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
    fraction: float | None = None,
) -> QualityOBBDataset:
    """Build a `QualityOBBDataset`, mirroring `ultralytics.data.build.build_yolo_dataset`."""
    pad = 0.0 if mode == "train" else 0.5
    if data.get("complete"):
        fraction = 1.0
    elif fraction is None:
        fraction = get_split_fraction(cfg.fraction, mode)
    return QualityOBBDataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=copy(cfg),
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=pad,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=fraction,
    )


def strip_quality(batch: dict) -> dict:
    """Return a shallow copy of `batch` whose `cls` holds class indices only, for stock plotting helpers."""
    cls = batch.get("cls")
    if cls is None or cls.shape[-1] <= 1:
        return batch
    return {**batch, "cls": cls[:, 0:1]}
