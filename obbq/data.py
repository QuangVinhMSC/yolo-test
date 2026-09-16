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


def check_quality(values: np.ndarray) -> np.ndarray:
    """Validate a quality column: a scalar in [0, 1]."""
    assert values.min() >= -0.01 and values.max() <= 1.01, (
        f"quality must be in [0, 1], got {values.min():.4g} to {values.max():.4g}"
    )
    return values.clip(0.0, 1.0)


def make_check_defect(nd: int):
    """Return a validator for a defect-class column: an integer index in [0, nd)."""

    def check_defect(values: np.ndarray) -> np.ndarray:
        assert np.allclose(values, np.round(values)), (
            f"defect class must be an integer index, got non-integer values {values[values != np.round(values)]}"
        )
        assert values.min() >= -0.01 and values.max() < nd, (
            f"defect class must be in 0-{nd - 1}, got {values.min():.4g} to {values.max():.4g}"
        )
        return np.round(values)

    return check_defect


def verify_image_label_extra(args: tuple) -> list:
    """Verify one image-label pair in an extra-column OBB format.

    Mirrors `ultralytics.data.utils.verify_image_label` for oriented boxes, with the trailing extra value split off
    before the polygon is parsed and handed to `check_extra` for validation. It is returned in place of the keypoints
    slot of the stock result tuple.
    """
    im_file, lb_file, prefix, _keypoint, num_cls, _nkpt, _ndim, single_cls, check_extra = args
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
                quality = check_extra(quality)

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


class ExtraOBBDataset(YOLODataset):
    """YOLO OBB dataset whose labels carry one extra per-object value.

    `cls` is (n, 2): column 0 is the class index, column 1 is the extra label. Everything else behaves like the
    stock OBB dataset. Subclasses set `cache_tag` and supply `check_extra`.
    """

    format_class = QualityFormat
    cache_tag: str = "extra"

    def check_extra(self, values: np.ndarray) -> np.ndarray:
        """Validate the extra column of one label file."""
        raise NotImplementedError

    def get_cache_hash(self) -> str:
        """Tag the cache so a label cache is never reused across differing label formats."""
        return f"{super().get_cache_hash()}-{self.cache_tag}"

    def verify_args(self) -> tuple:
        """Return the extra-column verification function and its argument iterable."""
        return verify_image_label_extra, zip(
            self.im_files,
            self.label_files,
            repeat(self.prefix),
            repeat(self.use_keypoints),
            repeat(len(self.data["names"])),
            repeat(0),
            repeat(0),
            repeat(self.single_cls),
            repeat(self.check_extra),
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


class QualityOBBDataset(ExtraOBBDataset):
    """OBB dataset whose extra column is a per-object quality score in [0, 1].

    Examples:
        >>> dataset = QualityOBBDataset(img_path="images/train", data={"names": {0: "rect"}}, task="obb")
    """

    cache_tag = "quality"

    def check_extra(self, values: np.ndarray) -> np.ndarray:
        """Validate the quality column."""
        return check_quality(values)


class DefectOBBDataset(ExtraOBBDataset):
    """OBB dataset whose extra column is a per-object defect class index.

    The number of defect classes comes from `defect_names` in the dataset YAML.

    Examples:
        >>> data = {"names": {0: "rect"}, "defect_names": {0: "ok", 1: "scratch"}}
        >>> dataset = DefectOBBDataset(img_path="images/train", data=data, task="obb")
    """

    cache_tag = "defect"

    def check_extra(self, values: np.ndarray) -> np.ndarray:
        """Validate the defect-class column against the dataset's defect classes."""
        return make_check_defect(len(self.data["defect_names"]))(values)


def build_extra_dataset(
    dataset_class: type[ExtraOBBDataset],
    cfg,
    img_path: str,
    batch: int,
    data: dict,
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
    fraction: float | None = None,
) -> ExtraOBBDataset:
    """Build an `ExtraOBBDataset`, mirroring `ultralytics.data.build.build_yolo_dataset`."""
    pad = 0.0 if mode == "train" else 0.5
    if data.get("complete"):
        fraction = 1.0
    elif fraction is None:
        fraction = get_split_fraction(cfg.fraction, mode)
    return dataset_class(
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


def build_quality_dataset(cfg, img_path, batch, data, **kwargs) -> QualityOBBDataset:
    """Build a `QualityOBBDataset`."""
    return build_extra_dataset(QualityOBBDataset, cfg, img_path, batch, data, **kwargs)


def build_defect_dataset(cfg, img_path, batch, data, **kwargs) -> DefectOBBDataset:
    """Build a `DefectOBBDataset`."""
    return build_extra_dataset(DefectOBBDataset, cfg, img_path, batch, data, **kwargs)


def strip_quality(batch: dict) -> dict:
    """Return a shallow copy of `batch` whose `cls` holds class indices only, for stock plotting helpers."""
    cls = batch.get("cls")
    if cls is None or cls.shape[-1] <= 1:
        return batch
    return {**batch, "cls": cls[:, 0:1]}
