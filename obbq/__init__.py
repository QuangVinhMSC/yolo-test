"""Quality-aware YOLO26 OBB: a third head output alongside boxes and classes.

Usage:
    >>> import obbq
    >>> obbq.register()  # adds the `quality` loss-gain hyperparameter
    >>> model = obbq.YOLOQuality(obbq.model_cfg("n"))
    >>> model.train(data="datasets/obb-quality/data.yaml", epochs=10, imgsz=320)
"""

from __future__ import annotations

from .data import QualityOBBDataset, build_quality_dataset, verify_image_label_quality
from .head import OBB26Quality
from .loss import DEFAULT_QUALITY_GAIN, OBBQualityLoss
from .model import (
    OBBQualityModel,
    OBBQualityPredictor,
    OBBQualityTrainer,
    OBBQualityValidator,
    YOLOQuality,
    model_cfg,
)

__all__ = [
    "DEFAULT_QUALITY_GAIN",
    "OBB26Quality",
    "OBBQualityLoss",
    "OBBQualityModel",
    "OBBQualityPredictor",
    "OBBQualityTrainer",
    "OBBQualityValidator",
    "QualityOBBDataset",
    "YOLOQuality",
    "build_quality_dataset",
    "model_cfg",
    "register",
    "verify_image_label_quality",
]


def register(default: float = DEFAULT_QUALITY_GAIN) -> None:
    """Register `quality` as a trainable hyperparameter so `train(quality=...)` sets the quality loss gain.

    Ultralytics validates training arguments against its default config, so a new gain has to be declared there
    before it can be passed. Calling this more than once is harmless.

    Args:
        default (float): Default quality loss gain.
    """
    from ultralytics import cfg as ultralytics_cfg
    from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT

    DEFAULT_CFG_DICT.setdefault("quality", default)
    if not hasattr(DEFAULT_CFG, "quality"):
        setattr(DEFAULT_CFG, "quality", default)
    if "quality" not in ultralytics_cfg.CFG_FLOAT_KEYS:
        ultralytics_cfg.CFG_FLOAT_KEYS = frozenset(ultralytics_cfg.CFG_FLOAT_KEYS | {"quality"})
