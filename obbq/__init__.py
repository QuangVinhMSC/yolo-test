"""Quality-aware YOLO26 OBB: a third head output alongside boxes and classes.

Usage:
    >>> import obbq
    >>> obbq.register()  # adds the `quality` loss-gain hyperparameter
    >>> model = obbq.YOLOQuality(obbq.model_cfg("n"))
    >>> model.train(data="datasets/obb-quality/data.yaml", epochs=10, imgsz=320)
"""

from __future__ import annotations

from .data import (
    DefectOBBDataset,
    QualityOBBDataset,
    build_defect_dataset,
    build_quality_dataset,
    verify_image_label_extra,
)
from .head import OBB26Defect, OBB26Quality
from .loss import DEFAULT_DEFECT_GAIN, DEFAULT_QUALITY_GAIN, OBBDefectLoss, OBBQualityLoss
from .defect import (
    OBBDefectModel,
    OBBDefectPredictor,
    OBBDefectTrainer,
    OBBDefectValidator,
    YOLODefect,
)
from .model import (
    OBBQualityModel,
    OBBQualityPredictor,
    OBBQualityTrainer,
    OBBQualityValidator,
    YOLOQuality,
    model_cfg,
)

__all__ = [
    "DEFAULT_DEFECT_GAIN",
    "DEFAULT_QUALITY_GAIN",
    "OBB26Defect",
    "OBB26Quality",
    "OBBDefectLoss",
    "OBBQualityLoss",
    "OBBDefectModel",
    "OBBDefectPredictor",
    "OBBDefectTrainer",
    "OBBDefectValidator",
    "OBBQualityModel",
    "OBBQualityPredictor",
    "OBBQualityTrainer",
    "OBBQualityValidator",
    "DefectOBBDataset",
    "QualityOBBDataset",
    "build_defect_dataset",
    "YOLODefect",
    "YOLOQuality",
    "build_quality_dataset",
    "model_cfg",
    "register",
    "verify_image_label_extra",
]


def register(quality: float = DEFAULT_QUALITY_GAIN, defect: float = DEFAULT_DEFECT_GAIN) -> None:
    """Register the extra loss gains so `train(quality=...)` / `train(defect=...)` set them.

    Ultralytics validates training arguments against its default config, so a new gain has to be declared there
    before it can be passed. Calling this more than once is harmless.

    Args:
        quality (float): Default gain for the quality variant's loss term.
        defect (float): Default gain for the defect variant's loss term.
    """
    from ultralytics import cfg as ultralytics_cfg
    from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT

    for key, value in (("quality", quality), ("defect", defect)):
        DEFAULT_CFG_DICT.setdefault(key, value)
        if not hasattr(DEFAULT_CFG, key):
            setattr(DEFAULT_CFG, key, value)
        if key not in ultralytics_cfg.CFG_FLOAT_KEYS:
            ultralytics_cfg.CFG_FLOAT_KEYS = frozenset(ultralytics_cfg.CFG_FLOAT_KEYS | {key})
