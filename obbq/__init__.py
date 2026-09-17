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
from .loss import (
    DEFAULT_DEFECT_CLASS_WEIGHTS,
    DEFAULT_DEFECT_FOCAL_ALPHA,
    DEFAULT_DEFECT_FOCAL_GAMMA,
    DEFAULT_DEFECT_GAIN,
    DEFAULT_QUALITY_GAIN,
    OBBDefectLoss,
    OBBQualityLoss,
)
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
    "DEFAULT_DEFECT_CLASS_WEIGHTS",
    "DEFAULT_DEFECT_FOCAL_ALPHA",
    "DEFAULT_DEFECT_FOCAL_GAMMA",
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


def register(
    quality: float = DEFAULT_QUALITY_GAIN,
    defect: float = DEFAULT_DEFECT_GAIN,
    defect_focal_gamma: float = DEFAULT_DEFECT_FOCAL_GAMMA,
    defect_focal_alpha: float = DEFAULT_DEFECT_FOCAL_ALPHA,
    defect_class_weights: str = DEFAULT_DEFECT_CLASS_WEIGHTS,
) -> None:
    """Register the extra hyperparameters so `train(quality=...)` / `train(defect=...)` etc. set them.

    Ultralytics validates training arguments against its default config, so a new key has to be declared there
    before it can be passed. Calling this more than once is harmless.

    Args:
        quality (float): Default gain for the quality variant's loss term.
        defect (float): Default gain for the defect variant's loss term.
        defect_focal_gamma (float): Default focal-loss gamma for the defect term; 0 disables focal weighting.
        defect_focal_alpha (float): Default focal-loss alpha for the defect term's positive class.
        defect_class_weights (str): Comma-separated per-defect-class weights, in `defect_names` order; "" disables.
    """
    from ultralytics import cfg as ultralytics_cfg
    from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT

    float_keys = (
        ("quality", quality),
        ("defect", defect),
        ("defect_focal_gamma", defect_focal_gamma),
        ("defect_focal_alpha", defect_focal_alpha),
    )
    for key, value in float_keys:
        DEFAULT_CFG_DICT.setdefault(key, value)
        if not hasattr(DEFAULT_CFG, key):
            setattr(DEFAULT_CFG, key, value)
        if key not in ultralytics_cfg.CFG_FLOAT_KEYS:
            ultralytics_cfg.CFG_FLOAT_KEYS = frozenset(ultralytics_cfg.CFG_FLOAT_KEYS | {key})

    key, value = "defect_class_weights", defect_class_weights
    DEFAULT_CFG_DICT.setdefault(key, value)
    if not hasattr(DEFAULT_CFG, key):
        setattr(DEFAULT_CFG, key, value)
    if key not in ultralytics_cfg.CFG_STR_KEYS:
        ultralytics_cfg.CFG_STR_KEYS = frozenset(ultralytics_cfg.CFG_STR_KEYS | {key})
