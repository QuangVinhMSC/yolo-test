"""Model, trainer, validator and predictor for quality-aware YOLO26 OBB."""

from __future__ import annotations

import contextlib
from copy import copy
from pathlib import Path

import numpy as np
import torch
from ultralytics.data.build import get_split_fraction
from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.models.yolo.obb.predict import OBBPredictor
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.models.yolo.obb.val import OBBValidator
from ultralytics.nn import tasks
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, ops
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.torch_utils import unwrap_model

from .data import build_quality_dataset, strip_quality
from .head import OBB26Quality
from .loss import OBBQualityLoss

CFG_DIR = Path(__file__).parent / "cfg"


def model_cfg(scale: str = "n") -> str:
    """Return the path of the quality-aware model YAML for a compound scale (n/s/m/l/x).

    The scale letter is read out of the file name by ultralytics, which then falls back to the unscaled YAML that
    actually exists on disk -- the same convention as `yolo26n-obb.yaml`.
    """
    if scale not in "nsmlx" or len(scale) != 1:
        raise ValueError(f"scale must be one of n, s, m, l, x; got {scale!r}")
    return str(CFG_DIR / f"yolo26{scale}-obb-quality.yaml")


@contextlib.contextmanager
def _quality_head():
    """Temporarily bind the name `OBB26` in `ultralytics.nn.tasks` to the quality-aware head.

    `parse_model` looks its head classes up by name in that module's globals and compares against them, so binding
    the name is what makes a YAML `OBB26` row build `OBB26Quality` with the right channel arguments. The binding is
    scoped to model construction and leaves stock OBB models alone.
    """
    original = tasks.OBB26
    tasks.OBB26 = OBB26Quality
    try:
        yield
    finally:
        tasks.OBB26 = original


class OBBQualityModel(OBBModel):
    """YOLO26 OBB model whose head predicts boxes, classes and a per-object quality score."""

    def __init__(self, cfg=None, ch: int = 3, nc: int | None = None, verbose: bool = True):
        """Initialize the model, building the head from `cfg` with the quality branch attached."""
        with _quality_head():
            super().__init__(cfg=cfg or model_cfg("n"), ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the quality-aware loss criterion."""
        return (
            E2ELoss(self, OBBQualityLoss)
            if getattr(self.model[-1], "one2one_cv2", None) is not None
            else OBBQualityLoss(self)
        )


class OBBQualityTrainer(OBBTrainer):
    """Trainer for quality-aware OBB models."""

    def get_model(self, cfg=None, weights=None, verbose: bool = True) -> OBBQualityModel:
        """Return an `OBBQualityModel` initialized with the given config and weights."""
        model = self.set_model_names_for_load(
            OBBQualityModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build a quality-aware OBB dataset."""
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_quality_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_validator(self):
        """Return an `OBBQualityValidator`."""
        return OBBQualityValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_class_counts(self):
        """Count instances per class, ignoring the quality column of `cls`."""
        classes = np.concatenate([lb["cls"][:, 0] for lb in self.train_loader.dataset.labels], 0)
        return np.bincount(classes.astype(int), minlength=self.data["nc"]).astype(np.float32)

    def plot_training_samples(self, batch: dict, ni: int) -> None:
        """Plot training samples, hiding the quality column from the stock plotter."""
        super().plot_training_samples(strip_quality(batch), ni)

    def plot_training_labels(self) -> None:
        """Plot the label distribution using class indices only."""
        labels = self.train_loader.dataset.labels
        boxes = np.concatenate([lb["bboxes"] for lb in labels], 0)
        cls = np.concatenate([lb["cls"][:, 0:1] for lb in labels], 0)
        from ultralytics.utils.plotting import plot_labels

        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)


class OBBQualityValidator(OBBValidator):
    """Validator for quality-aware OBB models.

    Box and class metrics are the stock OBB metrics. The quality output is scored separately as the mean absolute
    error between the predicted and annotated quality of matched objects, reported as `quality_mae`.
    """

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        """Build a quality-aware OBB dataset."""
        fraction = get_split_fraction(self.args.fraction, self.args.split or "val")
        return build_quality_dataset(
            self.args, img_path, batch, self.data, mode=mode, stride=self.stride, fraction=fraction
        )

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Split the head's extra channels into the angle (appended to the box) and the quality score."""
        outputs = DetectionValidator.postprocess(self, preds)
        for pred in outputs:
            extra = pred.pop("extra")
            pred["bboxes"] = torch.cat([pred["bboxes"], extra[:, 0:1]], dim=-1)  # xywhr
            pred["quality"] = extra[:, 1] if extra.shape[1] > 1 else torch.zeros_like(pred["conf"])
        return outputs

    def _prepare_batch(self, si: int, batch: dict) -> dict:
        """Prepare one image's ground truth, keeping its per-object quality alongside the classes."""
        pbatch = super()._prepare_batch(si, strip_quality(batch))
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx]
        pbatch["quality"] = cls[:, 1] if cls.shape[-1] > 1 else torch.ones_like(cls[:, 0])
        return pbatch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize metrics and reset the quality accumulators."""
        super().init_metrics(model)
        self._quality_pred: list[float] = []
        self._quality_true: list[float] = []

    def _process_batch(self, preds: dict, batch: dict) -> dict[str, np.ndarray]:
        """Match predictions to ground truth and accumulate the quality of every matched pair."""
        out = super()._process_batch(preds, batch)
        if batch["cls"].shape[0] and preds["cls"].shape[0] and "quality" in preds and "quality" in batch:
            iou = ops.batch_probiou(batch["bboxes"], preds["bboxes"])
            best = iou.argmax(dim=1)  # best prediction per ground-truth object
            keep = (iou.gather(1, best[:, None]).squeeze(1) > 0.5) & (preds["cls"][best] == batch["cls"])
            if keep.any():
                self._quality_pred.extend(preds["quality"][best][keep].cpu().tolist())
                self._quality_true.extend(batch["quality"][keep].cpu().tolist())
        return out

    def quality_stats(self) -> tuple[float, float, int]:
        """Return (mean absolute error, Pearson correlation, matched count) for the quality output."""
        pred = np.asarray(self._quality_pred, dtype=np.float64)
        true = np.asarray(self._quality_true, dtype=np.float64)
        if pred.size == 0:
            return float("nan"), float("nan"), 0
        mae = float(np.abs(pred - true).mean())
        corr = float("nan")
        if pred.size > 1 and pred.std() > 1e-8 and true.std() > 1e-8:
            corr = float(np.corrcoef(pred, true)[0, 1])
        return mae, corr, int(pred.size)

    def get_stats(self) -> dict:
        """Return the stock metrics plus the error and correlation of the quality output."""
        stats = super().get_stats()
        mae, corr, _ = self.quality_stats()
        stats["metrics/quality_mae"] = mae
        stats["metrics/quality_corr"] = corr
        return stats

    def print_results(self) -> None:
        """Print the stock results table followed by the quality output's scores."""
        super().print_results()
        mae, corr, n = self.quality_stats()
        LOGGER.info(f"quality: MAE {mae:.4f}, corr {corr:.4f}, over {n} matched object(s)")

    def plot_val_samples(self, batch: dict, ni: int) -> None:
        """Plot validation samples without the quality column."""
        super().plot_val_samples(strip_quality(batch), ni)

    def plot_predictions(self, batch: dict, preds: list[dict], ni: int) -> None:
        """Plot predictions without the quality column."""
        super().plot_predictions(strip_quality(batch), preds, ni)


class OBBQualityPredictor(OBBPredictor):
    """Predictor that keeps the head's third output on the returned results.

    Each `Results` object gains a `quality` tensor of shape (N,), aligned with `results.obb`.
    """

    def construct_result(self, pred: torch.Tensor, img, orig_img, img_path):
        """Build the OBB result and attach the predicted quality of each detection.

        Args:
            pred (torch.Tensor): Predictions of shape (N, 8): [x, y, w, h, conf, cls, angle, quality].
        """
        rboxes = torch.cat([pred[:, :4], pred[:, 6:7]], dim=-1)  # xywh + angle (not the last column any more)
        rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True)
        obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1)
        from ultralytics.engine.results import Results

        result = Results(orig_img, path=img_path, names=self.model.names, obb=obb)
        result.quality = pred[:, 7] if pred.shape[1] > 7 else torch.zeros(len(pred), device=pred.device)
        return result


class YOLOQuality(Model):
    """`ultralytics.YOLO`-style entry point for quality-aware OBB models.

    Examples:
        >>> from obbq import YOLOQuality, model_cfg
        >>> model = YOLOQuality(model_cfg("n"))
        >>> model.train(data="datasets/obb-quality/data.yaml", epochs=10, imgsz=320)
    """

    def __init__(self, model=None, verbose: bool = False):
        """Initialize with the quality-aware OBB config by default."""
        super().__init__(model=model or model_cfg("n"), task="obb", verbose=verbose)

    @property
    def task_map(self) -> dict:
        """Map the OBB task to the quality-aware model, trainer, validator and predictor."""
        return {
            "obb": {
                "model": OBBQualityModel,
                "trainer": OBBQualityTrainer,
                "validator": OBBQualityValidator,
                "predictor": OBBQualityPredictor,
            }
        }


__all__ = [
    "DEFAULT_CFG",
    "OBBQualityModel",
    "OBBQualityPredictor",
    "OBBQualityTrainer",
    "OBBQualityValidator",
    "YOLOQuality",
    "model_cfg",
    "yolo",
]
