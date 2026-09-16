"""Model, trainer, validator and predictor for OBB heads with a third output.

`OBBExtraModel` and friends hold everything the variants share: binding the
quality/defect head into `parse_model`, building the extra-column dataset,
splitting the extra channels back out after postprocessing, and keeping the
packed `cls` away from the stock plotting helpers. A variant supplies its head,
loss, dataset and metrics through class attributes and a few small hooks.
"""

from __future__ import annotations

import contextlib
import copy
from copy import copy as shallow_copy
from pathlib import Path

import numpy as np
import torch
from ultralytics.data.build import get_split_fraction
from ultralytics.engine.model import Model
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.models.yolo.obb.predict import OBBPredictor
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.models.yolo.obb.val import OBBValidator
from ultralytics.nn import tasks
from ultralytics.nn.tasks import OBBModel, yaml_model_load
from ultralytics.utils import LOGGER, RANK, ops
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.torch_utils import unwrap_model

from .data import ExtraOBBDataset, QualityOBBDataset, build_extra_dataset, strip_quality
from .head import OBB26ExtraBranch, OBB26Quality
from .loss import OBBExtraLoss, OBBQualityLoss

CFG_DIR = Path(__file__).parent / "cfg"
IOU_MATCH = 0.5  # IoU above which a prediction is counted as the match for a ground-truth object


def model_cfg(scale: str = "n", variant: str = "quality") -> str:
    """Return the path of a model YAML for a compound scale (n/s/m/l/x) and variant.

    The scale letter is read out of the file name by ultralytics, which then falls back to the unscaled YAML that
    actually exists on disk -- the same convention as `yolo26n-obb.yaml`.
    """
    if scale not in "nsmlx" or len(scale) != 1:
        raise ValueError(f"scale must be one of n, s, m, l, x; got {scale!r}")
    return str(CFG_DIR / f"yolo26{scale}-obb-{variant}.yaml")


@contextlib.contextmanager
def bind_head(head_class: type):
    """Temporarily bind the name `OBB26` in `ultralytics.nn.tasks` to `head_class`.

    `parse_model` looks its head classes up by name in that module's globals and compares against them, so binding
    the name is what makes a YAML `OBB26` row build the variant head with the right channel arguments. The binding
    is scoped to model construction and leaves stock OBB models alone.
    """
    original = tasks.OBB26
    tasks.OBB26 = head_class
    try:
        yield
    finally:
        tasks.OBB26 = original


class OBBExtraModel(OBBModel):
    """YOLO26 OBB model whose head predicts boxes, classes and one extra output."""

    head_class: type = OBB26ExtraBranch
    loss_class: type = OBBExtraLoss
    variant: str = "extra"

    def __init__(self, cfg=None, ch: int = 3, nc: int | None = None, verbose: bool = True, n_extra: int | None = None):
        """Initialize the model, building the head with `n_extra` extra channels."""
        cfg = cfg or model_cfg("n", self.variant)
        if n_extra is not None:
            cfg = self.with_n_extra(cfg, n_extra)
        with bind_head(self.head_class):
            super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    @staticmethod
    def with_n_extra(cfg, n_extra: int) -> dict:
        """Return the model dict with the head row's third argument set to `n_extra`."""
        d = copy.deepcopy(cfg if isinstance(cfg, dict) else yaml_model_load(cfg))
        head_args = d["head"][-1][-1]
        if len(head_args) < 3:
            raise ValueError(f"head row {d['head'][-1]} must pass three arguments, i.e. [nc, 1, 1]")
        head_args[2] = n_extra
        return d

    def init_criterion(self):
        """Initialize the variant's loss criterion."""
        return (
            E2ELoss(self, self.loss_class)
            if getattr(self.model[-1], "one2one_cv2", None) is not None
            else self.loss_class(self)
        )


class OBBExtraTrainer(OBBTrainer):
    """Trainer for OBB models with a third output."""

    model_class: type = OBBExtraModel
    dataset_class: type = ExtraOBBDataset
    validator_class: type | None = None

    def model_kwargs(self) -> dict:
        """Return extra keyword arguments for the model constructor."""
        return {}

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """Return the variant's model, initialized with the given config and weights."""
        model = self.set_model_names_for_load(
            self.model_class(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
                **self.model_kwargs(),
            )
        )
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build the variant's dataset."""
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_extra_dataset(
            self.dataset_class, self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs
        )

    def get_validator(self):
        """Return the variant's validator."""
        return self.validator_class(
            self.test_loader, save_dir=self.save_dir, args=shallow_copy(self.args), _callbacks=self.callbacks
        )

    def get_class_counts(self):
        """Count instances per class, ignoring the extra column of `cls`."""
        classes = np.concatenate([lb["cls"][:, 0] for lb in self.train_loader.dataset.labels], 0)
        return np.bincount(classes.astype(int), minlength=self.data["nc"]).astype(np.float32)

    def plot_training_samples(self, batch: dict, ni: int) -> None:
        """Plot training samples, hiding the extra column from the stock plotter."""
        super().plot_training_samples(strip_quality(batch), ni)

    def plot_training_labels(self) -> None:
        """Plot the label distribution using class indices only."""
        from ultralytics.utils.plotting import plot_labels

        labels = self.train_loader.dataset.labels
        boxes = np.concatenate([lb["bboxes"] for lb in labels], 0)
        cls = np.concatenate([lb["cls"][:, 0:1] for lb in labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)


class OBBExtraValidator(OBBValidator):
    """Validator for OBB models with a third output.

    Box and class metrics are the stock OBB metrics. The extra output is scored separately over ground-truth objects
    that a prediction of the right class matched above `IOU_MATCH`.
    """

    dataset_class: type = ExtraOBBDataset
    extra_key: str = "extra"
    extra_default: float = 1.0

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        """Build the variant's dataset."""
        fraction = get_split_fraction(self.args.fraction, self.args.split or "val")
        return build_extra_dataset(
            self.dataset_class, self.args, img_path, batch, self.data, mode=mode, stride=self.stride, fraction=fraction
        )

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Split the head's extra channels into the angle (appended to the box) and the extra output."""
        outputs = DetectionValidator.postprocess(self, preds)
        for pred in outputs:
            extra = pred.pop("extra")
            pred["bboxes"] = torch.cat([pred["bboxes"], extra[:, 0:1]], dim=-1)  # xywhr
            pred[self.extra_key] = extra[:, 1:]  # (N, n_extra)
        return outputs

    def _prepare_batch(self, si: int, batch: dict) -> dict:
        """Prepare one image's ground truth, keeping its extra labels alongside the classes."""
        pbatch = super()._prepare_batch(si, strip_quality(batch))
        cls = batch["cls"][batch["batch_idx"] == si]
        pbatch[self.extra_key] = (
            cls[:, 1] if cls.shape[-1] > 1 else torch.full_like(cls[:, 0], self.extra_default)
        )
        return pbatch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize metrics and reset the extra output's accumulators."""
        super().init_metrics(model)
        self.reset_extra(model)

    def reset_extra(self, model: torch.nn.Module) -> None:
        """Reset the extra output's accumulators."""
        self.extra_pred: list = []
        self.extra_true: list = []

    def _process_batch(self, preds: dict, batch: dict) -> dict[str, np.ndarray]:
        """Match predictions to ground truth and accumulate the extra output of every matched pair."""
        out = super()._process_batch(preds, batch)
        if batch["cls"].shape[0] and preds["cls"].shape[0] and self.extra_key in preds and self.extra_key in batch:
            iou = batch_probiou(batch["bboxes"], preds["bboxes"])
            best = iou.argmax(dim=1)  # best prediction per ground-truth object
            keep = (iou.gather(1, best[:, None]).squeeze(1) > IOU_MATCH) & (preds["cls"][best] == batch["cls"])
            if keep.any():
                self.accumulate_extra(preds[self.extra_key][best][keep], batch[self.extra_key][keep])
        return out

    def accumulate_extra(self, pred: torch.Tensor, true: torch.Tensor) -> None:
        """Record the extra output of matched pairs. `pred` is (N, n_extra), `true` is (N,)."""
        raise NotImplementedError

    def extra_metrics(self) -> dict[str, float]:
        """Return the extra output's metrics, keyed as they appear in `results.csv`."""
        raise NotImplementedError

    def extra_summary(self) -> str:
        """Return a one-line summary of the extra output for the results table."""
        raise NotImplementedError

    def get_stats(self) -> dict:
        """Return the stock metrics plus the extra output's metrics."""
        return {**super().get_stats(), **self.extra_metrics()}

    def print_results(self) -> None:
        """Print the stock results table followed by the extra output's scores."""
        super().print_results()
        LOGGER.info(self.extra_summary())

    def plot_val_samples(self, batch: dict, ni: int) -> None:
        """Plot validation samples without the extra column."""
        super().plot_val_samples(strip_quality(batch), ni)

    def plot_predictions(self, batch: dict, preds: list[dict], ni: int) -> None:
        """Plot predictions without the extra column."""
        super().plot_predictions(strip_quality(batch), preds, ni)


class OBBExtraPredictor(OBBPredictor):
    """Predictor that keeps the head's third output on the returned results."""

    def construct_result(self, pred: torch.Tensor, img, orig_img, img_path):
        """Build the OBB result and attach the extra output of each detection.

        Args:
            pred (torch.Tensor): Predictions of shape (N, 6 + 1 + n_extra):
                [x, y, w, h, conf, cls, angle, *extra].
        """
        from ultralytics.engine.results import Results

        rboxes = torch.cat([pred[:, :4], pred[:, 6:7]], dim=-1)  # xywh + angle (not the last column any more)
        rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True)
        obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1)
        result = Results(orig_img, path=img_path, names=self.model.names, obb=obb)
        return self.attach_extra(result, pred[:, 7:])

    def attach_extra(self, result, extra: torch.Tensor):
        """Attach the extra channels (N, n_extra) to a result object."""
        raise NotImplementedError


class OBBQualityModel(OBBExtraModel):
    """YOLO26 OBB model whose third output is a per-object quality score."""

    head_class = OBB26Quality
    loss_class = OBBQualityLoss
    variant = "quality"


class OBBQualityValidator(OBBExtraValidator):
    """Validator reporting the quality output's error and correlation."""

    dataset_class = QualityOBBDataset
    extra_key = "quality"
    extra_default = 1.0

    def accumulate_extra(self, pred: torch.Tensor, true: torch.Tensor) -> None:
        """Record the predicted and annotated quality of matched objects."""
        self.extra_pred.extend(pred[:, 0].cpu().tolist())
        self.extra_true.extend(true.cpu().tolist())

    def quality_stats(self) -> tuple[float, float, int]:
        """Return (mean absolute error, Pearson correlation, matched count) for the quality output."""
        pred = np.asarray(self.extra_pred, dtype=np.float64)
        true = np.asarray(self.extra_true, dtype=np.float64)
        if pred.size == 0:
            return float("nan"), float("nan"), 0
        mae = float(np.abs(pred - true).mean())
        corr = float("nan")
        if pred.size > 1 and pred.std() > 1e-8 and true.std() > 1e-8:
            corr = float(np.corrcoef(pred, true)[0, 1])
        return mae, corr, int(pred.size)

    def extra_metrics(self) -> dict[str, float]:
        """Return the quality output's metrics."""
        mae, corr, _ = self.quality_stats()
        return {"metrics/quality_mae": mae, "metrics/quality_corr": corr}

    def extra_summary(self) -> str:
        """Summarize the quality output."""
        mae, corr, n = self.quality_stats()
        return f"quality: MAE {mae:.4f}, corr {corr:.4f}, over {n} matched object(s)"


class OBBQualityPredictor(OBBExtraPredictor):
    """Predictor attaching a `quality` tensor of shape (N,) to each result."""

    def attach_extra(self, result, extra: torch.Tensor):
        """Attach the quality score of each detection."""
        result.quality = extra[:, 0] if extra.shape[1] else torch.zeros(extra.shape[0], device=extra.device)
        return result


class OBBQualityTrainer(OBBExtraTrainer):
    """Trainer for quality-aware OBB models."""

    model_class = OBBQualityModel
    dataset_class = QualityOBBDataset
    validator_class = OBBQualityValidator


class YOLOQuality(Model):
    """`ultralytics.YOLO`-style entry point for quality-aware OBB models.

    Examples:
        >>> from obbq import YOLOQuality, model_cfg
        >>> model = YOLOQuality(model_cfg("n", "quality"))
        >>> model.train(data="datasets/obb-quality/data.yaml", epochs=10, imgsz=320)
    """

    variant = "quality"

    def __init__(self, model=None, verbose: bool = False):
        """Initialize with the variant's config by default."""
        super().__init__(model=model or model_cfg("n", self.variant), task="obb", verbose=verbose)

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
