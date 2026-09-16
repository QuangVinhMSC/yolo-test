"""The defect-class variant: a second classification head as the third output.

Where the quality variant regresses one scalar per object, this predicts a
second, independent *class* per object -- a defect label, orthogonal to the
object class the first classifier predicts. The branch is a second copy of the
head's own classification branch (`cv3`), so "another classify head" is meant
literally.

Label format -- the standard DOTA-style OBB line with a trailing integer:

    cls x1 y1 x2 y2 x3 y3 x4 y4 defect_class

The defect class names come from `defect_names` in the dataset YAML, which also
fixes the branch's output width.
"""

from __future__ import annotations

import numpy as np
import torch
from ultralytics.engine.model import Model

from .data import DefectOBBDataset
from .head import OBB26Defect
from .loss import OBBDefectLoss
from .model import OBBExtraModel, OBBExtraPredictor, OBBExtraTrainer, OBBExtraValidator, model_cfg


def defect_names(data: dict) -> dict:
    """Return the dataset's defect class names, or raise a pointed error if they are missing."""
    names = data.get("defect_names")
    if not names:
        raise ValueError(
            "the dataset YAML must define 'defect_names' for the defect head, i.e.\n"
            "  defect_names:\n    0: ok\n    1: scratch\n    2: hole"
        )
    return names if isinstance(names, dict) else dict(enumerate(names))


class OBBDefectModel(OBBExtraModel):
    """YOLO26 OBB model whose third output is a per-object defect class."""

    head_class = OBB26Defect
    loss_class = OBBDefectLoss
    variant = "defect"


class OBBDefectValidator(OBBExtraValidator):
    """Validator reporting the defect classifier's accuracy and macro F1."""

    dataset_class = DefectOBBDataset
    extra_key = "defect"
    extra_default = 0.0

    def reset_extra(self, model: torch.nn.Module) -> None:
        """Reset the accumulators and read the defect class names off the model or the dataset."""
        super().reset_extra(model)
        names = getattr(model, "defect_names", None) or self.data.get("defect_names")
        self.defect_names = names if isinstance(names, dict) else dict(enumerate(names or []))

    def accumulate_extra(self, pred: torch.Tensor, true: torch.Tensor) -> None:
        """Record the predicted and annotated defect class of matched objects."""
        self.extra_pred.extend(pred.argmax(dim=-1).cpu().tolist())
        self.extra_true.extend(true.round().long().cpu().tolist())

    def defect_stats(self) -> tuple[float, float, int]:
        """Return (accuracy, macro F1, matched count) for the defect classifier."""
        pred = np.asarray(self.extra_pred, dtype=np.int64)
        true = np.asarray(self.extra_true, dtype=np.int64)
        if pred.size == 0:
            return float("nan"), float("nan"), 0
        acc = float((pred == true).mean())
        f1s = []
        for c in np.unique(true):  # macro F1 over the classes that actually appear
            tp = float(((pred == c) & (true == c)).sum())
            fp = float(((pred == c) & (true != c)).sum())
            fn = float(((pred != c) & (true == c)).sum())
            f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        return acc, float(np.mean(f1s)) if f1s else float("nan"), int(pred.size)

    def extra_metrics(self) -> dict[str, float]:
        """Return the defect classifier's metrics."""
        acc, f1, _ = self.defect_stats()
        return {"metrics/defect_acc": acc, "metrics/defect_f1": f1}

    def extra_summary(self) -> str:
        """Summarize the defect classifier, with a per-class breakdown."""
        acc, f1, n = self.defect_stats()
        lines = [f"defect: accuracy {acc:.4f}, macro F1 {f1:.4f}, over {n} matched object(s)"]
        if n:
            pred = np.asarray(self.extra_pred)
            true = np.asarray(self.extra_true)
            for c in sorted(set(true.tolist())):
                m = true == c
                name = self.defect_names.get(c, str(c)) if self.defect_names else str(c)
                lines.append(f"    {name:<12s} n={int(m.sum()):<5d} correct={float((pred[m] == c).mean()):.4f}")
        return "\n".join(lines)


class OBBDefectPredictor(OBBExtraPredictor):
    """Predictor attaching `defect` (class index) and `defect_conf` to each result."""

    def attach_extra(self, result, extra: torch.Tensor):
        """Attach the defect class and its score for each detection.

        `extra` already holds per-class probabilities: the head's `_inference` squashes the branch's logits with
        `sigmoid`, matching the independent one-hot BCE the branch is trained with. Do not re-normalize them --
        a softmax on top would flatten a confident `[1, 0, 0, 0]` to 0.475 and report it as near-chance.
        """
        result.defect_scores = extra
        result.defect = extra.argmax(dim=-1) if extra.shape[1] else torch.zeros(extra.shape[0], dtype=torch.long)
        result.defect_conf = extra.max(dim=-1).values if extra.shape[1] else torch.zeros(extra.shape[0])
        result.defect_names = self.model_defect_names()
        return result

    def model_defect_names(self) -> dict:
        """Find the defect class names, which ride on the checkpoint's model rather than on AutoBackend."""
        for obj in (self.model, getattr(self.model, "model", None)):
            names = getattr(obj, "defect_names", None)
            if names:
                return names if isinstance(names, dict) else dict(enumerate(names))
        return {}


class OBBDefectTrainer(OBBExtraTrainer):
    """Trainer for defect-classifying OBB models."""

    model_class = OBBDefectModel
    dataset_class = DefectOBBDataset
    validator_class = OBBDefectValidator

    def model_kwargs(self) -> dict:
        """Size the defect branch from the dataset's defect classes."""
        return {"n_extra": len(defect_names(self.data))}

    def set_model_attributes(self):
        """Attach the defect class names so they travel with the checkpoint."""
        super().set_model_attributes()
        self.model.defect_names = defect_names(self.data)


class YOLODefect(Model):
    """`ultralytics.YOLO`-style entry point for defect-classifying OBB models.

    Examples:
        >>> from obbq.defect import YOLODefect
        >>> model = YOLODefect()
        >>> model.train(data="datasets/obb-defect/data.yaml", epochs=10, imgsz=320)
    """

    def __init__(self, model=None, verbose: bool = False):
        """Initialize with the defect config by default."""
        super().__init__(model=model or model_cfg("n", "defect"), task="obb", verbose=verbose)

    @property
    def task_map(self) -> dict:
        """Map the OBB task to the defect model, trainer, validator and predictor."""
        return {
            "obb": {
                "model": OBBDefectModel,
                "trainer": OBBDefectTrainer,
                "validator": OBBDefectValidator,
                "predictor": OBBDefectPredictor,
            }
        }
