"""Losses for OBB heads that carry a third output.

`v8OBBLoss` supervises box, cls, dfl and angle. `OBBExtraLoss` adds a fifth term
for the head's third output. The target of an anchor is the extra label of the
ground-truth object that the task-aligned assigner matched it to, so the extra
output is trained on exactly the positives the box and class outputs are trained
on, with the same alignment weighting.

Subclasses supply the head key they read, the name the term is reported under,
and how the term itself is computed:

* `OBBQualityLoss` -- soft-target BCE against a scalar in [0, 1].
* `OBBDefectLoss`  -- one-hot BCE over defect classes, the same way the stock
  class term is computed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8OBBLoss
from ultralytics.utils.tal import make_anchors

DEFAULT_QUALITY_GAIN = 1.0
DEFAULT_DEFECT_GAIN = 1.0


class OBBExtraLoss(v8OBBLoss):
    """Compute box, cls, dfl, angle and one extra loss term for rotated YOLO models.

    Attributes:
        extra_key (str): Key of the extra branch in the head's prediction dict.
        extra_loss_name (str): Name the extra term is reported under.
        gain_key (str): Hyperparameter name holding the extra term's loss gain.
        default_gain (float): Gain used when the hyperparameter is not registered.
    """

    extra_key: str = "extra"
    extra_loss_name: str = "extra_loss"
    gain_key: str = "extra"
    default_gain: float = 1.0

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        """Initialize the loss and append the extra term to the reported loss names."""
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.loss_names = (*self.loss_names, self.extra_loss_name)
        self.extra_gain = getattr(self.hyp, self.gain_key, self.default_gain)

    @staticmethod
    def split_cls(cls: torch.Tensor, default: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Split the packed `cls` column into (class index, extra label).

        The dataset ships the extra label as a second column of `cls` so that it is filtered and concatenated by
        every existing augmentation alongside the class it belongs to. Labels without that column fall back to
        `default`.
        """
        cls = cls.view(-1, cls.shape[-1])
        if cls.shape[1] > 1:
            return cls[:, 0:1], cls[:, 1:2]
        return cls, torch.full_like(cls, default)

    def calculate_extra_loss(
        self,
        pred_extra: torch.Tensor,
        gt_extra: torch.Tensor,
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
        weight: torch.Tensor,
        target_scores_sum: torch.Tensor,
    ) -> torch.Tensor:
        """Return the extra term, computed over assigned positives.

        Args:
            pred_extra (torch.Tensor): Predicted logits, (bs, h*w, n_extra).
            gt_extra (torch.Tensor): Ground-truth extra label per object, (bs, max_objects, 1).
            target_gt_idx (torch.Tensor): Index of the object each anchor was assigned to, (bs, h*w).
            fg_mask (torch.Tensor): Foreground mask of assigned anchors, (bs, h*w).
            weight (torch.Tensor): Alignment weight of each positive anchor, (num_positives,).
            target_scores_sum (torch.Tensor): Sum of target scores, used for normalization.

        Returns:
            (torch.Tensor): The calculated loss term.
        """
        raise NotImplementedError

    def gather_targets(self, gt_extra: torch.Tensor, target_gt_idx: torch.Tensor) -> torch.Tensor:
        """Gather each anchor's extra target from the object the assigner matched it to, (bs, h*w)."""
        return gt_extra.squeeze(-1).gather(1, target_gt_idx)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets, preserving any per-object columns that follow the box.

        Same as `v8OBBLoss.preprocess` except the output width is taken from the targets instead of being fixed at
        six, so the trailing extra column survives into the padded, per-image target tensor and stays aligned with
        the boxes through the small-box filtering done by the caller.
        """
        nt = targets.shape[1] - 1  # per-object columns, excluding the batch index
        if targets.shape[0] == 0:
            return torch.zeros(batch_size, 0, nt, device=self.device)
        batch_idx = targets[:, 0].long()  # image index
        _, counts = batch_idx.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        out = torch.zeros(batch_size, counts.max(), nt, device=self.device)
        packed_targets = targets[:, 1:].clone()
        packed_targets[:, 1:5].mul_(scale_tensor)
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
        offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
        offsets = offsets.cumsum(0)
        within_idx = torch.arange(len(targets), device=self.device) - offsets[batch_idx]
        out[batch_idx, within_idx] = packed_targets
        return out

    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate box, cls, dfl, angle and the extra loss term for oriented bounding box detection."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, angle, extra
        pred_distri, pred_scores, pred_angle, pred_extra = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
            preds["angle"].permute(0, 2, 1).contiguous(),
            preds[self.extra_key].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size = pred_angle.shape[0]  # batch size

        dtype = pred_scores.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # targets, with the extra label carried as a trailing column so the filtering below cannot desynchronize it
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            gt_cls, gt_label = self.split_cls(batch["cls"])
            targets = torch.cat((batch_idx, gt_cls, batch["bboxes"].view(-1, 5), gt_label), 1)
            rw, rh = targets[:, 4] * float(imgsz[1]), targets[:, 5] * float(imgsz[0])
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes, gt_extra = targets.split((1, 5, 1), 2)  # cls, xywhr, extra
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted, or not a dataset for this head.\n"
                "Labels must be 'cls x1 y1 x2 y2 x3 y3 x4 y4 <extra>' with normalized coordinates. "
                "See tools/add_quality_column.py to migrate a plain OBB dataset."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xywhr, (b, h*w, 5)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        bboxes_for_assigner[..., :4] *= stride_tensor  # only the first four elements need to be scaled
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # BCE
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum

        # Bbox, angle and extra loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
            weight = target_scores[fg_mask].sum(-1)
            loss[3] = self.calculate_angle_loss(pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum)
            loss[4] = self.calculate_extra_loss(
                pred_extra, gt_extra, target_gt_idx, fg_mask, weight, target_scores_sum
            )
        else:
            # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            loss[0] += (pred_angle * 0).sum() + pred_distri[..., :0].sum()
            loss[4] += (pred_extra * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= self.hyp.angle  # angle gain
        loss[4] *= self.extra_gain  # extra gain

        return loss * batch_size, dict(zip(self.loss_names, loss.detach()))  # loss(box, cls, dfl, angle, extra)


class OBBQualityLoss(OBBExtraLoss):
    """Add a per-object quality regression term to the OBB losses."""

    extra_key = "quality"
    extra_loss_name = "qual_loss"
    gain_key = "quality"
    default_gain = DEFAULT_QUALITY_GAIN

    def calculate_extra_loss(self, pred_extra, gt_extra, target_gt_idx, fg_mask, weight, target_scores_sum):
        """Soft-target BCE between the predicted and the annotated quality of each positive anchor."""
        target_quality = self.gather_targets(gt_extra, target_gt_idx)  # (bs, h*w)
        qual_loss = self.bce(pred_extra.squeeze(-1), target_quality)[fg_mask]
        return (qual_loss * weight).sum() / target_scores_sum

    # Backwards-compatible alias for the name this term had before the base class was extracted.
    calculate_quality_loss = calculate_extra_loss


class OBBDefectLoss(OBBExtraLoss):
    """Add a second, independent classification term to the OBB losses.

    The defect class of a positive anchor is supervised with one-hot BCE -- the same loss the stock class term uses,
    just over the defect classes and restricted to assigned positives, since an anchor only has a defect class when
    it has an object.
    """

    extra_key = "defect"
    extra_loss_name = "def_loss"
    gain_key = "defect"
    default_gain = DEFAULT_DEFECT_GAIN

    @staticmethod
    def split_cls(cls: torch.Tensor, default: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Split `cls`, defaulting a missing defect column to class 0 rather than to 1."""
        return OBBExtraLoss.split_cls(cls, default=default)

    def calculate_extra_loss(self, pred_extra, gt_extra, target_gt_idx, fg_mask, weight, target_scores_sum):
        """One-hot BCE over the defect classes of each positive anchor."""
        target_defect = self.gather_targets(gt_extra, target_gt_idx).long().clamp_(0, pred_extra.shape[-1] - 1)
        one_hot = F.one_hot(target_defect, pred_extra.shape[-1]).to(pred_extra.dtype)  # (bs, h*w, nd)
        def_loss = self.bce(pred_extra, one_hot).sum(-1)[fg_mask]  # sum over classes, positives only
        return (def_loss * weight).sum() / target_scores_sum
