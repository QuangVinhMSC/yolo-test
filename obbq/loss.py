"""Loss for the quality-aware YOLO26 OBB head.

`v8OBBLoss` supervises box, cls, dfl and angle. This adds a fifth term for the
head's third output. The quality target of an anchor is the quality annotated
on the ground-truth object that the task-aligned assigner matched it to, so the
extra output is trained on exactly the positives the box and class outputs are
trained on, with the same alignment weighting and the same BCE the class output
already uses.
"""

from __future__ import annotations

import torch
from ultralytics.utils.loss import v8OBBLoss
from ultralytics.utils.tal import make_anchors

DEFAULT_QUALITY_GAIN = 1.0


class OBBQualityLoss(v8OBBLoss):
    """Compute box, cls, dfl, angle and quality losses for rotated YOLO models."""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        """Initialize the loss and append the quality term to the reported loss names."""
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.loss_names = (*self.loss_names, "qual_loss")
        self.quality_gain = getattr(self.hyp, "quality", DEFAULT_QUALITY_GAIN)

    @staticmethod
    def split_cls(cls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split the packed `cls` column into (class index, quality).

        The dataset ships quality as a second column of `cls` so that it is filtered and concatenated by every
        existing augmentation alongside the class it belongs to. Labels without a quality column default to 1.0.
        """
        cls = cls.view(-1, cls.shape[-1])
        if cls.shape[1] > 1:
            return cls[:, 0:1], cls[:, 1:2]
        return cls, torch.ones_like(cls)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets, preserving any per-object columns that follow the box.

        Same as `v8OBBLoss.preprocess` except the output width is taken from the targets instead of being fixed at
        six, so the trailing quality column survives into the padded, per-image target tensor and stays aligned with
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
        """Calculate box, cls, dfl, angle and quality loss for oriented bounding box detection."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, angle, quality
        pred_distri, pred_scores, pred_angle, pred_quality = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
            preds["angle"].permute(0, 2, 1).contiguous(),
            preds["quality"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size = pred_angle.shape[0]  # batch size

        dtype = pred_scores.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # targets, with quality carried as a trailing column so the filtering below cannot desynchronize it
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            gt_cls, gt_qual = self.split_cls(batch["cls"])
            targets = torch.cat((batch_idx, gt_cls, batch["bboxes"].view(-1, 5), gt_qual), 1)
            rw, rh = targets[:, 4] * float(imgsz[1]), targets[:, 5] * float(imgsz[0])
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes, gt_quality = targets.split((1, 5, 1), 2)  # cls, xywhr, quality
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a quality-aware OBB dataset.\n"
                "Labels must be 'cls x1 y1 x2 y2 x3 y3 x4 y4 quality' with normalized coordinates and "
                "quality in [0, 1]. See tools/add_quality_column.py to migrate a plain OBB dataset."
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

        # Bbox, angle and quality loss
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
            loss[4] = self.calculate_quality_loss(
                pred_quality, gt_quality, target_gt_idx, fg_mask, weight, target_scores_sum
            )
        else:
            # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            loss[0] += (pred_angle * 0).sum() + pred_distri[..., :0].sum()
            loss[4] += (pred_quality * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= self.hyp.angle  # angle gain
        loss[4] *= self.quality_gain  # quality gain

        return loss * batch_size, dict(zip(self.loss_names, loss.detach()))  # loss(box, cls, dfl, angle, quality)

    def calculate_quality_loss(
        self,
        pred_quality: torch.Tensor,
        gt_quality: torch.Tensor,
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
        weight: torch.Tensor,
        target_scores_sum: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the per-anchor quality loss over assigned positives.

        Args:
            pred_quality (torch.Tensor): Predicted quality logits, (bs, h*w, nq).
            gt_quality (torch.Tensor): Ground-truth quality per object, (bs, max_objects, 1).
            target_gt_idx (torch.Tensor): Index of the object each anchor was assigned to, (bs, h*w).
            fg_mask (torch.Tensor): Foreground mask of assigned anchors, (bs, h*w).
            weight (torch.Tensor): Alignment weight of each positive anchor, (num_positives,).
            target_scores_sum (torch.Tensor): Sum of target scores, used for normalization.

        Returns:
            (torch.Tensor): The calculated quality loss.
        """
        # Gather each positive anchor's target from the object the assigner matched it to
        target_quality = gt_quality.squeeze(-1).gather(1, target_gt_idx)  # (bs, h*w)
        qual_loss = self.bce(pred_quality.squeeze(-1), target_quality)[fg_mask]  # soft-target BCE on positives
        return (qual_loss * weight).sum() / target_scores_sum
