"""Quality-aware YOLO26 OBB head.

The stock `OBB26` head emits two per-anchor predictions that the loss and the
dataset both know about: the (rotated) box and the class scores, with the angle
riding along as an extra channel of the box. This module adds a third one,
`quality`, a per-anchor scalar in [0, 1].

The quality branch is built with the head's *own* construction pattern -- the
same `Conv -> Conv -> 1x1 Conv` stack the head already uses for its angle
branch (`cv4`), replicated per feature level. No new architecture block is
introduced: the branch reads the same P3/P4/P5 features as the box and class
branches, produces one value per anchor, and is concatenated into the same
prediction tensor, so it is decoded, top-k'd and exported exactly like the
outputs that were already there.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import OBB26


class OBB26Quality(OBB26):
    """YOLO26 OBB head with a third output: a per-anchor quality score.

    Attributes:
        nq (int): Number of quality channels predicted per anchor.
        cv5 (nn.ModuleList): Quality branch, one stack per feature level.
        one2one_cv5 (nn.ModuleList): One-to-one copy of the quality branch (end-to-end models).

    Examples:
        >>> head = OBB26Quality(nc=2, ne=1, nq=1, reg_max=1, end2end=True, ch=(64, 128, 256))
        >>> x = [torch.randn(1, 64, 40, 40), torch.randn(1, 128, 20, 20), torch.randn(1, 256, 10, 10)]
        >>> preds = head(x)
    """

    def __init__(self, nc: int = 80, ne: int = 1, nq: int = 1, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize the head with `nq` extra quality channels.

        Args:
            nc (int): Number of classes.
            ne (int): Number of angle parameters.
            nq (int): Number of quality parameters predicted per anchor.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        if not isinstance(ch, (list, tuple)) or not len(ch):
            raise TypeError(
                f"OBB26Quality expects channels as its last argument, got {ch!r}. Its model YAML row must pass "
                "three head arguments, i.e. '[[16, 19, 22], 1, OBB26, [nc, 1, 1]]' for nc, ne and nq."
            )
        super().__init__(nc, ne, reg_max, end2end, ch)
        self.nq = nq  # number of quality parameters

        # Same branch pattern the head already uses for its angle output (cv4).
        c5 = max(ch[0] // 4, self.nq)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nq, 1)) for x in ch)
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    @property
    def one2many(self):
        """Return the one-to-many head components."""
        return {**super().one2many, "quality_head": self.cv5}

    @property
    def one2one(self):
        """Return the one-to-one head components."""
        return {**super().one2one, "quality_head": self.one2one_cv5}

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        angle_head: torch.nn.Module,
        quality_head: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        """Concatenate predicted boxes, class probabilities, raw angles and raw quality logits."""
        preds = super().forward_head(x, box_head, cls_head, angle_head)
        if quality_head is not None:
            bs = x[0].shape[0]  # batch size
            preds["quality"] = torch.cat(
                [quality_head[i](x[i]).view(bs, self.nq, -1) for i in range(self.nl)], 2
            )  # quality logits, squashed at inference like the class scores
        return preds

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Append the quality score to the decoded boxes, class scores and angles."""
        preds = super()._inference(x)
        return torch.cat([preds, x["quality"].sigmoid()], dim=1)
