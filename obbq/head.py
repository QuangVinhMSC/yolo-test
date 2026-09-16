"""OBB heads that add a third output alongside boxes and classes.

The stock `OBB26` head emits two per-anchor predictions that the loss and the
dataset both know about: the (rotated) box and the class scores, with the angle
riding along as an extra channel of the box.

`OBB26ExtraBranch` adds a third, built with the head's *own* construction
patterns -- `cv5`, a per-feature-level stack shaped exactly like one of the
branches the head already has. No new architecture block is introduced: the
branch reads the same P3/P4/P5 features as the box and class branches, produces
values per anchor, and is concatenated into the same prediction tensor, so it is
decoded, top-k'd and exported exactly like the outputs that were already there.

Two variants ship:

* `OBB26Quality` -- one channel, a scalar in [0, 1], branch shaped like the
  head's angle branch (`cv4`). A regression output.
* `OBB26Defect` -- `nd` channels, a second independent label per object, branch
  shaped like the head's classification branch (`cv3`). A classification output.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import OBB26


class OBB26ExtraBranch(OBB26):
    """YOLO26 OBB head with one extra per-anchor branch.

    Subclasses set `extra_key` -- the name the branch's predictions take in the head's output dict -- and implement
    `build_branch`, which returns the per-level stack. Everything else (the one-to-many / one-to-one plumbing, the
    end-to-end copy, `fuse()`, decoding and export) is inherited and needs no per-variant code.

    Attributes:
        extra_key (str): Key the branch's predictions take in the prediction dict.
        n_extra (int): Number of channels the branch predicts per anchor.
        cv5 (nn.ModuleList): The extra branch, one stack per feature level.
        one2one_cv5 (nn.ModuleList): One-to-one copy of the branch (end-to-end models).
    """

    extra_key: str = "extra"

    def __init__(self, nc: int = 80, ne: int = 1, n_extra: int = 1, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize the head with `n_extra` extra channels.

        Args:
            nc (int): Number of classes.
            ne (int): Number of angle parameters.
            n_extra (int): Number of channels the extra branch predicts per anchor.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        if not isinstance(ch, (list, tuple)) or not len(ch):
            raise TypeError(
                f"{type(self).__name__} expects channels as its last argument, got {ch!r}. Its model YAML row must "
                "pass three head arguments, i.e. '[[16, 19, 22], 1, OBB26, [nc, 1, 1]]' for nc, ne and the extra "
                "channel count."
            )
        super().__init__(nc, ne, reg_max, end2end, ch)
        self.n_extra = n_extra
        self.cv5 = nn.ModuleList(self.build_branch(x, ch) for x in ch)
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    def build_branch(self, c_in: int, ch: tuple) -> nn.Module:
        """Return the extra branch's stack for one feature level."""
        raise NotImplementedError

    @property
    def one2many(self):
        """Return the one-to-many head components."""
        return {**super().one2many, "extra_head": self.cv5}

    @property
    def one2one(self):
        """Return the one-to-one head components."""
        return {**super().one2one, "extra_head": self.one2one_cv5}

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        angle_head: torch.nn.Module,
        extra_head: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        """Concatenate predicted boxes, class probabilities, raw angles and the extra branch's raw logits."""
        preds = super().forward_head(x, box_head, cls_head, angle_head)
        if extra_head is not None:
            bs = x[0].shape[0]  # batch size
            preds[self.extra_key] = torch.cat(
                [extra_head[i](x[i]).view(bs, self.n_extra, -1) for i in range(self.nl)], 2
            )  # logits, squashed at inference like the class scores
        return preds

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Append the extra branch's scores to the decoded boxes, class scores and angles."""
        preds = super()._inference(x)
        return torch.cat([preds, x[self.extra_key].sigmoid()], dim=1)


class OBB26Quality(OBB26ExtraBranch):
    """YOLO26 OBB head whose third output is a per-anchor quality score.

    The branch is shaped like the head's angle branch (`cv4`): `Conv -> Conv -> 1x1 Conv`.

    Examples:
        >>> head = OBB26Quality(nc=2, ne=1, n_extra=1, reg_max=1, end2end=True, ch=(64, 128, 256))
        >>> x = [torch.randn(1, 64, 40, 40), torch.randn(1, 128, 20, 20), torch.randn(1, 256, 10, 10)]
        >>> preds = head(x)
    """

    extra_key = "quality"

    def build_branch(self, c_in: int, ch: tuple) -> nn.Module:
        """Return the angle branch's stack, sized for the quality channels."""
        c5 = max(ch[0] // 4, self.n_extra)
        return nn.Sequential(Conv(c_in, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.n_extra, 1))

    @property
    def nq(self) -> int:
        """Number of quality channels predicted per anchor."""
        return self.n_extra


class OBB26Defect(OBB26ExtraBranch):
    """YOLO26 OBB head whose third output is a second, independent classification.

    The branch is a second copy of the head's *classification* branch (`cv3`) -- the same
    `DWConv -> Conv, DWConv -> Conv, 1x1 Conv` stack -- with `nd` outputs instead of `nc`. It predicts a defect class
    per anchor, independent of the object class the first classifier predicts.

    Examples:
        >>> head = OBB26Defect(nc=2, ne=1, n_extra=4, reg_max=1, end2end=True, ch=(64, 128, 256))
        >>> x = [torch.randn(1, 64, 40, 40), torch.randn(1, 128, 20, 20), torch.randn(1, 256, 10, 10)]
        >>> preds = head(x)
    """

    extra_key = "defect"

    def build_branch(self, c_in: int, ch: tuple) -> nn.Module:
        """Return the classification branch's stack, sized for the defect classes."""
        c5 = max(ch[0], min(self.n_extra, 100))
        return nn.Sequential(
            nn.Sequential(DWConv(c_in, c_in, 3), Conv(c_in, c5, 1)),
            nn.Sequential(DWConv(c5, c5, 3), Conv(c5, c5, 1)),
            nn.Conv2d(c5, self.n_extra, 1),
        )

    @property
    def nd(self) -> int:
        """Number of defect classes predicted per anchor."""
        return self.n_extra

    def bias_init(self):
        """Initialize biases, giving the defect branch a uniform prior over its classes.

        The stock class branch is initialized for a rare-object prior, because most anchors are background. The
        defect branch is only ever supervised on assigned positives, where exactly one of its classes is correct, so
        its prior is 1/nd rather than "almost never".
        """
        super().bias_init()
        prior = -math.log(max(self.n_extra - 1, 1))  # logit(1 / nd)
        for branch in self.cv5 if self.cv5 is not None else []:
            branch[-1].bias.data[:] = prior
        if getattr(self, "one2one_cv5", None) is not None:
            for branch in self.one2one_cv5:
                branch[-1].bias.data[:] = prior
