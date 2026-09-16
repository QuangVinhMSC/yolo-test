"""Checks for the quality-aware OBB pipeline.

Run with: python -m pytest tests/test_quality.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import obbq
from obbq.data import QUALITY_COLUMNS, verify_image_label_quality


@pytest.fixture(scope="module", autouse=True)
def _register():
    obbq.register()


def _write_sample(tmp_path, rows, size=64):
    import cv2

    img = tmp_path / "im.jpg"
    lbl = tmp_path / "im.txt"
    cv2.imwrite(str(img), np.full((size, size, 3), 40, dtype=np.uint8))
    lbl.write_text("\n".join(rows) + "\n")
    return str(img), str(lbl)


def test_label_parsing_reads_quality(tmp_path):
    """A 10-column line yields the polygon plus its quality."""
    row = "1 0.10 0.10 0.30 0.10 0.30 0.20 0.10 0.20 0.375"
    im, lb = _write_sample(tmp_path, [row])
    _, box, _, segments, quality, nm, nf, ne, nc, msg = verify_image_label_quality(
        (im, lb, "", False, 2, 0, 0, False)
    )
    assert (nm, nf, ne, nc) == (0, 1, 0, 0), msg
    assert box[0, 0] == 1
    assert segments[0].shape == (4, 2)
    assert quality.shape == (1, 1)
    assert quality[0, 0] == pytest.approx(0.375)


def test_label_parsing_defaults_plain_obb_to_one(tmp_path):
    """A 9-column (plain OBB) line still loads, with quality 1.0."""
    row = "0 0.10 0.10 0.30 0.10 0.30 0.20 0.10 0.20"
    im, lb = _write_sample(tmp_path, [row])
    *_, quality, _, nf, _, nc, msg = verify_image_label_quality((im, lb, "", False, 2, 0, 0, False))
    assert (nf, nc) == (1, 0), msg
    assert quality[0, 0] == pytest.approx(1.0)


def test_label_parsing_rejects_out_of_range_quality(tmp_path):
    """Quality outside [0, 1] marks the pair corrupt instead of training on it."""
    row = "0 0.10 0.10 0.30 0.10 0.30 0.20 0.10 0.20 1.90"
    im, lb = _write_sample(tmp_path, [row])
    result = verify_image_label_quality((im, lb, "", False, 2, 0, 0, False))
    assert result[8] == 1  # nc, corrupt
    assert "quality must be in [0, 1]" in result[9]


def test_head_output_layout():
    """Quality is the last channel, after box, classes and angle."""
    nc = 3
    head = obbq.OBB26Quality(nc=nc, ne=1, nq=1, reg_max=1, end2end=True, ch=(32, 64, 128))
    head.stride = torch.tensor([8.0, 16.0, 32.0])
    head.eval()
    x = [torch.randn(1, 32, 16, 16), torch.randn(1, 64, 8, 8), torch.randn(1, 128, 4, 4)]
    y, preds = head(x)
    assert y.shape[1] == 4 + nc + 1 + 1  # xywh, classes, angle, quality
    quality = preds["one2many"]["quality"]
    assert quality.shape == (1, 1, y.shape[2])
    # the decoded channel is the sigmoid of the raw logits
    assert torch.allclose(y[:, -1, :], quality.sigmoid()[:, 0, :], atol=1e-6)


def test_head_yaml_arity_is_checked():
    """Passing the stock two-argument head row fails loudly instead of misbinding nq."""
    with pytest.raises(TypeError, match="expects channels"):
        obbq.OBB26Quality(3, 1, 1, 1, True)


def test_quality_survives_a_dropped_instance():
    """Filtering instances by index filters quality with them, because it rides in cls."""
    cls = np.array([[0.0, 0.1], [1.0, 0.9], [0.0, 0.5]], dtype=np.float32)
    keep = np.array([0, 2])
    assert np.allclose(cls[keep][:, 1], [0.1, 0.5])


def test_loss_splits_cls_into_class_and_quality():
    """split_cls recovers both columns, and defaults to 1.0 for plain labels."""
    packed = torch.tensor([[1.0, 0.25], [0.0, 0.75]])
    cls, quality = obbq.OBBQualityLoss.split_cls(packed)
    assert torch.allclose(cls, torch.tensor([[1.0], [0.0]]))
    assert torch.allclose(quality, torch.tensor([[0.25], [0.75]]))

    plain = torch.tensor([[1.0], [0.0]])
    cls, quality = obbq.OBBQualityLoss.split_cls(plain)
    assert torch.allclose(cls, plain)
    assert torch.allclose(quality, torch.ones_like(plain))


def test_quality_loss_is_minimized_at_the_target():
    """The quality term bottoms out where the predicted score equals the annotated one."""
    from ultralytics.utils import DEFAULT_CFG

    model = obbq.OBBQualityModel(obbq.model_cfg("n"), nc=2, verbose=False)
    model.args = DEFAULT_CFG  # the criterion reads its gains off the model, as the trainer attaches them
    criterion = obbq.OBBQualityLoss(model)

    target = 0.3
    gt_quality = torch.full((1, 1, 1), target)
    target_gt_idx = torch.zeros((1, 4), dtype=torch.long)
    fg_mask = torch.tensor([[True, True, False, False]])
    weight = torch.ones(2)
    scores_sum = torch.tensor(2.0)

    def value(q):
        logit = torch.logit(torch.tensor(q))
        pred = logit.expand(1, 4, 1)
        return criterion.calculate_quality_loss(pred, gt_quality, target_gt_idx, fg_mask, weight, scores_sum).item()

    at_target = value(target)
    assert at_target < value(0.1)
    assert at_target < value(0.6)
    assert at_target < value(0.9)
