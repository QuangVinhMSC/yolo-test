"""Checks for the defect-class variant of the third output.

Run with: python -m pytest tests/test_defect.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import obbq
from obbq.data import make_check_defect, verify_image_label_extra
from obbq.defect import OBBDefectModel, defect_names

ND = 4


@pytest.fixture(scope="module", autouse=True)
def _register():
    obbq.register()


def _write_sample(tmp_path, rows, size=64):
    import cv2

    img, lbl = tmp_path / "im.jpg", tmp_path / "im.txt"
    cv2.imwrite(str(img), np.full((size, size, 3), 40, dtype=np.uint8))
    lbl.write_text("\n".join(rows) + "\n")
    return str(img), str(lbl)


def test_label_parsing_reads_defect_class(tmp_path):
    """A 10-column line yields the polygon plus its defect class index."""
    im, lb = _write_sample(tmp_path, ["1 0.10 0.10 0.30 0.10 0.30 0.20 0.10 0.20 3"])
    _, box, _, segments, defect, _, nf, _, nc, msg = verify_image_label_extra(
        (im, lb, "", False, 2, 0, 0, False, make_check_defect(ND))
    )
    assert (nf, nc) == (1, 0), msg
    assert box[0, 0] == 1  # object class is untouched
    assert segments[0].shape == (4, 2)
    assert defect[0, 0] == 3  # defect class is independent of it


def test_label_parsing_rejects_unknown_defect_class(tmp_path):
    """A defect index outside the dataset's defect_names marks the pair corrupt."""
    im, lb = _write_sample(tmp_path, ["0 0.10 0.10 0.30 0.10 0.30 0.20 0.10 0.20 9"])
    result = verify_image_label_extra((im, lb, "", False, 2, 0, 0, False, make_check_defect(ND)))
    assert result[8] == 1  # nc, corrupt
    assert "defect class must be in 0-3" in result[9]


def test_label_parsing_rejects_non_integer_defect_class(tmp_path):
    """A fractional defect index is a quality label in disguise, and is refused."""
    im, lb = _write_sample(tmp_path, ["0 0.10 0.10 0.30 0.10 0.30 0.20 0.10 0.20 0.5"])
    result = verify_image_label_extra((im, lb, "", False, 2, 0, 0, False, make_check_defect(ND)))
    assert result[8] == 1
    assert "integer index" in result[9]


def test_defect_branch_is_a_copy_of_the_class_branch():
    """The defect branch has the same structure as the head's own classifier, with nd outputs."""
    head = obbq.OBB26Defect(nc=2, ne=1, n_extra=ND, reg_max=1, end2end=True, ch=(64, 128, 256))
    cls_branch, defect_branch = head.cv3[0], head.cv5[0]
    assert [type(m).__name__ for m in cls_branch] == [type(m).__name__ for m in defect_branch]
    assert defect_branch[-1].out_channels == ND
    assert cls_branch[-1].out_channels == head.nc


def test_head_output_layout():
    """The defect scores are the last nd channels, after box, classes and angle."""
    nc = 2
    head = obbq.OBB26Defect(nc=nc, ne=1, n_extra=ND, reg_max=1, end2end=True, ch=(32, 64, 128))
    head.stride = torch.tensor([8.0, 16.0, 32.0])
    head.eval()
    x = [torch.randn(1, 32, 16, 16), torch.randn(1, 64, 8, 8), torch.randn(1, 128, 4, 4)]
    y, preds = head(x)
    assert y.shape[1] == 4 + nc + 1 + ND
    logits = preds["one2many"]["defect"]
    assert logits.shape == (1, ND, y.shape[2])
    assert torch.allclose(y[:, -ND:, :], logits.sigmoid(), atol=1e-6)


def test_bias_init_uses_a_uniform_prior():
    """The defect branch starts at 1/nd per class, not at the rare-object prior of the class branch."""
    head = obbq.OBB26Defect(nc=2, ne=1, n_extra=ND, reg_max=1, end2end=True, ch=(32, 64, 128))
    head.stride = torch.tensor([8.0, 16.0, 32.0])
    head.bias_init()
    assert torch.allclose(head.cv5[0][-1].bias.data.sigmoid(), torch.full((ND,), 1 / ND), atol=1e-6)


def test_model_sizes_the_branch_from_the_dataset():
    """nd comes from the dataset, overriding the YAML's placeholder."""
    model = OBBDefectModel(nc=2, n_extra=ND, verbose=False)
    assert model.model[-1].nd == ND


def test_defect_names_demands_the_dataset_key():
    """A dataset without defect_names fails with an actionable message, not a shape error."""
    assert defect_names({"defect_names": {0: "ok", 1: "scratch"}}) == {0: "ok", 1: "scratch"}
    assert defect_names({"defect_names": ["ok", "scratch"]}) == {0: "ok", 1: "scratch"}
    with pytest.raises(ValueError, match="must define 'defect_names'"):
        defect_names({"names": {0: "rect"}})


def test_defect_loss_is_minimized_at_the_annotated_class():
    """The defect term bottoms out when the predicted class matches the annotation."""
    from ultralytics.utils import DEFAULT_CFG

    model = OBBDefectModel(nc=2, n_extra=ND, verbose=False)
    model.args = DEFAULT_CFG
    criterion = obbq.OBBDefectLoss(model)

    target = 2
    gt_defect = torch.full((1, 1, 1), float(target))
    target_gt_idx = torch.zeros((1, 4), dtype=torch.long)
    fg_mask = torch.tensor([[True, True, False, False]])
    weight = torch.ones(2)
    scores_sum = torch.tensor(2.0)

    def value(predicted_class):
        logits = torch.full((1, 4, ND), -4.0)
        logits[..., predicted_class] = 4.0
        return criterion.calculate_extra_loss(
            logits, gt_defect, target_gt_idx, fg_mask, weight, scores_sum
        ).item()

    at_target = value(target)
    assert all(at_target < value(c) for c in range(ND) if c != target)


def test_defect_loss_ignores_background_anchors():
    """Anchors with no assigned object contribute nothing to the defect term."""
    from ultralytics.utils import DEFAULT_CFG

    model = OBBDefectModel(nc=2, n_extra=ND, verbose=False)
    model.args = DEFAULT_CFG
    criterion = obbq.OBBDefectLoss(model)

    gt_defect = torch.zeros((1, 1, 1))
    target_gt_idx = torch.zeros((1, 4), dtype=torch.long)
    weight = torch.ones(1)
    scores_sum = torch.tensor(1.0)
    logits = torch.zeros((1, 4, ND))

    only_first = torch.tensor([[True, False, False, False]])
    loss_a = criterion.calculate_extra_loss(logits, gt_defect, target_gt_idx, only_first, weight, scores_sum)
    # Make the background anchors wildly wrong; the loss must not move.
    logits[:, 1:, :] = 9.0
    loss_b = criterion.calculate_extra_loss(logits, gt_defect, target_gt_idx, only_first, weight, scores_sum)
    assert torch.allclose(loss_a, loss_b)
