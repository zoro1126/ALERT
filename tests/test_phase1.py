"""
Unit tests for Phase 1 components:
- Preprocessing & relative pose calibration
- FocalLoss & loss factory
- FeatureNormalizer serialization
- Data pipeline compatibility
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from training.losses import FocalLoss, get_criterion
from training.normalizer import FeatureNormalizer
from training.preprocess import (
    center_head_pose,
    clip_window,
    preprocess_window,
)


def test_preprocess_pipeline():
    """preprocess_window must clip outliers and center residual head pose."""
    rng = np.random.default_rng(42)
    window = rng.uniform(0.0, 0.5, size=(120, 12)).astype(np.float32)
    window[:, 7] = 30.0   # static pitch offset on col 7
    window[:, 9] = -15.0  # static yaw offset on col 9

    result = preprocess_window(window)

    # Residual head pose should be centered to near-zero median
    assert abs(np.median(result[:, 7])) < 0.01
    assert abs(np.median(result[:, 9])) < 0.01

    # EAR relative columns should be positive
    assert (result[:, :3] >= 0).all()



def test_clip_window():
    # Construct a window with extreme values for 12 features
    window = np.zeros((10, 12), dtype=np.float32)
    window[0, 0] = 3.5       # EAR_L_rel > 2.5
    window[1, 4] = 1.80      # MAR > 1.2
    window[2, 7] = -120.0    # Pitch < -60
    window[3, 9] = 150.0     # Yaw > 60
    window[4, 10] = 8.5      # Blink rate > 4.0

    clipped = clip_window(window.copy())
    assert clipped[0, 0] == 2.5
    assert clipped[1, 4] == 1.2
    assert clipped[2, 7] == -60.0
    assert clipped[3, 9] == 60.0
    assert clipped[4, 10] == 4.0


def test_center_head_pose():
    # Simulate a window with static camera offset (+30 deg yaw on col 9, -15 deg pitch on col 7)
    window = np.zeros((20, 12), dtype=np.float32)
    static_pitch = -15.0
    static_yaw = 30.0

    # Add static offset plus nodding motion
    window[:, 7] = static_pitch
    window[10:15, 7] += 12.0  # head nod down

    window[:, 9] = static_yaw
    window[5:8, 9] += 5.0    # small drift

    centered = center_head_pose(window.copy())

    # Static camera offset should be neutralized: median must be zero
    assert np.isclose(np.median(centered[:, 7]), 0.0, atol=1e-5)
    assert np.isclose(np.median(centered[:, 9]), 0.0, atol=1e-5)
    # The relative nod spike should remain intact (+12 deg relative to baseline)
    assert np.max(centered[:, 7]) == pytest.approx(12.0, abs=1e-4)


def test_focal_loss():
    batch_size = 16
    n_classes = 3
    logits = torch.randn(batch_size, n_classes, requires_grad=True)
    targets = torch.randint(0, n_classes, (batch_size,))

    loss_fn = FocalLoss(alpha=[1.0, 2.0, 1.2], gamma=2.0, label_smoothing=0.05)
    loss = loss_fn(logits, targets)

    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_ordinal_focal_loss():
    # If target is Alert (0):
    # predicting Drowsy (2) must incur greater loss than predicting Low Vigilant (1)
    target = torch.tensor([0])
    logits_pred_class1 = torch.tensor([[0.0, 5.0, 0.0]], requires_grad=True)  # predicts Low Vigilant
    logits_pred_class2 = torch.tensor([[0.0, 0.0, 5.0]], requires_grad=True)  # predicts Drowsy

    loss_fn = get_criterion("ordinal", lambda_ord=0.5, label_smoothing=0.0)

    loss_class1 = loss_fn(logits_pred_class1, target)
    loss_class2 = loss_fn(logits_pred_class2, target)

    # Distant error (0 vs 2) must be strictly greater than adjacent error (0 vs 1)
    assert loss_class2.item() > loss_class1.item()

    loss_class2.backward()
    assert logits_pred_class2.grad is not None
    assert not torch.isnan(logits_pred_class2.grad).any()


def test_loss_factory():
    crit_ord = get_criterion("ordinal", class_weights=[1.0, 1.15, 1.10])
    assert isinstance(crit_ord, nn.Module)
    assert hasattr(crit_ord, "lambda_ord")

    crit_focal = get_criterion("focal", class_weights=[1.0, 2.0, 1.2], gamma=2.0)
    assert isinstance(crit_focal, FocalLoss)

    crit_ce = get_criterion("ce", class_weights=[1.0, 2.0, 1.2])
    assert isinstance(crit_ce, torch.nn.CrossEntropyLoss)


def test_normalizer_dict_serialization():
    data = np.random.randn(50, 120, 12).astype(np.float32)
    norm = FeatureNormalizer()
    norm.fit(data)

    as_dict = norm.to_dict()
    assert "mean" in as_dict
    assert "std" in as_dict
    assert len(as_dict["mean"]) == 12
    assert len(as_dict["std"]) == 12

    norm_restored = FeatureNormalizer.from_dict(as_dict)
    np.testing.assert_allclose(norm.mean_, norm_restored.mean_)
    np.testing.assert_allclose(norm.std_, norm_restored.std_)

    test_window = np.random.randn(120, 12).astype(np.float32)
    t1 = norm.transform_window(test_window)
    t2 = norm_restored.transform_window(test_window)
    np.testing.assert_allclose(t1, t2)

