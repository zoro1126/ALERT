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

from training.losses import FocalLoss, get_criterion
from training.normalizer import FeatureNormalizer
from training.preprocess import (
    adaptive_ear_normalization,
    center_head_pose,
    clip_window,
    preprocess_window,
)


def test_adaptive_ear_normalization():
    """EAR columns should be scaled relative to each window's 90th-percentile."""
    # Simulate a narrow-eyed subject: resting EAR = 0.22 (would look "drowsy" globally)
    window = np.zeros((60, 8), dtype=np.float32)
    window[:, 0] = 0.22   # EAR_L constant
    window[:, 1] = 0.22   # EAR_R constant
    window[:, 2] = 0.22   # EAR_avg constant
    # 5 blink frames (eyes half-closed)
    window[10:15, :3] = 0.10

    normed = adaptive_ear_normalization(window.copy())

    # p90 of EAR_avg ≈ 0.22; after division most frames should be ≈ 1.0
    open_frames = np.delete(normed[:, 2], list(range(10, 15)))
    assert np.allclose(open_frames, open_frames[0], atol=0.01), \
        "Open-eye frames should normalize to same relative value"

    # Blink frames should be less than open frames (ratio preserved)
    assert normed[10, 2] < normed[0, 2], "Blink frames must remain smaller than open frames"

    # Values should be positive
    assert (normed[:, :3] >= 0).all(), "Normalized EAR must be non-negative"


def test_preprocess_pipeline_order():
    """preprocess_window must clip → EAR normalize → head pose center."""
    rng = np.random.default_rng(42)
    window = rng.uniform(0.0, 0.5, size=(60, 8)).astype(np.float32)
    window[:, 5] = 30.0  # large static yaw offset
    window[:, 6] = -15.0

    result = preprocess_window(window)

    # Head pose should be centered to near-zero median
    assert abs(np.median(result[:, 5])) < 0.01
    assert abs(np.median(result[:, 6])) < 0.01

    # EAR columns should be positive
    assert (result[:, :3] >= 0).all()


def test_clip_window():
    # Construct a window with extreme values
    window = np.zeros((10, 8), dtype=np.float32)
    window[0, 0] = 0.95      # EAR > 0.6
    window[1, 3] = 1.80      # MAR > 0.8
    window[2, 5] = -120.0    # Pitch < -60
    window[3, 6] = 150.0     # Yaw > 90
    window[4, 7] = 8.5       # Blink rate > 4.0

    clipped = clip_window(window.copy())
    assert clipped[0, 0] == 0.6
    assert clipped[1, 3] == 0.8
    assert clipped[2, 5] == -60.0
    assert clipped[3, 6] == 90.0
    assert clipped[4, 7] == 4.0


def test_center_head_pose():
    # Simulate a window with static camera offset (+30 deg yaw, -15 deg pitch)
    window = np.zeros((20, 8), dtype=np.float32)
    static_pitch = -15.0
    static_yaw = 30.0

    # Add static offset plus nodding motion
    window[:, 5] = static_pitch
    window[10:15, 5] += 12.0  # head nod down

    window[:, 6] = static_yaw
    window[5:8, 6] += 5.0    # small drift

    centered = center_head_pose(window.copy())

    # Static camera offset should be neutralized: median must be zero
    assert np.isclose(np.median(centered[:, 5]), 0.0, atol=1e-5)
    assert np.isclose(np.median(centered[:, 6]), 0.0, atol=1e-5)
    # The relative nod spike should remain intact (+12 deg relative to baseline)
    assert np.max(centered[:, 5]) == pytest.approx(12.0, abs=1e-4)


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


def test_loss_factory():
    crit_focal = get_criterion("focal", class_weights=[1.0, 2.0, 1.2], gamma=2.0)
    assert isinstance(crit_focal, FocalLoss)

    crit_ce = get_criterion("ce", class_weights=[1.0, 2.0, 1.2])
    assert isinstance(crit_ce, torch.nn.CrossEntropyLoss)


def test_normalizer_dict_serialization():
    data = np.random.randn(50, 60, 8).astype(np.float32)
    norm = FeatureNormalizer()
    norm.fit(data)

    as_dict = norm.to_dict()
    assert "mean" in as_dict
    assert "std" in as_dict
    assert len(as_dict["mean"]) == 8
    assert len(as_dict["std"]) == 8

    norm_restored = FeatureNormalizer.from_dict(as_dict)
    np.testing.assert_allclose(norm.mean_, norm_restored.mean_)
    np.testing.assert_allclose(norm.std_, norm_restored.std_)

    test_window = np.random.randn(60, 8).astype(np.float32)
    t1 = norm.transform_window(test_window)
    t2 = norm_restored.transform_window(test_window)
    np.testing.assert_allclose(t1, t2)
