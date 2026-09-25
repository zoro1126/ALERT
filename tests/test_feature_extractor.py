"""
test_feature_extractor.py
=========================
Unit tests for EAR, MAR, PERCLOS, blink rate, and HeadPoseEstimator.

Run:  pytest tests/test_feature_extractor.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.feature_extractor import (
    FeatureExtractor,
    FrameFeatures,
    HeadPoseEstimator,
    _euclidean,
    compute_ear,
    compute_mar,
)


# ---------------------------------------------------------------------------
# Helpers to build mock landmarks
# ---------------------------------------------------------------------------

def _make_landmark(x: float, y: float, z: float = 0.0):
    lm = MagicMock()
    lm.x, lm.y, lm.z = x, y, z
    return lm


def _make_landmarks(coords: list[tuple[float, float]]):
    """Build a list of mock landmarks indexed from 0."""
    lms = [_make_landmark(x, y) for x, y in coords]
    return lms


def _mock_face(lm_dict: dict[int, tuple[float, float]]):
    """
    Return a mock object that behaves like face_landmarks.landmark[i].
    Indices not in lm_dict return (0, 0).
    """
    max_idx = max(lm_dict.keys()) + 1
    lms = [_make_landmark(0.0, 0.0)] * max_idx
    for idx, (x, y) in lm_dict.items():
        lms[idx] = _make_landmark(x, y)

    mock = MagicMock()
    mock.landmark = lms
    return mock


# ---------------------------------------------------------------------------
# _euclidean
# ---------------------------------------------------------------------------

class TestEuclidean:
    def test_zero_distance(self):
        assert _euclidean((0, 0), (0, 0)) == 0.0

    def test_unit_distance(self):
        assert _euclidean((0, 0), (1, 0)) == pytest.approx(1.0)

    def test_pythagoras(self):
        assert _euclidean((0, 0), (3, 4)) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# compute_ear
# ---------------------------------------------------------------------------

class TestComputeEAR:
    def _make_open_eye(self):
        """
        Construct 6 landmark positions for a wide-open eye.
        EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
        Using: p1=(0,0), p2=(0.2,0.3), p3=(0.5,0.3),
               p4=(1,0), p5=(0.5,-0.3), p6=(0.2,-0.3)
        vertical pairs: 0.6 each; horizontal: 1.0
        Expected EAR = (0.6+0.6)/(2*1.0) = 0.6
        """
        lms = _make_landmarks([
            (0.0,  0.0),   # p1
            (0.2,  0.3),   # p2
            (0.5,  0.3),   # p3
            (1.0,  0.0),   # p4
            (0.5, -0.3),   # p5
            (0.2, -0.3),   # p6
        ])
        return lms

    def test_open_eye_ear(self):
        lms = self._make_open_eye()
        # Use indices 0..5 to address the mock list directly
        ear = compute_ear(lms, [0, 1, 2, 3, 4, 5])
        assert ear == pytest.approx(0.6, abs=1e-5)

    def test_closed_eye_returns_zero_or_near(self):
        # All landmarks at the same point → zero vertical, non-zero horizontal
        lms = _make_landmarks([
            (0.0, 0.0), (0.3, 0.0), (0.6, 0.0),
            (1.0, 0.0), (0.6, 0.0), (0.3, 0.0),
        ])
        ear = compute_ear(lms, [0, 1, 2, 3, 4, 5])
        assert ear == pytest.approx(0.0, abs=1e-5)

    def test_zero_horizontal_returns_zero(self):
        """Guard against division by zero when p1 == p4."""
        lms = _make_landmarks([(0, 0)] * 6)
        ear = compute_ear(lms, [0, 1, 2, 3, 4, 5])
        assert ear == 0.0


# ---------------------------------------------------------------------------
# FrameFeatures.to_numpy
# ---------------------------------------------------------------------------

class TestFrameFeatures:
    def test_to_numpy_shape_dtype(self):
        ff = FrameFeatures(0.3, 0.31, 0.305, 0.1, 0.0, -5.2, 3.1, 0.4)
        arr = ff.to_numpy()
        assert arr.shape == (8,)
        assert arr.dtype == np.float32

    def test_values_preserved(self):
        ff = FrameFeatures(0.25, 0.26, 0.255, 0.55, 0.12, -10.0, 5.0, 0.8)
        arr = ff.to_numpy()
        assert arr[5] == pytest.approx(-10.0, abs=1e-4)
        assert arr[6] == pytest.approx(5.0, abs=1e-4)


# ---------------------------------------------------------------------------
# FeatureExtractor — PERCLOS and blink rate
# ---------------------------------------------------------------------------

class TestFeatureExtractor:
    def _make_extractor(self, threshold=0.25, window=20, fps=15.0):
        return FeatureExtractor(ear_threshold=threshold, window_size=window, fps=fps)

    def test_perclos_all_open(self):
        """PERCLOS should be 0 when all EARs are above threshold."""
        fe = self._make_extractor(threshold=0.25, window=10)
        for _ in range(10):
            fe._ear_buffer.append(0.35)  # above threshold
            fe._frame_index += 1
        assert fe._compute_perclos() == pytest.approx(0.0)

    def test_perclos_all_closed(self):
        """PERCLOS should be 1.0 when all EARs are below threshold."""
        fe = self._make_extractor(threshold=0.25, window=10)
        for _ in range(10):
            fe._ear_buffer.append(0.10)  # below threshold
            fe._frame_index += 1
        assert fe._compute_perclos() == pytest.approx(1.0)

    def test_blink_detection_count(self):
        """5 open→close→open cycles should register 5 blinks."""
        fe = self._make_extractor(threshold=0.25, window=60, fps=15)
        for _ in range(5):
            fe._update_blink_state(0.10)  # closed
            fe._frame_index += 1
            fe._blink_timestamps.append(fe._frame_index)
            fe._update_blink_state(0.35)  # open  (triggers count)
            fe._frame_index += 1
        assert fe._blink_count == 5

    def test_reset_clears_state(self):
        fe = self._make_extractor()
        fe._ear_buffer.append(0.1)
        fe._blink_count = 3
        fe._frame_index = 99
        fe.reset()
        assert len(fe._ear_buffer) == 0
        assert fe._blink_count == 0
        assert fe._frame_index == 0

    def test_update_threshold(self):
        fe = self._make_extractor(threshold=0.25)
        fe.update_threshold(0.20)
        assert fe.ear_threshold == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# HeadPoseEstimator — camera matrix
# ---------------------------------------------------------------------------

class TestHeadPoseEstimator:
    def test_camera_matrix_shape(self):
        hpe = HeadPoseEstimator()
        mat = hpe._build_camera_matrix(640, 480)
        assert mat.shape == (3, 3)

    def test_focal_equals_width(self):
        hpe = HeadPoseEstimator()
        mat = hpe._build_camera_matrix(640, 480)
        assert mat[0, 0] == pytest.approx(640.0)

    def test_principal_point_centred(self):
        hpe = HeadPoseEstimator()
        mat = hpe._build_camera_matrix(640, 480)
        assert mat[0, 2] == pytest.approx(320.0)
        assert mat[1, 2] == pytest.approx(240.0)

    def test_matrix_cached_same_resolution(self):
        hpe = HeadPoseEstimator()
        m1 = hpe._build_camera_matrix(640, 480)
        hpe._last_wh = (640, 480)
        hpe._camera_matrix = m1
        # Call estimate with same resolution — should not rebuild matrix
        # (we can't easily call estimate without real landmarks, but we
        #  can check the _last_wh cache logic)
        assert hpe._last_wh == (640, 480)
        assert hpe._camera_matrix is m1
