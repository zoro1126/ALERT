"""
feature_extractor.py
====================
Computes per-frame fatigue features from MediaPipe FaceMesh landmarks.

Features extracted
------------------
0  EAR_left    – Eye Aspect Ratio, left eye
1  EAR_right   – Eye Aspect Ratio, right eye
2  EAR_avg     – Mean of left & right EAR
3  MAR         – Mouth Aspect Ratio (yawn proxy)
4  PERCLOS     – % frames in rolling buffer where EAR_avg < threshold
5  head_pitch  – Vertical head tilt in degrees (down = positive)
6  head_yaw    – Horizontal head turn in degrees (right = positive)
7  blink_rate  – Blinks per second over rolling buffer

References
----------
- Soukupová & Čech (2016): EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
- MediaPipe FaceMesh landmark map:
  https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
"""

from __future__ import annotations

import math
from collections import deque
from typing import NamedTuple

import numpy as np

from utils.config import Config

# ---------------------------------------------------------------------------
# MediaPipe landmark indices for relevant facial regions
# (verified against the 468-point canonical face model)
# ---------------------------------------------------------------------------

# Left eye  – clockwise from outer corner: p1..p6
_LEFT_EYE = [362, 385, 387, 263, 373, 380]
# Right eye – clockwise from outer corner: p1..p6
_RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Mouth – outer lip landmarks for MAR
# Vertical pairs: top(13), bottom(14) and corners: left(78), right(308)
# Using a 6-point formulation mirroring EAR for consistency
_MOUTH_OUTER = [78, 81, 82, 308, 402, 311]  # [left, top-l, top-r, right, bot-r, bot-l]

# Nose tip and chin for head-pose reference points (see increment [4])
_NOSE_TIP = 1
_CHIN = 199
_LEFT_EYE_CORNER = 263
_RIGHT_EYE_CORNER = 33
_LEFT_MOUTH = 61
_RIGHT_MOUTH = 291


class FrameFeatures(NamedTuple):
    """All computed features for a single frame."""
    ear_left: float
    ear_right: float
    ear_avg: float
    mar: float
    perclos: float       # requires a FeatureExtractor instance (rolling buffer)
    head_pitch: float    # degrees; populated in increment [4], 0.0 until then
    head_yaw: float      # degrees; populated in increment [4], 0.0 until then
    blink_rate: float    # blinks/second over rolling buffer

    def to_numpy(self) -> np.ndarray:
        """Return as a float32 array of shape (8,)."""
        return np.array(
            [self.ear_left, self.ear_right, self.ear_avg,
             self.mar, self.perclos,
             self.head_pitch, self.head_yaw,
             self.blink_rate],
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# Pure geometry helpers (no state, easily unit-testable)
# ---------------------------------------------------------------------------

def _euclidean(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """2-D Euclidean distance between two (x, y) points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _landmark_xy(landmarks, idx: int) -> tuple[float, float]:
    """Extract the (x, y) pixel pair for a landmark index."""
    lm = landmarks[idx]
    return lm.x, lm.y


def compute_ear(landmarks, eye_indices: list[int]) -> float:
    """
    Eye Aspect Ratio — Soukupová & Čech (2016).

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Parameters
    ----------
    landmarks : mediapipe NormalizedLandmarkList
    eye_indices : 6-element list [p1, p2, p3, p4, p5, p6]
        p1 = outer corner, p4 = inner corner,
        p2/p3 = upper lid, p5/p6 = lower lid

    Returns
    -------
    float : EAR value (higher = more open; ~0.25 threshold for closure)
    """
    p = [_landmark_xy(landmarks, i) for i in eye_indices]
    vertical_1 = _euclidean(p[1], p[5])
    vertical_2 = _euclidean(p[2], p[4])
    horizontal = _euclidean(p[0], p[3])
    if horizontal < 1e-6:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def compute_mar(landmarks) -> float:
    """
    Mouth Aspect Ratio — yawn proxy.

    Uses the same 6-point formulation as EAR applied to outer lip landmarks.

    MAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Returns
    -------
    float : MAR value (higher = more open mouth; >0.6 typically signals yawn)
    """
    p = [_landmark_xy(landmarks, i) for i in _MOUTH_OUTER]
    vertical_1 = _euclidean(p[1], p[5])
    vertical_2 = _euclidean(p[2], p[4])
    horizontal = _euclidean(p[0], p[3])
    if horizontal < 1e-6:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


# ---------------------------------------------------------------------------
# Stateful extractor — tracks rolling buffers for PERCLOS + blink rate
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Stateful per-session feature extractor.

    Maintains a rolling buffer of `window_size` frames to compute:
    - PERCLOS  : % frames with EAR_avg below threshold
    - blink_rate: detected blinks per second (state-machine based)

    Usage
    -----
    extractor = FeatureExtractor()
    features  = extractor.extract(face_landmarks)   # call once per frame
    vec       = features.to_numpy()                 # → shape (8,)
    """

    def __init__(
        self,
        ear_threshold: float = Config.EAR_THRESHOLD,
        window_size: int = Config.WINDOW_SIZE,
        fps: float = 30.0,
    ) -> None:
        self.ear_threshold = ear_threshold
        self.window_size = window_size
        self.fps = fps

        # Rolling buffer: stores EAR_avg for last `window_size` frames
        self._ear_buffer: deque[float] = deque(maxlen=window_size)

        # Blink detection state machine
        self._blink_count: int = 0
        self._eye_closed: bool = False
        self._blink_timestamps: deque[float] = deque(maxlen=window_size)
        self._frame_index: int = 0

    # ------------------------------------------------------------------
    def update_threshold(self, ear_threshold: float) -> None:
        """Update EAR threshold (called after adaptive calibration)."""
        self.ear_threshold = ear_threshold

    # ------------------------------------------------------------------
    def extract(self, face_landmarks) -> FrameFeatures:
        """
        Compute all features for the current frame.

        Parameters
        ----------
        face_landmarks : mediapipe NormalizedLandmarkList
            `.landmark` attribute containing 468 points.

        Returns
        -------
        FrameFeatures namedtuple
        """
        lm = face_landmarks.landmark

        # --- Eye Aspect Ratios ---
        ear_l = compute_ear(lm, _LEFT_EYE)
        ear_r = compute_ear(lm, _RIGHT_EYE)
        ear_avg = (ear_l + ear_r) / 2.0

        # --- Mouth Aspect Ratio ---
        mar = compute_mar(lm)

        # --- Update rolling buffer ---
        self._ear_buffer.append(ear_avg)
        self._frame_index += 1

        # --- PERCLOS ---
        perclos = self._compute_perclos()

        # --- Blink detection ---
        self._update_blink_state(ear_avg)
        blink_rate = self._compute_blink_rate()

        return FrameFeatures(
            ear_left=ear_l,
            ear_right=ear_r,
            ear_avg=ear_avg,
            mar=mar,
            perclos=perclos,
            head_pitch=0.0,   # filled in by HeadPoseEstimator (increment [4])
            head_yaw=0.0,
            blink_rate=blink_rate,
        )

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all buffers (call at session start or after re-calibration)."""
        self._ear_buffer.clear()
        self._blink_count = 0
        self._eye_closed = False
        self._blink_timestamps.clear()
        self._frame_index = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_perclos(self) -> float:
        """
        PERCLOS = fraction of frames in buffer where EAR_avg < threshold.
        Returns 0.0 if buffer not yet full enough to be meaningful.
        """
        if len(self._ear_buffer) < 2:
            return 0.0
        closed = sum(1 for e in self._ear_buffer if e < self.ear_threshold)
        return closed / len(self._ear_buffer)

    def _update_blink_state(self, ear_avg: float) -> None:
        """
        Simple two-state blink detector:
        closed → open transition = one blink.
        Avoids double-counting sustained closure (drowsy state).
        """
        is_closed = ear_avg < self.ear_threshold
        if is_closed and not self._eye_closed:
            self._eye_closed = True
        elif not is_closed and self._eye_closed:
            self._eye_closed = False
            self._blink_count += 1
            self._blink_timestamps.append(self._frame_index)

    def _compute_blink_rate(self) -> float:
        """
        Blinks per second computed over the rolling window.
        """
        if len(self._blink_timestamps) < 2:
            return 0.0
        # Frames spanned by the blink history currently in buffer
        span_frames = self._blink_timestamps[-1] - self._blink_timestamps[0]
        if span_frames < 1:
            return 0.0
        span_seconds = span_frames / max(self.fps, 1.0)
        return len(self._blink_timestamps) / span_seconds
