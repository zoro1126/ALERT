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
- Guo et al. (2020): 6-point solvePnP head pose (nose, chin, eye corners, mouth corners)
- MediaPipe FaceMesh landmark map:
  https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
"""

from __future__ import annotations

import math
from collections import deque
from typing import NamedTuple

import cv2
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
    """All computed features for a single frame (12 kinematic channels)."""
    ear_left_rel: float
    ear_right_rel: float
    ear_avg_rel: float
    d_ear_dt: float
    mar: float
    d_mar_dt: float
    perclos_calib: float       # Calibrated PERCLOS (% frames below 0.75 * ear_base)
    head_pitch_rel: float      # Degrees relative to resting posture (down = positive)
    d_pitch_dt: float          # Head nodding velocity / jerk (degrees/second)
    head_yaw_rel: float        # Degrees relative to resting posture (right = positive)
    blink_rate: float          # Blinks/second over rolling buffer
    blink_duration: float      # Mean blink closure duration (seconds)

    # Backward compatibility properties for HUD / overlays
    @property
    def ear_left(self) -> float:
        return self.ear_left_rel

    @property
    def ear_right(self) -> float:
        return self.ear_right_rel

    @property
    def ear_avg(self) -> float:
        return self.ear_avg_rel

    @property
    def perclos(self) -> float:
        return self.perclos_calib

    @property
    def head_pitch(self) -> float:
        return self.head_pitch_rel

    @property
    def head_yaw(self) -> float:
        return self.head_yaw_rel

    def to_numpy(self) -> np.ndarray:
        """Return as a float32 array of shape (12,)."""
        return np.array(
            [
                self.ear_left_rel,
                self.ear_right_rel,
                self.ear_avg_rel,
                self.d_ear_dt,
                self.mar,
                self.d_mar_dt,
                self.perclos_calib,
                self.head_pitch_rel,
                self.d_pitch_dt,
                self.head_yaw_rel,
                self.blink_rate,
                self.blink_duration,
            ],
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
# Head pose estimation using PnP solver
# ---------------------------------------------------------------------------

# 3-D reference coordinates of 6 canonical face points in a generic model.
# Origin at nose tip; units are arbitrary (millimetre-scale works fine with
# a normalised focal-length camera matrix). Sourced from the standard
# solvePnP head-pose recipe (Guo et al., 2020).
_HEAD_POSE_3D = np.array([
    [0.0,    0.0,    0.0   ],   # Nose tip          (landmark 1)
    [0.0,   -63.6,  -12.5 ],   # Chin              (landmark 199)
    [-43.3,  32.7,  -26.0 ],   # Left eye corner   (landmark 263)
    [43.3,   32.7,  -26.0 ],   # Right eye corner  (landmark 33)
    [-28.9, -28.9,  -24.1 ],   # Left mouth corner (landmark 61)
    [28.9,  -28.9,  -24.1 ],   # Right mouth corner(landmark 291)
], dtype=np.float64)

# Corresponding MediaPipe landmark indices (must match _HEAD_POSE_3D row order)
_HEAD_POSE_LM_IDX = [1, 199, 263, 33, 61, 291]


class HeadPoseEstimator:
    """
    Estimates head pitch and yaw (degrees) from 6 MediaPipe landmarks
    using OpenCV's solvePnP (EPNP algorithm).

    Focal length is approximated from image width (reasonable for a webcam).
    A new instance can be shared across frames; call `estimate()` each frame.

    Usage
    -----
    estimator = HeadPoseEstimator()
    pitch, yaw = estimator.estimate(face_landmarks, frame_w, frame_h)
    """

    def __init__(self) -> None:
        # Cache the last camera matrix to avoid rebuilding it every frame when
        # the resolution stays the same (typical for a live webcam stream).
        self._last_wh: tuple[int, int] | None = None
        self._camera_matrix: np.ndarray | None = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # assume no lens distortion

    # ------------------------------------------------------------------
    def _build_camera_matrix(self, w: int, h: int) -> np.ndarray:
        """Approximate pinhole camera matrix from image dimensions."""
        focal = w  # f ≈ image_width is a common heuristic for ~60° FOV webcams
        cx, cy = w / 2.0, h / 2.0
        return np.array([
            [focal,  0,    cx],
            [0,      focal, cy],
            [0,      0,    1.0],
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    def estimate(
        self,
        face_landmarks,
        frame_w: int,
        frame_h: int,
    ) -> tuple[float, float]:
        """
        Estimate head pitch and yaw for the current frame.

        Parameters
        ----------
        face_landmarks : mediapipe NormalizedLandmarkList
        frame_w, frame_h : int
            Pixel dimensions of the source image (needed to convert
            normalised landmark coords to pixel coords).

        Returns
        -------
        (pitch_deg, yaw_deg) : tuple[float, float]
            pitch > 0 → head tilting downward
            yaw   > 0 → head turning right
            Returns (0.0, 0.0) if solvePnP fails.
        """
        # Rebuild camera matrix only when resolution changes
        if self._last_wh != (frame_w, frame_h):
            self._camera_matrix = self._build_camera_matrix(frame_w, frame_h)
            self._last_wh = (frame_w, frame_h)

        lm = face_landmarks.landmark

        # Convert normalised (x, y) → pixel (x, y) for the 6 reference points
        image_points = np.array(
            [[lm[i].x * frame_w, lm[i].y * frame_h] for i in _HEAD_POSE_LM_IDX],
            dtype=np.float64,
        )

        success, rvec, _ = cv2.solvePnP(
            _HEAD_POSE_3D,
            image_points,
            self._camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
        )

        if not success:
            return 0.0, 0.0

        # Convert rotation vector → rotation matrix → Euler angles
        rmat, _ = cv2.Rodrigues(rvec)
        # Decompose via RQ (same as getEulerAngles in many references)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch_deg = float(angles[0])  # X-axis rotation → pitch
        yaw_deg   = float(angles[1])  # Y-axis rotation → yaw
        return pitch_deg, yaw_deg


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
    vec       = features.to_numpy()                 # → shape (12,)
    """

    def __init__(
        self,
        ear_base: float = 0.30,
        pitch_base: float = 0.0,
        yaw_base: float = 0.0,
        ear_threshold: float | None = None,
        window_size: int = Config.WINDOW_SIZE,
        fps: float = Config.TARGET_FPS,
    ) -> None:
        self.ear_base   = max(float(ear_base), 0.15)
        self.pitch_base = float(pitch_base)
        self.yaw_base   = float(yaw_base)
        if ear_threshold is not None:
            self.ear_threshold = float(ear_threshold)
        else:
            self.ear_threshold = self.ear_base * Config.EAR_CALIB_MULTIPLIER

        self.window_size = window_size
        self.fps = fps
        self._head_pose = HeadPoseEstimator()

        # Rolling buffer: stores raw EAR_avg for PERCLOS
        self._ear_buffer: deque[float] = deque(maxlen=window_size)

        # Differential velocity state
        self._prev_ear_avg: float | None = None
        self._prev_mar: float | None = None
        self._prev_pitch: float | None = None

        # Blink detection & duration state machine
        self._blink_count: int = 0
        self._eye_closed: bool = False
        self._blink_start_frame: int | None = None
        self._blink_timestamps: deque[float] = deque(maxlen=window_size)
        self._blink_durations: deque[float] = deque(maxlen=window_size)
        self._frame_index: int = 0

    # ------------------------------------------------------------------
    def update_baseline(
        self,
        ear_base: float,
        pitch_base: float = 0.0,
        yaw_base: float = 0.0,
    ) -> None:
        """Update personal resting baseline values from calibration."""
        self.ear_base   = max(float(ear_base), 0.15)
        self.ear_threshold = self.ear_base * Config.EAR_CALIB_MULTIPLIER
        self.pitch_base = float(pitch_base)
        self.yaw_base   = float(yaw_base)

    # ------------------------------------------------------------------
    def update_threshold(self, ear_threshold: float) -> None:
        """Update EAR threshold (backward compatibility)."""
        self.ear_threshold = float(ear_threshold)
        self.ear_base = max(self.ear_threshold / max(Config.EAR_CALIB_MULTIPLIER, 0.1), 0.15)

    # ------------------------------------------------------------------
    def extract(
        self,
        face_landmarks,
        frame_w: int = 640,
        frame_h: int = 480,
    ) -> FrameFeatures:
        """
        Compute all 12 kinematic features for the current frame.
        """
        lm = face_landmarks.landmark

        # --- Raw Geometry ---
        ear_l = compute_ear(lm, _LEFT_EYE)
        ear_r = compute_ear(lm, _RIGHT_EYE)
        ear_avg = (ear_l + ear_r) / 2.0
        mar = compute_mar(lm)
        pitch, yaw = self._head_pose.estimate(face_landmarks, frame_w, frame_h)

        # --- Differential Velocities ---
        dt = 1.0 / max(self.fps, 1.0)
        d_ear = 0.0 if self._prev_ear_avg is None else (ear_avg - self._prev_ear_avg) / dt
        d_mar = 0.0 if self._prev_mar is None else (mar - self._prev_mar) / dt
        d_pitch = 0.0 if self._prev_pitch is None else (pitch - self._prev_pitch) / dt

        self._prev_ear_avg = ear_avg
        self._prev_mar = mar
        self._prev_pitch = pitch

        # --- Relative Kinematics (Zero Train-Serve Skew) ---
        denom_ear = max(self.ear_base, 0.15)
        ear_l_rel = ear_l / denom_ear
        ear_r_rel = ear_r / denom_ear
        ear_avg_rel = ear_avg / denom_ear
        pitch_rel = pitch - self.pitch_base
        yaw_rel = yaw - self.yaw_base

        # --- Rolling Buffer Update ---
        self._ear_buffer.append(ear_avg)
        self._frame_index += 1

        # --- Calibrated PERCLOS ---
        perclos = self._compute_perclos()

        # --- Blink State & Duration Tracking ---
        self._update_blink_state(ear_avg)
        blink_rate = self._compute_blink_rate()

        if len(self._blink_durations) > 0:
            blink_dur = float(np.mean(self._blink_durations))
        else:
            blink_dur = 0.15  # normal human blink duration is ~0.15s

        return FrameFeatures(
            ear_left_rel=ear_l_rel,
            ear_right_rel=ear_r_rel,
            ear_avg_rel=ear_avg_rel,
            d_ear_dt=d_ear,
            mar=mar,
            d_mar_dt=d_mar,
            perclos_calib=perclos,
            head_pitch_rel=pitch_rel,
            d_pitch_dt=d_pitch,
            head_yaw_rel=yaw_rel,
            blink_rate=blink_rate,
            blink_duration=blink_dur,
        )

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all buffers (call at session start or after re-calibration)."""
        self._ear_buffer.clear()
        self._blink_count = 0
        self._eye_closed = False
        self._blink_start_frame = None
        self._blink_timestamps.clear()
        self._blink_durations.clear()
        self._prev_ear_avg = None
        self._prev_mar = None
        self._prev_pitch = None
        self._frame_index = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_perclos(self) -> float:
        """
        PERCLOS = fraction of frames in buffer where EAR_avg < threshold.
        Returns 0.0 if buffer has fewer than 2 frames.
        """
        if len(self._ear_buffer) < 2:
            return 0.0
        closed = sum(1 for e in self._ear_buffer if e < self.ear_threshold)
        return closed / len(self._ear_buffer)

    def _update_blink_state(self, ear_avg: float) -> None:
        """
        Two-state blink detector with duration calculation.
        """
        is_closed = ear_avg < self.ear_threshold
        if is_closed and not self._eye_closed:
            self._eye_closed = True
            self._blink_start_frame = self._frame_index
        elif not is_closed and self._eye_closed:
            self._eye_closed = False
            self._blink_count += 1
            self._blink_timestamps.append(self._frame_index)
            if self._blink_start_frame is not None:
                dur_frames = self._frame_index - self._blink_start_frame
                dur_sec = dur_frames / max(self.fps, 1.0)
                # Valid physiological blink duration: 40ms to 2.5s
                if 0.04 <= dur_sec <= 2.5:
                    self._blink_durations.append(dur_sec)
                self._blink_start_frame = None

    def _compute_blink_rate(self) -> float:
        """
        Blinks per second computed over the rolling window.
        """
        if len(self._blink_timestamps) < 2:
            return 0.0
        span_frames = self._blink_timestamps[-1] - self._blink_timestamps[0]
        if span_frames < 1:
            return 0.0
        span_seconds = span_frames / max(self.fps, 1.0)
        return len(self._blink_timestamps) / span_seconds
