"""
preprocess.py
=============
Window-level feature preprocessing and pose calibration for ALERT.

Key Functions
-------------
1. Outlier Clipping:
   Suppresses MediaPipe landmark tracking failures, gimbal lock in solvePnP,
   and numerical artifacts before feature normalization.

2. Relative Pose Calibration (Camera Bias Neutralization):
   Removes static camera-angle offsets (which act as subject fingerprints in
   UTA-RLDD) by subtracting the window's median Pitch and Yaw. This transforms
   absolute spatial orientation into relative dynamic movement:
     - Pitch_rel: isolates downward head nodding and nodding jerks
     - Yaw_rel  : isolates horizontal drifting and shoulder-look distractions
"""

from __future__ import annotations

import numpy as np

# Physical and biological bounds per feature column
# 0: EAR_L, 1: EAR_R, 2: EAR_avg, 3: MAR, 4: PERCLOS, 5: Pitch, 6: Yaw, 7: Blink_rate
CLIP_BOUNDS: dict[int, tuple[float, float]] = {
    0: (0.0,   0.6),    # EAR_L
    1: (0.0,   0.6),    # EAR_R
    2: (0.0,   0.6),    # EAR_avg
    3: (0.0,   0.8),    # MAR  — biological max at full yawn ~0.7
    # 4: PERCLOS is bounded [0, 1] by definition
    5: (-60.0, 60.0),   # Head Pitch (degrees) — genuine nodding is within ±45°
    6: (-90.0, 90.0),   # Head Yaw (degrees)   — genuine turns are within ±60°
    7: (0.0,   4.0),    # Blink rate (blinks/s) — normal ~0.2-1.0; 4.0 is ceiling
}


def clip_window(window: np.ndarray) -> np.ndarray:
    """
    Clip per-feature values in a (T, F) window in-place to physical limits.
    """
    for col, (lo, hi) in CLIP_BOUNDS.items():
        if col < window.shape[1]:
            np.clip(window[:, col], lo, hi, out=window[:, col])
    return window


def center_head_pose(window: np.ndarray) -> np.ndarray:
    """
    Zero-center head Pitch (col 5) and Yaw (col 6) relative to window median.

    Static camera mounting angles create an artificial baseline offset:
        observed_angle(t) = camera_offset + head_angle(t)

    Subtracting the median removes the fixed camera offset C:
        observed(t) - median(observed) ≈ head_angle(t) - median(head_angle)

    This eliminates camera-angle memorization across subjects while preserving
    nodding dynamics and head slumps.
    """
    if window.shape[1] > 5:
        # Col 5: Pitch
        median_pitch = float(np.median(window[:, 5]))
        window[:, 5] -= median_pitch

    if window.shape[1] > 6:
        # Col 6: Yaw
        median_yaw = float(np.median(window[:, 6]))
        window[:, 6] -= median_yaw

    return window


def preprocess_window(window: np.ndarray) -> np.ndarray:
    """
    Full pipeline applied to a single (T, F) window before normalization:
      1. Outlier clipping
      2. Static camera-angle neutralization via median centering
    """
    out = window.copy()
    out = clip_window(out)
    out = center_head_pose(out)
    return out
