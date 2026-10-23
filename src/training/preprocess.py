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

3. Adaptive EAR Normalization (Subject Anatomy Invariance):
   Eye Aspect Ratio differs vastly by facial geometry (narrow vs wide eyes).
   A person with naturally narrow eyes at rest (EAR=0.22) looks "drowsy" to a
   model trained on wide-eyed subjects (EAR=0.35 at rest).
   
   FIX: Divide EAR channels by the 90th percentile of EAR within each window.
   The 90th-pctile represents the "maximum open eye capacity" during that 4s
   clip (blinks are short ~150ms, so most frames show open eyes). Dividing by
   this value makes EAR a subject-invariant RELATIVE eye closure ratio:
     EAR_rel = EAR / max(p90(EAR_in_window), 0.15)
   
   This is mathematically equivalent to per-subject baseline calibration but
   computable entirely within a window (no separate alert video needed).
"""

from __future__ import annotations

import numpy as np

# Physical and biological bounds per 12-channel feature column:
# 0: ear_left_rel, 1: ear_right_rel, 2: ear_avg_rel, 3: d_ear_dt
# 4: mar, 5: d_mar_dt, 6: perclos_calib
# 7: head_pitch_rel, 8: d_pitch_dt, 9: head_yaw_rel
# 10: blink_rate, 11: blink_duration
CLIP_BOUNDS: dict[int, tuple[float, float]] = {
    0:  (0.0,    2.5),    # ear_left_rel (relative to personal baseline)
    1:  (0.0,    2.5),    # ear_right_rel
    2:  (0.0,    2.5),    # ear_avg_rel
    3:  (-10.0, 10.0),    # d_ear_dt (eye velocity: 1/s)
    4:  (0.0,    1.2),    # mar (biological max at full yawn ~0.8-1.0)
    5:  (-10.0, 10.0),    # d_mar_dt (mouth opening velocity: 1/s)
    6:  (0.0,    1.0),    # perclos_calib [0, 1]
    7:  (-60.0, 60.0),    # head_pitch_rel (degrees relative to resting baseline)
    8:  (-180.0, 180.0),  # d_pitch_dt (head nodding jerk: deg/s)
    9:  (-60.0, 60.0),    # head_yaw_rel (degrees relative to resting baseline)
    10: (0.0,    4.0),    # blink_rate (blinks/s)
    11: (0.0,    3.0),    # blink_duration (mean duration in seconds)
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
    Zero-center residual head Pitch (col 7) and Yaw (col 9) relative to window median.
    Removes residual mounting offsets across recordings.
    """
    if window.shape[1] > 7:
        median_pitch = float(np.median(window[:, 7]))
        window[:, 7] -= median_pitch

    if window.shape[1] > 9:
        median_yaw = float(np.median(window[:, 9]))
        window[:, 9] -= median_yaw

    return window


def preprocess_window(window: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline applied to a single (T, 12) window before
    z-score normalization.

    Order:
      1. Outlier clipping      — physical biological limits
      2. Residual pose centering — eliminates residual camera mount offsets

    NOTE: EAR is calibrated against the true personal baseline in FeatureExtractor,
    completely eliminating the window-p90 blindness bug.
    """
    out = window.copy()
    out = clip_window(out)
    out = center_head_pose(out)
    return out

