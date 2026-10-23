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

# EAR columns — these receive adaptive window normalization
_EAR_COLS = (0, 1, 2)

# Minimum denominator for EAR normalization to prevent division-by-zero on
# frames with fully closed / occluded eyes (e.g. glasses glare)
_EAR_MIN_DENOM = 0.15


def clip_window(window: np.ndarray) -> np.ndarray:
    """
    Clip per-feature values in a (T, F) window in-place to physical limits.
    """
    for col, (lo, hi) in CLIP_BOUNDS.items():
        if col < window.shape[1]:
            np.clip(window[:, col], lo, hi, out=window[:, col])
    return window


def adaptive_ear_normalization(window: np.ndarray) -> np.ndarray:
    """
    Normalize EAR channels by the 90th-percentile of each EAR column within
    the window, making EAR invariant to subject anatomy and camera distance.

    Rationale
    ---------
    In a 60-frame (~4s) window, genuine blinks occupy 2–5 frames (<10%).
    The 90th-percentile of EAR approximates the "wide-open eye baseline" for
    that specific subject in that specific clip. Dividing by this value:
      - Collapses all subjects to a common [0, ~1.0] open-eye scale.
      - Preserves the fatigue signal (progressive EAR decline toward 0).
      - Handles glasses, makeup, ethnicity, and individual anatomy differences.
    
    A minimum clamp of 0.15 prevents division by near-zero on frames where
    mediapipe loses tracking (e.g. side profiles, occlusion).
    """
    for col in _EAR_COLS:
        if col >= window.shape[1]:
            continue
        p90 = float(np.percentile(window[:, col], 90))
        denom = max(p90, _EAR_MIN_DENOM)
        window[:, col] = window[:, col] / denom
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
    Full preprocessing pipeline applied to a single (T, F) window before
    z-score normalization. Order matters:

      1. Outlier clipping            — remove tracker failures / gimbal lock
      2. Adaptive EAR normalization  — subject-anatomy invariance [NEW FIX]
      3. Pose median centering        — static camera-angle invariance

    This pipeline is applied identically during training AND real-time inference
    to guarantee no train/inference domain mismatch.
    """
    out = window.copy()
    out = clip_window(out)
    out = adaptive_ear_normalization(out)  # FIX: window-relative EAR scaling
    out = center_head_pose(out)
    return out
