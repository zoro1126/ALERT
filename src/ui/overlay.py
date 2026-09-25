"""
overlay.py
==========
OpenCV HUD drawing functions for the ALERT real-time dashboard.

All functions accept a BGR numpy frame (H, W, 3) and draw in-place,
returning the same array.  This avoids unnecessary copies in the hot loop.

Sections drawn
--------------
- Status bar     : alert state pill (NORMAL / WARNING / CRITICAL)
- Feature panel  : EAR, MAR, PERCLOS, blink rate, head pitch/yaw
- Progress bar   : model warm-up fill (first 60 frames)
- Prob bars      : per-class confidence bars
- Attention strip: 60-pixel heatmap of attention weights
- Landmark dots  : optional eye/mouth keypoint overlay
"""

from __future__ import annotations

import cv2
import numpy as np

from core.alert_manager import AlertState
from utils.config import Config

# ---------------------------------------------------------------------------
# Colour palette (BGR)
# ---------------------------------------------------------------------------
_C_WHITE   = (255, 255, 255)
_C_BLACK   = (0,   0,   0)
_C_GREY    = (80,  80,  80)
_C_GREEN   = (0,   220, 0)
_C_ORANGE  = (0,   165, 255)
_C_RED     = (0,   0,   255)
_C_CYAN    = (255, 220, 0)
_C_ALPHA   = 0.45   # panel transparency

_STATE_COLOR = {
    AlertState.NORMAL:   _C_GREEN,
    AlertState.WARNING:  _C_ORANGE,
    AlertState.CRITICAL: _C_RED,
}

_CLASS_COLORS = [_C_GREEN, _C_ORANGE, _C_RED]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _blend_rect(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple[int, int, int],
    alpha: float = _C_ALPHA,
) -> None:
    """Draw a semi-transparent filled rectangle."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _text(
    frame: np.ndarray,
    msg: str,
    x: int, y: int,
    color: tuple[int, int, int] = _C_WHITE,
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    cv2.putText(
        frame, msg, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Public drawing functions
# ---------------------------------------------------------------------------

def draw_status_bar(
    frame: np.ndarray,
    state: AlertState,
    label: str,
    confidence: float,
) -> None:
    """
    Draw an alert state pill at the top of the frame.

    Parameters
    ----------
    frame      : BGR (H, W, 3) ndarray
    state      : current AlertState
    label      : e.g. 'Drowsy'
    confidence : 0–1 float, model probability for predicted class
    """
    h, w = frame.shape[:2]
    color = _STATE_COLOR[state]
    _blend_rect(frame, 0, 0, w, 44, _C_BLACK, alpha=0.55)
    cv2.rectangle(frame, (0, 0), (w, 44), color, 2)

    status_text = f"[{state.value}]  {label}  ({confidence:.0%})"
    _text(frame, status_text, 12, 28, color, scale=0.75, thickness=2)


def draw_feature_panel(
    frame: np.ndarray,
    ear_l: float, ear_r: float, ear_avg: float,
    mar: float, perclos: float,
    pitch: float, yaw: float, blink_rate: float,
    ear_threshold: float = Config.EAR_THRESHOLD,
) -> None:
    """Draw a feature readout panel on the left side of the frame."""
    x0, y0 = 8, 60
    line_h  = 22

    _blend_rect(frame, x0 - 4, y0 - 16, x0 + 200, y0 + line_h * 8, _C_BLACK)

    rows = [
        (f"EAR L: {ear_l:.3f}",    _C_RED   if ear_l   < ear_threshold else _C_WHITE),
        (f"EAR R: {ear_r:.3f}",    _C_RED   if ear_r   < ear_threshold else _C_WHITE),
        (f"EAR  : {ear_avg:.3f}",  _C_RED   if ear_avg < ear_threshold else _C_CYAN),
        (f"MAR  : {mar:.3f}",      _C_ORANGE if mar > Config.MAR_THRESHOLD else _C_WHITE),
        (f"PERC : {perclos:.2f}",  _C_RED   if perclos > Config.PERCLOS_WARNING else _C_WHITE),
        (f"Pitch: {pitch:+.1f}°",  _C_ORANGE if abs(pitch) > Config.HEAD_PITCH_THRESHOLD else _C_WHITE),
        (f"Yaw  : {yaw:+.1f}°",    _C_WHITE),
        (f"Blink: {blink_rate:.1f}/s", _C_WHITE),
    ]
    for i, (msg, color) in enumerate(rows):
        _text(frame, msg, x0, y0 + i * line_h, color)


def draw_warmup_bar(frame: np.ndarray, fill: float) -> None:
    """
    Draw a warm-up progress bar while the rolling buffer fills.

    Parameters
    ----------
    fill : 0.0 – 1.0 fraction of buffer filled
    """
    if fill >= 1.0:
        return
    h, w = frame.shape[:2]
    bar_w, bar_h = 220, 14
    x0 = w // 2 - bar_w // 2
    y0 = h - 30
    _blend_rect(frame, x0 - 2, y0 - 2, x0 + bar_w + 2, y0 + bar_h + 2, _C_BLACK)
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), _C_GREY, 1)
    filled = int(fill * bar_w)
    if filled > 0:
        cv2.rectangle(frame, (x0, y0), (x0 + filled, y0 + bar_h), _C_CYAN, -1)
    _text(frame, f"Warming up {fill:.0%}", x0, y0 - 5, _C_CYAN, scale=0.45)


def draw_prob_bars(
    frame: np.ndarray,
    probs: np.ndarray,
) -> None:
    """
    Draw per-class probability bars in the bottom-right corner.

    Parameters
    ----------
    probs : (3,) float32 array [Alert, LowVigilant, Drowsy]
    """
    h, w = frame.shape[:2]
    bar_w, bar_h = 120, 14
    x0 = w - bar_w - 12
    y0 = h - 10 - (bar_h + 6) * 3

    for i, (p, name) in enumerate(zip(probs, Config.CLASS_NAMES)):
        y = y0 + i * (bar_h + 6)
        _blend_rect(frame, x0 - 70, y - 2, x0 + bar_w + 4, y + bar_h + 2, _C_BLACK)
        cv2.rectangle(frame, (x0, y), (x0 + bar_w, y + bar_h), _C_GREY, 1)
        filled = int(p * bar_w)
        if filled > 0:
            cv2.rectangle(frame, (x0, y), (x0 + filled, y + bar_h),
                          _CLASS_COLORS[i], -1)
        _text(frame, f"{name[:7]:<7} {p:.0%}", x0 - 68, y + 11, _CLASS_COLORS[i], scale=0.44)


def draw_attention_strip(
    frame: np.ndarray,
    attention: np.ndarray,
) -> None:
    """
    Draw a 60-pixel wide attention heatmap strip at the bottom of the frame.

    Parameters
    ----------
    attention : (60,) float32 attention weights
    """
    h, w = frame.shape[:2]
    strip_h = 10
    y0 = h - strip_h - 1

    # Normalise to [0, 1]
    a = attention.copy()
    a_min, a_max = a.min(), a.max()
    if a_max > a_min:
        a = (a - a_min) / (a_max - a_min)

    strip_w = w
    indices  = np.linspace(0, len(a) - 1, strip_w).astype(int)
    for x, idx in enumerate(indices):
        intensity = int(a[idx] * 255)
        color = (0, intensity, 255 - intensity)   # blue → green gradient
        cv2.line(frame, (x, y0), (x, y0 + strip_h), color, 1)


def draw_no_face(frame: np.ndarray) -> None:
    """Draw a 'No face detected' warning overlay."""
    h, w = frame.shape[:2]
    _blend_rect(frame, 0, 0, w, 44, _C_BLACK, alpha=0.7)
    _text(frame, "NO FACE DETECTED", 12, 28, _C_ORANGE, scale=0.75, thickness=2)
