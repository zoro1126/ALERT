"""
calibrator.py
=============
Adaptive EAR / MAR threshold calibration.

During a short (~10 second) open-eye calibration session at session start,
FeatureExtractor records the user's natural resting EAR.  The calibrated
threshold is stored to disk and reloaded on subsequent sessions.

Algorithm
---------
1. Collect `n_frames` of open-eye EAR values via a live MediaPipe feed.
2. Compute mean and std of the collected EAR values.
3. threshold = mean * EAR_CALIB_MULTIPLIER   (default multiplier = 0.75)
   → sits 25% below the user's own open-eye mean, well above closed-eye (~0.15)
4. Persist to JSON at CALIBRATION_PATH.

Usage
-----
cal = Calibrator()
cal.add_frame(ear_avg)           # call once per frame during calibration
if cal.is_ready():
    threshold = cal.finalize()   # writes JSON, returns float
    extractor.update_threshold(threshold)
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from utils.config import Config


class Calibrator:
    """
    Per-user EAR threshold calibrator.

    Parameters
    ----------
    n_frames : int
        Number of frames to collect before computing the threshold.
        At 15 FPS, 150 frames ≈ 10 seconds.
    multiplier : float
        Fraction of open-eye mean used as the closed-eye threshold.
    save_path : Path
        Where to persist calibration data.
    """

    def __init__(
        self,
        n_frames:   int   = Config.TARGET_FPS * Config.CALIBRATION_SECONDS,
        multiplier: float = Config.EAR_CALIB_MULTIPLIER,
        save_path:  Path  = Path(Config.CALIBRATION_PATH),
    ) -> None:
        self.n_frames   = n_frames
        self.multiplier = multiplier
        self.save_path  = save_path
        self._samples:  list[float] = []
        self._pitch_samples: list[float] = []
        self._yaw_samples:   list[float] = []
        self._threshold: float | None = None
        self._profile: dict | None = None

    # ------------------------------------------------------------------
    def add_frame(
        self,
        ear_avg: float,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ) -> None:
        """Add one frame's EAR_avg and head pose readings during calibration."""
        if not self.is_ready():
            self._samples.append(float(ear_avg))
            self._pitch_samples.append(float(pitch))
            self._yaw_samples.append(float(yaw))

    # ------------------------------------------------------------------
    def is_ready(self) -> bool:
        """True when enough frames have been collected to compute threshold."""
        return len(self._samples) >= self.n_frames

    # ------------------------------------------------------------------
    def progress(self) -> float:
        """Fraction of calibration frames collected (0.0 – 1.0)."""
        return min(len(self._samples) / self.n_frames, 1.0)

    # ------------------------------------------------------------------
    def finalize(self) -> float:
        """
        Compute threshold and baselines, save to disk, and return the threshold.

        Returns
        -------
        float : calibrated EAR threshold
        """
        if not self._samples:
            return Config.EAR_THRESHOLD

        mean = statistics.mean(self._samples)
        std  = statistics.stdev(self._samples) if len(self._samples) > 1 else 0.0
        threshold = max(mean * self.multiplier, Config.EAR_THRESHOLD * 0.5)
        self._threshold = threshold

        pitch_base = statistics.median(self._pitch_samples) if self._pitch_samples else 0.0
        yaw_base = statistics.median(self._yaw_samples) if self._yaw_samples else 0.0

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ear_threshold": round(threshold, 5),
            "ear_open_mean": round(mean, 5),
            "ear_open_std":  round(std,  5),
            "pitch_base":    round(pitch_base, 3),
            "yaw_base":      round(yaw_base, 3),
            "n_frames":      len(self._samples),
        }
        self._profile = payload
        self.save_path.write_text(json.dumps(payload, indent=2))
        return threshold

    # ------------------------------------------------------------------
    @staticmethod
    def load(save_path: Path = Path(Config.CALIBRATION_PATH)) -> float | None:
        """
        Load a previously saved threshold.

        Returns
        -------
        float : saved EAR threshold, or None if file not found / invalid.
        """
        try:
            data = json.loads(save_path.read_text())
            return float(data["ear_threshold"])
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
            return None

    # ------------------------------------------------------------------
    @staticmethod
    def load_profile(save_path: Path = Path(Config.CALIBRATION_PATH)) -> dict | None:
        """
        Load full calibration profile (EAR and head pose baselines).

        Returns
        -------
        dict with keys: ear_threshold, ear_base, pitch_base, yaw_base, or None.
        """
        try:
            data = json.loads(save_path.read_text())
            return {
                "ear_threshold": float(data.get("ear_threshold", Config.EAR_THRESHOLD)),
                "ear_base": float(data.get("ear_open_mean", 0.30)),
                "pitch_base": float(data.get("pitch_base", 0.0)),
                "yaw_base": float(data.get("yaw_base", 0.0)),
            }
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
            return None

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear samples — allows re-running calibration in the same session."""
        self._samples.clear()
        self._pitch_samples.clear()
        self._yaw_samples.clear()
        self._threshold = None
        self._profile = None
