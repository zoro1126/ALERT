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
        self._threshold: float | None = None

    # ------------------------------------------------------------------
    def add_frame(self, ear_avg: float) -> None:
        """Add one frame's EAR_avg reading during the calibration window."""
        if not self.is_ready():
            self._samples.append(ear_avg)

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
        Compute threshold, save to disk, and return the value.

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

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ear_threshold": round(threshold, 5),
            "ear_open_mean": round(mean, 5),
            "ear_open_std":  round(std,  5),
            "n_frames":      len(self._samples),
        }
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
    def reset(self) -> None:
        """Clear samples — allows re-running calibration in the same session."""
        self._samples.clear()
        self._threshold = None
