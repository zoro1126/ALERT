"""
normalizer.py
=============
Per-feature z-score normalization for (N, T, F) window tensors.

Problem it solves
-----------------
Raw features fed to the BiGRU span wildly different scales:
  - EAR / PERCLOS : ~0.0 – 0.6  (unitless ratios)
  - Pitch         : -179° – +155°
  - Yaw           : -88° – +89°  (with a ~30° per-recording bias)
  - Blink rate    : 0 – 10 blinks/s

Without normalization, the GRU gradient is dominated by high-magnitude
Pitch/Yaw features that are subject-specific and non-generalizable,
causing train loss to drop while val loss rises every epoch.

Design
------
- Statistics (mean, std) are computed from the **training split only**.
  The validation split is transformed using training stats (standard
  practice — no val leakage).
- Stats are saved as `models/feature_stats.npz` so that the same
  normalization can be applied at inference time (run_alert.py).
- Std values of 0 are replaced with 1 to avoid division-by-zero on
  constant features.
- The normalizer operates on raw numpy arrays inside the Dataset so it
  runs transparently in DataLoader workers without touching PyTorch tensors.

Usage (training)
----------------
    from training.normalizer import FeatureNormalizer
    norm = FeatureNormalizer()
    norm.fit(train_windows_array)          # (N, T, F) float32
    norm.save("models/feature_stats.npz")

    # Then wrap datasets:
    class MyDataset(Dataset):
        def __getitem__(self, idx):
            x = self._windows[idx]         # (T, F)
            x = norm.transform_window(x)   # (T, F) — z-scored
            return torch.from_numpy(x), y

Usage (inference)
-----------------
    norm = FeatureNormalizer.load("models/feature_stats.npz")
    normalized = norm.transform_window(feature_vector_60x8)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class FeatureNormalizer:
    """
    Z-score normalizer for (T, F) or (N, T, F) fatigue feature arrays.

    After fitting on the training split, call `transform_window` inside
    Dataset.__getitem__ to normalize on the fly (thread-safe — read-only
    after fitting).
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None   # (F,) float64
        self.std_:  np.ndarray | None = None   # (F,) float64
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, windows: np.ndarray) -> "FeatureNormalizer":
        """
        Compute per-feature mean and std from a (N, T, F) array.

        Parameters
        ----------
        windows : (N, T, F) float32
            All training windows concatenated.

        Returns
        -------
        self  (for chaining)
        """
        if windows.ndim != 3:
            raise ValueError(
                f"Expected 3-D array (N, T, F), got shape {windows.shape}"
            )

        # Flatten N and T, compute stats per feature dimension F
        flat = windows.reshape(-1, windows.shape[-1])   # (N*T, F)
        self.mean_ = flat.mean(axis=0).astype(np.float64)
        self.std_  = flat.std(axis=0).astype(np.float64)

        # Replace zero std with 1 to avoid division-by-zero on constant features
        self.std_ = np.where(self.std_ == 0.0, 1.0, self.std_)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform_window(self, window: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization to a single (T, F) window.

        Parameters
        ----------
        window : (T, F) float32

        Returns
        -------
        (T, F) float32 — each feature column zero-meaned, unit-variance
        """
        self._check_fitted()
        mean = self.mean_.astype(np.float32)
        std  = self.std_.astype(np.float32)
        return ((window - mean) / std).astype(np.float32)

    def transform_batch(self, windows: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization to a batch of (N, T, F) windows.

        Parameters
        ----------
        windows : (N, T, F) float32

        Returns
        -------
        (N, T, F) float32
        """
        self._check_fitted()
        mean = self.mean_.astype(np.float32)
        std  = self.std_.astype(np.float32)
        return ((windows - mean) / std).astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save mean and std to a compressed .npz file.

        Parameters
        ----------
        path : str or Path
            Destination file (e.g. ``models/feature_stats.npz``).
        """
        self._check_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, mean=self.mean_, std=self.std_)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureNormalizer":
        """
        Load a previously saved normalizer from disk.

        Parameters
        ----------
        path : str or Path
            Path to a .npz file saved by :meth:`save`.

        Returns
        -------
        FeatureNormalizer  — ready to call transform_window / transform_batch
        """
        data = np.load(path)
        norm = cls()
        norm.mean_ = data["mean"].astype(np.float64)
        norm.std_  = data["std"].astype(np.float64)
        norm._fitted = True
        return norm

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "FeatureNormalizer has not been fitted yet. "
                "Call .fit(train_windows) before transforming."
            )

    def summary(self) -> str:
        """Return a human-readable table of per-feature stats."""
        self._check_fitted()
        feat_names = [
            "EAR_L", "EAR_R", "EAR_avg", "MAR",
            "PERCLOS", "Pitch", "Yaw", "Blink_rate",
        ]
        lines = [
            f"{'Feature':<12} {'Mean':>10} {'Std':>10}",
            "-" * 34,
        ]
        for i, name in enumerate(feat_names):
            lines.append(
                f"{name:<12} {self.mean_[i]:>10.4f} {self.std_[i]:>10.4f}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return f"FeatureNormalizer({status})"
