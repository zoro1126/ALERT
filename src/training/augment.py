"""
augment.py
==========
Temporal data augmentation for (seq_len, n_features) fatigue windows.

All functions operate on numpy arrays of shape (T, F) and return the same
shape. They are designed to be fast enough for on-the-fly application
inside a DataLoader worker (no GPU, pure numpy).

Augmentations implemented
--------------------------
1. gaussian_noise     — adds N(0, std) to every value
2. time_warp          — smoothly stretches/compresses the time axis via
                        a random cumulative warp path (DTW-inspired)
3. channel_mask       — zeros out one randomly selected feature channel
4. temporal_crop      — takes a random 80–100% sub-window, linearly
                        interpolates back to the original length

Usage
-----
from training.augment import AugmentedWindowDataset
from training.trainer import CachedWindowDataset

base_ds = CachedWindowDataset(train_files)
aug_ds  = AugmentedWindowDataset(base_ds, p_noise=0.5, p_warp=0.3,
                                  p_mask=0.2, p_crop=0.3)

References
----------
- Um et al. (2017): Data Augmentation of Wearable Sensor Data for
  Parkinson's Disease Monitoring. ACM ICMI.
- Iwana & Uchida (2021): An empirical survey of data augmentation for
  time series classification. PLOS ONE.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Pure augmentation functions  (T, F) → (T, F)
# ---------------------------------------------------------------------------

def gaussian_noise(
    window: np.ndarray,
    std: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to every element.

    Parameters
    ----------
    window : (T, F) float32
    std    : noise standard deviation (relative to feature scale ~0–1)

    Returns
    -------
    (T, F) float32
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, std, size=window.shape).astype(np.float32)
    return window + noise


def time_warp(
    window: np.ndarray,
    sigma: float = 0.15,
    knots: int = 4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Smoothly warp the time axis by sampling random knot displacements and
    interpolating them into a monotonic warp path.

    Parameters
    ----------
    window : (T, F) float32
    sigma  : std of random knot displacement (fraction of T)
    knots  : number of interior warp control points

    Returns
    -------
    (T, F) float32, same length as input
    """
    if rng is None:
        rng = np.random.default_rng()

    T = window.shape[0]
    # Knot positions: 0, T/(knots+1), ..., T
    orig_steps = np.linspace(0, T - 1, num=knots + 2)
    # Random displacements for interior knots only (endpoints stay fixed)
    displace = rng.normal(0.0, sigma * T, size=knots)
    warped_steps = orig_steps.copy()
    warped_steps[1:-1] += displace
    # Enforce monotonicity by clipping and sorting
    warped_steps[1:-1] = np.clip(warped_steps[1:-1], 0, T - 1)
    warped_steps = np.sort(warped_steps)

    # Build warp map: for each output timestep, what is the source timestep?
    output_grid = np.linspace(0, T - 1, num=T)
    source_grid = np.interp(output_grid, warped_steps, orig_steps)
    source_grid = np.clip(source_grid, 0, T - 1)

    # Interpolate each feature channel
    orig_grid = np.arange(T, dtype=np.float32)
    warped = np.stack(
        [np.interp(source_grid, orig_grid, window[:, f])
         for f in range(window.shape[1])],
        axis=1,
    ).astype(np.float32)

    return warped


def channel_mask(
    window: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Zero out one randomly selected feature channel for the entire window.

    Simulates sensor dropout (e.g., face partially occluded → EAR = 0).

    Returns
    -------
    (T, F) float32 with one column zeroed
    """
    if rng is None:
        rng = np.random.default_rng()
    out = window.copy()
    col = int(rng.integers(0, window.shape[1]))
    out[:, col] = 0.0
    return out


def temporal_crop(
    window: np.ndarray,
    min_ratio: float = 0.80,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Randomly crop 80–100% of the window and linearly interpolate back to
    the original length T.

    Simulates variable-speed video clips and prevents overfitting to
    absolute temporal position.

    Parameters
    ----------
    min_ratio : float
        Minimum fraction of the original sequence to keep (default 0.80).

    Returns
    -------
    (T, F) float32
    """
    if rng is None:
        rng = np.random.default_rng()

    T = window.shape[0]
    crop_len = int(rng.uniform(min_ratio, 1.0) * T)
    crop_len = max(crop_len, 2)                       # must have at least 2 pts

    max_start = T - crop_len
    start = int(rng.integers(0, max_start + 1))
    cropped = window[start : start + crop_len]        # (crop_len, F)

    # Resize back to T via linear interpolation per channel
    old_grid = np.linspace(0, 1, num=crop_len)
    new_grid = np.linspace(0, 1, num=T)
    resized  = np.stack(
        [np.interp(new_grid, old_grid, cropped[:, f])
         for f in range(window.shape[1])],
        axis=1,
    ).astype(np.float32)

    return resized


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class AugmentedWindowDataset(Dataset):
    """
    Wraps a CachedWindowDataset and applies random augmentations
    on-the-fly during iteration (training only).

    Parameters
    ----------
    base_dataset : CachedWindowDataset
    p_noise : float — probability of applying gaussian_noise
    p_warp  : float — probability of applying time_warp
    p_mask  : float — probability of applying channel_mask
    p_crop  : float — probability of applying temporal_crop

    Augmentations are applied independently with their respective
    probabilities. Multiple may be applied to the same sample.
    """

    def __init__(
        self,
        base_dataset,
        p_noise: float = 0.5,
        p_warp:  float = 0.3,
        p_mask:  float = 0.2,
        p_crop:  float = 0.3,
    ) -> None:
        self._base  = base_dataset
        self.p_noise = p_noise
        self.p_warp  = p_warp
        self.p_mask  = p_mask
        self.p_crop  = p_crop

    def __len__(self) -> int:
        return len(self._base)

    def class_weights(self):
        """Delegate to base dataset for WeightedRandomSampler support."""
        return self._base.class_weights()

    def __getitem__(self, idx: int):
        x, y = self._base[idx]
        arr  = x.numpy()                # (60, 8) float32

        rng = np.random.default_rng()   # fresh RNG per call (thread-safe)

        if rng.random() < self.p_crop:
            arr = temporal_crop(arr, rng=rng)
        if rng.random() < self.p_warp:
            arr = time_warp(arr, rng=rng)
        if rng.random() < self.p_noise:
            arr = gaussian_noise(arr, rng=rng)
        if rng.random() < self.p_mask:
            arr = channel_mask(arr, rng=rng)

        import torch
        return torch.from_numpy(arr), y
