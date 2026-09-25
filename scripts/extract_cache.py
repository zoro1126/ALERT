"""
extract_cache.py
================
Batch feature extraction script for the UTA-RLDD dataset.

For every VideoRecord found by scan_dataset():
  1. Extract per-frame 8-feature vectors via extract_features()
  2. Slice into overlapping windows: shape (M, WINDOW_SIZE, N_FEATURES)
  3. Save one .npz file per video into cache/

Cache file layout
-----------------
cache/<fold>_<subject>_<label>.npz
  windows  : float32 (M, 60, 8)  — M overlapping windows from this video
  label    : int                  — 0=Alert, 1=LowVigilant, 2=Drowsy
  fold     : int
  subject  : str (stored as bytes in npz; decode on load)
  n_frames : int                  — total sampled frames before windowing

Usage
-----
  # From project root:
  python scripts/extract_cache.py

  # Dry-run (print what would be done, no disk writes):
  python scripts/extract_cache.py --dry-run

  # Force re-extraction (overwrite existing cache files):
  python scripts/extract_cache.py --force

  # Only process a specific fold:
  python scripts/extract_cache.py --fold 1
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running from project root without installing the package
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from training.dataset import scan_dataset, dataset_summary, VideoRecord
from training.frame_extractor import extract_features
from utils.config import Config


# ---------------------------------------------------------------------------
# Sliding window helper
# ---------------------------------------------------------------------------

def sliding_windows(
    features: np.ndarray,
    window_size: int = Config.WINDOW_SIZE,
    stride: int = Config.STRIDE,
) -> np.ndarray:
    """
    Slice a (N, 8) feature matrix into overlapping windows.

    Parameters
    ----------
    features : np.ndarray, shape (N, 8)
    window_size : int  — frames per window (default 60)
    stride : int       — step between window starts (default 15)

    Returns
    -------
    np.ndarray of shape (M, window_size, 8), float32
        Returns empty (0, window_size, 8) if N < window_size.
    """
    n_frames = features.shape[0]
    if n_frames < window_size:
        return np.empty((0, window_size, features.shape[1]), dtype=np.float32)

    starts = range(0, n_frames - window_size + 1, stride)
    windows = np.stack(
        [features[s : s + window_size] for s in starts],
        axis=0,
    )
    return windows.astype(np.float32)


# ---------------------------------------------------------------------------
# Cache path for a single VideoRecord
# ---------------------------------------------------------------------------

def cache_path_for(record: VideoRecord, cache_dir: Path) -> Path:
    """Return the .npz path for a given VideoRecord."""
    fname = f"fold{record.fold}_{record.subject}_{record.label}.npz"
    return cache_dir / fname


# ---------------------------------------------------------------------------
# Process one video
# ---------------------------------------------------------------------------

def process_video(
    record: VideoRecord,
    cache_dir: Path,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Extract features + windows for one video, write cache file.

    Returns a status dict with keys: path, status, n_frames, n_windows, elapsed.
    """
    out_path = cache_path_for(record, cache_dir)
    result = {
        "path": str(record.path),
        "out": str(out_path),
        "status": "ok",
        "n_frames": 0,
        "n_windows": 0,
        "elapsed": 0.0,
    }

    if out_path.exists() and not force:
        result["status"] = "skipped"
        return result

    if dry_run:
        result["status"] = "dry-run"
        return result

    t0 = time.perf_counter()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = extract_features(record.path)

    if features.shape[0] == 0:
        result["status"] = "no-face"
        result["elapsed"] = time.perf_counter() - t0
        return result

    windows = sliding_windows(features)

    if windows.shape[0] == 0:
        result["status"] = "too-short"
        result["elapsed"] = time.perf_counter() - t0
        return result

    np.savez_compressed(
        out_path,
        windows=windows,
        label=np.int32(record.label),
        fold=np.int32(record.fold),
        subject=np.bytes_(record.subject),
        n_frames=np.int32(features.shape[0]),
    )

    result["n_frames"]  = features.shape[0]
    result["n_windows"] = windows.shape[0]
    result["elapsed"]   = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and cache windowed features for all UTA-RLDD videos."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing files.")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if cache file already exists.")
    parser.add_argument("--fold", type=int, default=None,
                        help="Only process videos from this fold number.")
    args = parser.parse_args()

    cache_dir = Path(Config.CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ALERT — UTA-RLDD Feature Cache Builder")
    print("=" * 60)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        records = scan_dataset()

    if args.fold is not None:
        records = [r for r in records if r.fold == args.fold]
        print(f"Filtered to fold {args.fold}: {len(records)} videos")

    print(dataset_summary(records))
    print(f"\nCache dir  : {cache_dir}")
    print(f"Window size: {Config.WINDOW_SIZE} frames  |  Stride: {Config.STRIDE} frames")
    print(f"Target FPS : {Config.TARGET_FPS}")
    if args.dry_run:
        print("** DRY-RUN MODE — no files will be written **")
    print()

    total_windows = 0
    failed = []

    for i, rec in enumerate(records, 1):
        prefix = f"[{i:3d}/{len(records)}] fold{rec.fold} sub={rec.subject} {rec.label_name:<12}"
        res = process_video(rec, cache_dir, force=args.force, dry_run=args.dry_run)

        if res["status"] == "ok":
            print(f"{prefix} → {res['n_windows']:4d} windows  ({res['elapsed']:.1f}s)")
            total_windows += res["n_windows"]
        elif res["status"] == "skipped":
            print(f"{prefix} → [cached]")
        elif res["status"] == "dry-run":
            print(f"{prefix} → [dry-run]")
        else:
            print(f"{prefix} → WARN: {res['status']}")
            failed.append(rec)

    print()
    print("=" * 60)
    print(f"Done. Total windows written: {total_windows}")
    if failed:
        print(f"Failed/skipped ({len(failed)} videos):")
        for r in failed:
            print(f"  fold{r.fold} sub={r.subject} {r.label_name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
