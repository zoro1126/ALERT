"""
extract_cache.py
================
Batch feature extraction script for the UTA-RLDD dataset with
Zero-Leakage Subject Baseline Calibration (Phase 2).

Two-Pass Extraction:
  Pass 1: For each subject, read their Alert (0.mov / 0.mp4) video to compute
          resting personal baselines (EAR_base, Pitch_base, Yaw_base).
          Saves baselines to cache/baselines.json for instant reuse.
  Pass 2: Extract per-frame 12-channel kinematic features calibrated
          against the subject's baseline, slice into 120-frame windows (stride 15),
          and save cache/fold{fold}_{subject}_{label}.npz.

Cache file layout
-----------------
cache/fold<fold>_<subject>_<label>.npz
  windows    : float32 (M, 120, 12) — M overlapping calibrated windows
  label      : int32                — 0=Alert, 1=LowVigilant, 2=Drowsy
  fold       : int32                — 1, 2, 3, or 4
  subject    : bytes                — e.g. b"01"
  n_frames   : int32                — total sampled frames before windowing
  ear_base   : float32              — subject's resting open-eye EAR
  pitch_base : float32              — subject's resting head pitch
  yaw_base   : float32              — subject's resting head yaw

Usage
-----
  # Full extraction with 6 parallel workers:
  python scripts/extract_cache.py --workers 6

  # Force re-extraction (overwrite existing cache):
  python scripts/extract_cache.py --force --workers 6

  # Only process fold 1:
  python scripts/extract_cache.py --fold 1 --workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from training.dataset import VideoRecord, dataset_summary, scan_dataset
from training.frame_extractor import extract_features, extract_subject_baseline
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
    Slice a (N, F) feature matrix into overlapping windows.

    Parameters
    ----------
    features : np.ndarray, shape (N, F)
    window_size : int — frames per window (default 120)
    stride : int — step between window starts (default 15)

    Returns
    -------
    np.ndarray of shape (M, window_size, F), float32
    """
    n_frames = features.shape[0]
    n_feat = features.shape[1] if features.ndim > 1 else Config.N_FEATURES
    if n_frames < window_size:
        return np.empty((0, window_size, n_feat), dtype=np.float32)

    starts = range(0, n_frames - window_size + 1, stride)
    windows = np.stack(
        [features[s : s + window_size] for s in starts],
        axis=0,
    )
    return windows.astype(np.float32)


# ---------------------------------------------------------------------------
# Cache path helper
# ---------------------------------------------------------------------------

def cache_path_for(record: VideoRecord, cache_dir: Path) -> Path:
    """Return the .npz path for a given VideoRecord."""
    fname = f"fold{record.fold}_{record.subject}_{record.label}.npz"
    return cache_dir / fname


# ---------------------------------------------------------------------------
# Worker functions for multiprocessing
# ---------------------------------------------------------------------------

def _baseline_worker(task: tuple[str, str]) -> tuple[str, float, float, float]:
    """Extract baseline for one subject from their Alert video."""
    subject, alert_path_str = task
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ear_b, pitch_b, yaw_b = extract_subject_baseline(alert_path_str)
    return subject, ear_b, pitch_b, yaw_b


def _video_worker(
    task: tuple[VideoRecord, str, float, float, float, bool, bool]
) -> dict:
    """Extract features and windows for a single video."""
    record, cache_dir_str, ear_b, pitch_b, yaw_b, force, dry_run = task
    cache_dir = Path(cache_dir_str)
    out_path = cache_path_for(record, cache_dir)

    result = {
        "fold": record.fold,
        "subject": record.subject,
        "label": record.label_name,
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
        features = extract_features(
            record.path,
            ear_base=ear_b,
            pitch_base=pitch_b,
            yaw_base=yaw_b,
        )

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
        ear_base=np.float32(ear_b),
        pitch_base=np.float32(pitch_b),
        yaw_base=np.float32(yaw_b),
    )

    result["n_frames"] = features.shape[0]
    result["n_windows"] = windows.shape[0]
    result["elapsed"] = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: Zero-Leakage Dataset Re-Extraction (Calibrated 12-channel windows)."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing files.")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if cache file already exists.")
    parser.add_argument("--fold", type=int, default=None,
                        help="Only process videos from this fold number.")
    parser.add_argument("--subject", type=str, default=None,
                        help="Only process videos for a specific subject (e.g. '01').")
    parser.add_argument("--workers", type=int, default=6,
                        help="Number of parallel worker processes (default: 6).")
    args = parser.parse_args()

    cache_dir = Path(Config.CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    baselines_path = cache_dir / "baselines.json"

    print("=" * 70)
    print("ALERT - Phase 2: Calibrated 12-Channel Feature Extraction")
    print("=" * 70)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        all_records = scan_dataset()

    records = all_records
    if args.fold is not None:
        records = [r for r in records if r.fold == args.fold]
        print(f"Filtered to fold {args.fold}: {len(records)} videos")
    if args.subject is not None:
        records = [r for r in records if r.subject == args.subject]
        print(f"Filtered to subject {args.subject}: {len(records)} videos")

    print(dataset_summary(records))
    print(f"\nCache dir  : {cache_dir}")
    print(f"Window size: {Config.WINDOW_SIZE} frames ({Config.WINDOW_SIZE / Config.TARGET_FPS:.1f}s) | Stride: {Config.STRIDE} frames")
    print(f"Features   : {Config.N_FEATURES} kinematic channels")
    print(f"Workers    : {args.workers} CPU processes")
    if args.dry_run:
        print("** DRY-RUN MODE - no files will be written **")
    print()

    # -----------------------------------------------------------------------
    # Pass 1: Extract / Load Subject Resting Baselines
    # -----------------------------------------------------------------------
    print("[Pass 1/2] Subject Resting Baseline Calibration...")
    subject_baselines: dict[str, tuple[float, float, float]] = {}

    if baselines_path.exists() and not args.force:
        try:
            cached_b = json.loads(baselines_path.read_text())
            subject_baselines = {
                k: (v["ear_base"], v["pitch_base"], v["yaw_base"])
                for k, v in cached_b.items()
            }
            print(f"  Loaded {len(subject_baselines)} pre-calibrated subject baselines from {baselines_path.name}")
        except Exception as e:
            print(f"  Warning: failed to load {baselines_path.name} ({e}), recomputing...")

    # Group all dataset records by subject to locate Alert videos
    subject_map: dict[str, list[VideoRecord]] = defaultdict(list)
    for r in all_records:
        subject_map[r.subject].append(r)

    missing_subjects = [s for s in subject_map if s not in subject_baselines]
    if missing_subjects:
        print(f"  Computing baselines for {len(missing_subjects)} subjects...")
        baseline_tasks = []
        for s in missing_subjects:
            alert_recs = [r for r in subject_map[s] if r.label == 0]
            if alert_recs:
                baseline_tasks.append((s, str(alert_recs[0].path)))
            else:
                subject_baselines[s] = (0.30, 0.0, 0.0)

        if baseline_tasks and not args.dry_run:
            workers = min(args.workers, len(baseline_tasks))
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                for s, ear_b, pitch_b, yaw_b in executor.map(_baseline_worker, baseline_tasks):
                    subject_baselines[s] = (ear_b, pitch_b, yaw_b)
                    print(f"    Subject {s}: EAR_base={ear_b:.3f}, Pitch_base={pitch_b:.1f} deg, Yaw_base={yaw_b:.1f} deg")

            # Persist baselines to disk
            payload = {
                s: {
                    "ear_base": round(subject_baselines[s][0], 5),
                    "pitch_base": round(subject_baselines[s][1], 2),
                    "yaw_base": round(subject_baselines[s][2], 2),
                }
                for s in sorted(subject_baselines.keys())
            }
            baselines_path.write_text(json.dumps(payload, indent=2))
            print(f"  Saved all {len(subject_baselines)} subject baselines to {baselines_path}")
    print()

    # -----------------------------------------------------------------------
    # Pass 2: Extract & Cache Calibrated 12-Channel Windows
    # -----------------------------------------------------------------------
    print(f"[Pass 2/2] Extracting {len(records)} videos across {args.workers} workers...")
    t_start = time.perf_counter()

    video_tasks = []
    for r in records:
        ear_b, pitch_b, yaw_b = subject_baselines.get(r.subject, (0.30, 0.0, 0.0))
        video_tasks.append((r, str(cache_dir), ear_b, pitch_b, yaw_b, args.force, args.dry_run))

    total_windows = 0
    failed = []
    completed_count = 0

    if args.workers <= 1:
        # Sequential execution
        for task in video_tasks:
            res = _video_worker(task)
            completed_count += 1
            prefix = f"[{completed_count:3d}/{len(video_tasks)}] fold{res['fold']} sub={res['subject']} {res['label']:<12}"
            if res["status"] == "ok":
                print(f"{prefix} -> {res['n_windows']:4d} windows ({res['elapsed']:.1f}s)")
                total_windows += res["n_windows"]
            elif res["status"] == "skipped":
                print(f"{prefix} -> [cached]")
            elif res["status"] == "dry-run":
                print(f"{prefix} -> [dry-run]")
            else:
                print(f"{prefix} -> WARN: {res['status']}")
                failed.append(res)
    else:
        # Parallel execution across processes
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            for res in executor.map(_video_worker, video_tasks):
                completed_count += 1
                prefix = f"[{completed_count:3d}/{len(video_tasks)}] fold{res['fold']} sub={res['subject']} {res['label']:<12}"
                if res["status"] == "ok":
                    print(f"{prefix} -> {res['n_windows']:4d} windows ({res['elapsed']:.1f}s)")
                    total_windows += res["n_windows"]
                elif res["status"] == "skipped":
                    print(f"{prefix} -> [cached]")
                elif res["status"] == "dry-run":
                    print(f"{prefix} -> [dry-run]")
                else:
                    print(f"{prefix} -> WARN: {res['status']}")
                    failed.append(res)

    t_total = time.perf_counter() - t_start
    print()
    print("=" * 70)
    print(f"Phase 2 Complete in {t_total / 60:.1f} min ({t_total:.1f}s).")
    print(f"Total calibrated windows written: {total_windows}")
    if failed:
        print(f"Failed/skipped ({len(failed)} videos):")
        for f_item in failed:
            print(f"  fold{f_item['fold']} sub={f_item['subject']} {f_item['label']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
