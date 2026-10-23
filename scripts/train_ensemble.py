"""
train_ensemble.py
=================
Phase 3: 12-Model Heterogeneous Ensemble Training Suite
(3 distinct architectures x 4 Subject-Independent Holdout Folds).

Architectures:
  1. HybridCNNGRU   (~35k params): 1D-CNN temporal feature extractor + GRU + temporal attention
  2. TCNClassifier  (~95k params): Dilated causal convolutions (RF=125 frames)
  3. BiGRUAttention (~48k params): Bidirectional GRU + multi-head temporal attention

Training Specifications (master_plan.md Phase 3):
  - Loss: Ordinal Distance-Penalized Focal Loss (lambda_ord=0.35, gamma=2.0, label_smoothing=0.08)
  - Class Weights: [Alert=1.0, LowVigilant=1.15, Drowsy=1.10]
  - Sequence Window: 120 frames (8.0s @ 15 FPS)
  - Features: 12 kinematic channels with velocity differentials
  - Normalization: Per-fold subject baseline calibration + Z-score normalizer
  - Optimizer: AdamW (lr=1e-3, weight_decay=1e-3) + CosineAnnealingLR
  - Hardware: NVIDIA CUDA with Mixed-Precision (torch.amp.autocast fp16)

Outputs
-------
  models/ensemble_{arch}_fold{k}.pt         - Trained model checkpoint
  models/ensemble_{arch}_fold{k}_stats.npz   - Associated fold Z-score normalizer
  logs/ensemble_training_summary.json       - Comprehensive cross-validation leaderboard
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from core.model import build_model
from training.losses import get_criterion
from training.normalizer import FeatureNormalizer
from training.preprocess import preprocess_window
from training.trainer import (
    CachedWindowDataset,
    _make_loader,
    _run_epoch,
    _split_cache_files,
    compute_f1_scores,
)
from utils.config import Config


# ---------------------------------------------------------------------------
# Detect available folds
# ---------------------------------------------------------------------------

def detect_available_folds(cache_dir: Path) -> list[int]:
    """Detect which fold numbers have valid 12-feature cached windows."""
    fold_counts: dict[int, int] = {}
    for p in sorted(cache_dir.glob("fold*.npz")):
        try:
            with np.load(p, mmap_mode="r") as d:
                if d["windows"].ndim == 3 and d["windows"].shape[-1] == Config.N_FEATURES:
                    # Parse fold from name e.g. fold1_01_0.npz
                    fold_num = int(p.name[4:5])
                    fold_counts[fold_num] = fold_counts.get(fold_num, 0) + 1
        except Exception:
            continue

    # Only include folds with at least 15 valid video files
    available = [f for f, count in sorted(fold_counts.items()) if count >= 15]
    return available


# ---------------------------------------------------------------------------
# Train Single Model
# ---------------------------------------------------------------------------

def train_single_model(
    arch: str,
    val_fold: int,
    cache_dir: Path,
    output_dir: Path,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    train_stride: int = 2,
    patience: int = 8,
    device: torch.device | None = None,
) -> dict:
    """Train a single model architecture on a subject-independent fold."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = f"ensemble_{arch}_fold{val_fold}"
    ckpt_path = output_dir / f"{model_name}.pt"
    stats_path = output_dir / f"{model_name}_stats.npz"

    print("-" * 65)
    print(f"Training: {arch.upper()} | Holdout: Fold {val_fold} | Device: {device.type.upper()}")
    print("-" * 65)

    train_files, val_files = _split_cache_files(cache_dir, val_fold)
    if not train_files or not val_files:
        raise RuntimeError(f"Insufficient cache files for fold {val_fold}: train={len(train_files)}, val={len(val_files)}")

    print(f"  Train Videos: {len(train_files)} | Holdout Videos: {len(val_files)}")

    # 1. Fit normalizer on training split preprocessed windows
    raw_train_ds = CachedWindowDataset(train_files, normalizer=None, subsample_stride=1)
    preprocessed_train_windows = np.stack(
        [preprocess_window(raw_train_ds._windows[i].copy())
         for i in range(len(raw_train_ds))]
    )
    normalizer = FeatureNormalizer()
    normalizer.fit(preprocessed_train_windows)
    normalizer.save(stats_path)

    # 2. Build datasets and loaders
    train_ds = CachedWindowDataset(train_files, normalizer=normalizer, subsample_stride=train_stride)
    val_ds = CachedWindowDataset(val_files, normalizer=normalizer, subsample_stride=1)

    pin_mem = (device.type == "cuda")
    train_loader = _make_loader(train_ds, batch_size, shuffle=True, balance=True, pin_memory=pin_mem)
    val_loader = _make_loader(val_ds, batch_size, shuffle=False, balance=False, pin_memory=pin_mem)

    print(f"  Train Windows: {len(train_ds):,} | Val Windows: {len(val_ds):,}")

    # 3. Model setup
    torch.manual_seed(Config.RANDOM_SEED + val_fold)
    model = build_model(arch=arch, n_features=Config.N_FEATURES, n_classes=Config.N_CLASSES).to(device)

    # 4. Ordinal Distance-Penalized Loss
    weights_tensor = torch.tensor([1.0, 1.15, 1.10], dtype=torch.float32, device=device)
    criterion = get_criterion(
        loss_name="ordinal",
        class_weights=weights_tensor,
        gamma=2.0,
        label_smoothing=0.08,
        lambda_ord=0.35,
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_macro_f1 = 0.0
    best_val_loss = float("inf")
    best_acc = 0.0
    best_f1s: list[float] = [0.0, 0.0, 0.0]
    best_epoch = 0
    epochs_no_improve = 0

    header = f"{'Epoch':>5}  {'Tr Loss':>7}  {'Tr Acc':>7}  {'Val Loss':>8}  {'Val Acc':>7}  {'F1-Alert':>8}  {'F1-LowV':>8}  {'F1-Drowsy':>9}  {'Macro-F1':>8}  {'Time':>5}"
    print(header)
    print("." * len(header))

    t_train_start = time.perf_counter()

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()

        tr_loss, tr_acc, _, _ = _run_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
        val_loss, val_acc, val_f1s, val_macro_f1 = _run_epoch(model, val_loader, criterion, None, device, scaler=None)

        scheduler.step()
        ep_time = time.perf_counter() - t0

        f1_a = val_f1s[0] if len(val_f1s) > 0 else 0.0
        f1_l = val_f1s[1] if len(val_f1s) > 1 else 0.0
        f1_d = val_f1s[2] if len(val_f1s) > 2 else 0.0

        is_best = (val_macro_f1 > best_macro_f1) or (val_macro_f1 == best_macro_f1 and val_loss < best_val_loss)
        star = " *" if is_best else ""

        print(
            f"{ep:>5}  {tr_loss:>7.4f}  {tr_acc:>7.4f}  {val_loss:>8.4f}  "
            f"{val_acc:>7.4f}  {f1_a:>8.4f}  {f1_l:>8.4f}  {f1_d:>9.4f}  "
            f"{val_macro_f1:>8.4f}  {ep_time:>4.1f}s{star}"
        )

        if is_best:
            best_macro_f1 = val_macro_f1
            best_val_loss = val_loss
            best_acc = val_acc
            best_f1s = val_f1s
            best_epoch = ep
            epochs_no_improve = 0

            # Save checkpoint
            torch.save(
                {
                    "epoch": ep,
                    "arch": arch,
                    "val_fold": val_fold,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "macro_f1": best_macro_f1,
                    "accuracy": best_acc,
                    "per_class_f1": best_f1s,
                    "normalizer_mean": normalizer.mean_.tolist(),
                    "normalizer_std": normalizer.std_.tolist(),
                    "config": {
                        "n_features": Config.N_FEATURES,
                        "window_size": Config.WINDOW_SIZE,
                        "n_classes": Config.N_CLASSES,
                    },
                },
                ckpt_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping triggered after epoch {ep} (no improvement for {patience} epochs).")
                break

    t_total = time.perf_counter() - t_train_start
    print(f"  Fold {val_fold} Best Macro-F1: {best_macro_f1:.4f} (Epoch {best_epoch}) in {t_total:.1f}s\n")

    return {
        "arch": arch,
        "fold": val_fold,
        "best_epoch": best_epoch,
        "macro_f1": round(float(best_macro_f1), 4),
        "accuracy": round(float(best_acc), 4),
        "f1_alert": round(float(best_f1s[0]), 4) if len(best_f1s) > 0 else 0.0,
        "f1_low_vig": round(float(best_f1s[1]), 4) if len(best_f1s) > 1 else 0.0,
        "f1_drowsy": round(float(best_f1s[2]), 4) if len(best_f1s) > 2 else 0.0,
        "best_val_loss": round(float(best_val_loss), 4),
        "ckpt_path": str(ckpt_path),
        "stats_path": str(stats_path),
        "train_time_sec": round(t_total, 1),
    }


# ---------------------------------------------------------------------------
# Main Training Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3: Train 12-Model Heterogeneous Ensemble across holdout folds."
    )
    parser.add_argument("--epochs", type=int, default=25,
                        help="Maximum epochs per model (default: 25).")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64).")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3).")
    parser.add_argument("--patience", type=int, default=8,
                        help="Early stopping patience (default: 8).")
    parser.add_argument("--train-stride", type=int, default=2,
                        help="Window subsample stride during training (default: 2).")
    parser.add_argument("--archs", type=str, default="cnn_gru,tcn,bigru",
                        help="Comma-separated architectures to train (default: 'cnn_gru,tcn,bigru').")
    parser.add_argument("--folds", type=str, default="auto",
                        help="Comma-separated fold numbers or 'auto' to detect completed folds.")
    args = parser.parse_args()

    cache_dir = Path(Config.CACHE_DIR)
    models_dir = Path(Config.MODEL_PATH).parent
    logs_dir = Path(Config.TRAINING_LOG).parent
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ALERT - Phase 3: Heterogeneous Ensemble Suite Training")
    print("=" * 70)

    # Determine folds
    if args.folds.lower() == "auto":
        folds = detect_available_folds(cache_dir)
        print(f"Auto-detected available holdout folds: {folds}")
    else:
        folds = [int(f.strip()) for f in args.folds.split(",")]

    arch_list = [a.strip() for a in args.archs.split(",")]

    print(f"Architectures : {arch_list}")
    print(f"Folds         : {folds} ({len(arch_list) * len(folds)} total models)")
    print(f"Input Shape   : ({Config.WINDOW_SIZE}, {Config.N_FEATURES})")
    print(f"Loss Function : Ordinal Distance-Penalized Focal Loss (lambda=0.35)")
    print(f"Epochs/Patience: {args.epochs} / {args.patience}")
    print()

    results: list[dict] = []
    t_start = time.perf_counter()

    for arch in arch_list:
        for f in folds:
            res = train_single_model(
                arch=arch,
                val_fold=f,
                cache_dir=cache_dir,
                output_dir=models_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                train_stride=args.train_stride,
                patience=args.patience,
            )
            results.append(res)

    t_total = time.perf_counter() - t_start

    # Save summary
    summary_path = logs_dir / "ensemble_training_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))

    # Print final Leaderboard Table
    print("=" * 78)
    print(f"{'ARCHITECTURE':<14} {'FOLD':<6} {'MACRO-F1':<10} {'ACCURACY':<10} {'F1-ALERT':<10} {'F1-LOWV':<10} {'F1-DROW':<10}")
    print("-" * 78)

    macro_f1s = []
    for r in results:
        macro_f1s.append(r["macro_f1"])
        print(
            f"{r['arch'].upper():<14} {r['fold']:<6} {r['macro_f1']:<10.4f} "
            f"{r['accuracy']:<10.4f} {r['f1_alert']:<10.4f} {r['f1_low_vig']:<10.4f} {r['f1_drowsy']:<10.4f}"
        )

    print("-" * 78)
    avg_macro = np.mean(macro_f1s) if macro_f1s else 0.0
    print(f"Average Macro-F1 across all {len(results)} models: {avg_macro:.4f}")
    print(f"Suite training completed in {t_total / 60:.1f} minutes ({t_total:.1f}s).")
    print(f"Leaderboard saved to: {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
