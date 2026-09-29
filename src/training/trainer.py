"""
trainer.py
==========
Training + validation loop for the ALERT FatigueClassifier.

Usage (from project root)
--------------------------
  python -m training.trainer                 # train with default config
  python -m training.trainer --val-fold 4   # hold out fold 4 as validation
  python -m training.trainer --epochs 50 --batch-size 32

Expects
-------
  cache/*.npz files produced by:  python scripts/extract_cache.py

Outputs
-------
  models/alert_model_best.pt        — best checkpoint (lowest val loss)
  logs/training_metrics.json        — per-epoch train/val metrics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from core.model import FatigueClassifier
from training.augment import AugmentedWindowDataset
from training.normalizer import FeatureNormalizer
from utils.config import Config


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Outlier clipping bounds (applied per-frame before normalization)
#
# Rationale:
#   EAR  is a ratio of eye landmark distances, physically bounded 0–~0.5.
#   Values > 0.6 are MediaPipe gimbal-lock / collapsed-landmark artifacts.
#   MAR  similarly bounded 0–~1.5; values > 2.0 are detection failures.
#   Pitch from solvePnP wraps to ±179° at near-perpendicular poses (gimbal
#   lock). Genuine fatigue head-drops are within ±60°.
#   Yaw  is already within ±90° in the real data — bounds are a safety net.
# ---------------------------------------------------------------------------
_CLIP_BOUNDS: dict[int, tuple[float, float]] = {
    0: (0.0,  0.6),    # EAR_L
    1: (0.0,  0.6),    # EAR_R
    2: (0.0,  0.6),    # EAR_avg
    3: (0.0,  0.8),    # MAR  — biological max at full yawn ~0.7; 0.8 gives headroom
    # 4: PERCLOS — already bounded [0, 1] by construction, skip
    5: (-60.0, 60.0),  # Pitch  (degrees)
    6: (-90.0, 90.0),  # Yaw    (degrees)
    7: (0.0,   4.0),   # Blink_rate — p99.9 of training data; 10/s values are artifacts
}


def _clip_window(x: np.ndarray) -> np.ndarray:
    """Clip per-feature outliers in a (T, F) window in-place."""
    for col, (lo, hi) in _CLIP_BOUNDS.items():
        np.clip(x[:, col], lo, hi, out=x[:, col])
    return x


class CachedWindowDataset(Dataset):
    """
    Loads pre-extracted .npz window files from the cache directory.

    Each .npz contains:
      windows : (M, 60, 8) float32
      label   : int scalar
    """

    def __init__(
        self,
        npz_paths: list[Path],
        normalizer: FeatureNormalizer | None = None,
    ) -> None:
        self.windows: list[np.ndarray] = []
        self.labels:  list[int]        = []

        for p in npz_paths:
            data = np.load(p)
            wins = data["windows"]          # (M, 60, 8)
            label = int(data["label"])
            self.windows.append(wins)
            self.labels.extend([label] * len(wins))

        self._windows = np.concatenate(self.windows, axis=0)   # (N, 60, 8)
        self._labels  = np.array(self.labels, dtype=np.int64)  # (N,)
        self._normalizer = normalizer

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._windows[idx].copy()      # (60, 8) float32 — copy before mutate
        x = _clip_window(x)                # 1. clip outliers (gimbal-lock etc.)
        if self._normalizer is not None:
            x = self._normalizer.transform_window(x)  # 2. z-score normalize
        y = torch.tensor(self._labels[idx], dtype=torch.long)
        return torch.from_numpy(x), y

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for WeightedRandomSampler."""
        counts = np.bincount(self._labels, minlength=Config.N_CLASSES).astype(float)
        counts = np.where(counts == 0, 1.0, counts)       # avoid div-by-zero
        weights = 1.0 / counts
        sample_weights = torch.tensor(
            weights[self._labels], dtype=torch.float32
        )
        return sample_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_cache_files(
    cache_dir: Path,
    val_fold: int,
) -> tuple[list[Path], list[Path]]:
    """Split .npz files in cache_dir by fold number encoded in filename."""
    all_files = sorted(cache_dir.glob("fold*.npz"))
    train = [p for p in all_files if not p.name.startswith(f"fold{val_fold}_")]
    val   = [p for p in all_files if p.name.startswith(f"fold{val_fold}_")]
    return train, val


def _make_loader(
    dataset: CachedWindowDataset,
    batch_size: int,
    shuffle: bool,
    balance: bool = False,
) -> DataLoader:
    sampler = None
    if balance:
        w = dataset.class_weights()
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        shuffle = False   # mutually exclusive with sampler

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=2,
        pin_memory=False,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Training + validation
# ---------------------------------------------------------------------------

def _run_epoch(
    model: FatigueClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW | None,
    device: torch.device,
) -> tuple[float, float]:
    """Run one epoch. If optimizer is None → eval mode."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(dim=1) == y).sum().item()
            total      += len(y)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    val_fold:   int   = 4,
    epochs:     int   = Config.MAX_EPOCHS,
    batch_size: int   = Config.BATCH_SIZE,
    lr:         float = Config.LR,
    patience:   int   = Config.PATIENCE,
    cache_dir:  Path  = Path(Config.CACHE_DIR),
    model_path: Path  = Path(Config.MODEL_PATH),
    log_path:   Path  = Path(Config.TRAINING_LOG),
) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # --- Data ---
    train_files, val_files = _split_cache_files(cache_dir, val_fold)
    if not train_files:
        raise RuntimeError(
            f"No training .npz files found in {cache_dir}.\n"
            "Run:  python scripts/extract_cache.py"
        )
    print(f"Train  : {len(train_files)} videos | Val: {len(val_files)} videos")

    # Fit normalizer on CLIPPED training windows so stats match what the model
    # actually sees in __getitem__ (clip then normalize). Fitting on raw data
    # would compute e.g. MAR std=0.026 from outlier-skewed data, producing
    # normalized MAR values of ~60 for legitimate clipped MAR=2.0.
    raw_train_ds = CachedWindowDataset(train_files)      # unnormalized, for clipping
    clipped_train_windows = np.stack(
        [_clip_window(raw_train_ds._windows[i].copy())
         for i in range(len(raw_train_ds))]
    )                                                    # (N, 60, 8) clipped
    print(f"Train windows : {len(raw_train_ds)} | Val windows (raw count): {len(CachedWindowDataset(val_files))}")

    normalizer = FeatureNormalizer()
    normalizer.fit(clipped_train_windows)                # fit on clipped distribution
    stats_path = model_path.parent / "feature_stats.npz"
    normalizer.save(stats_path)
    print(f"Normalizer    : fitted on clipped data, saved to {stats_path}")
    print(normalizer.summary())

    # Rebuild datasets with normalization applied inside __getitem__
    train_ds = CachedWindowDataset(train_files, normalizer=normalizer)
    val_ds   = CachedWindowDataset(val_files,   normalizer=normalizer)
    print(f"Train windows : {len(train_ds)} | Val windows: {len(val_ds)}")

    # Wrap training set with on-the-fly augmentations; val stays clean
    aug_train_ds = AugmentedWindowDataset(train_ds)
    train_loader = _make_loader(aug_train_ds, batch_size, shuffle=True, balance=True)
    val_loader   = _make_loader(val_ds,       batch_size, shuffle=False)

    # --- Model ---
    torch.manual_seed(Config.RANDOM_SEED)
    model = FatigueClassifier().to(device)
    print(f"Parameters : {model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.COSINE_T_MAX, eta_min=1e-6)

    # --- Training loop ---
    best_val_loss = float("inf")
    epochs_no_improve = 0
    metrics: list[dict] = []

    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}  {'Time':>6}")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, None, device
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.perf_counter() - t0

        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>9.4f}  "
              f"{val_loss:>8.4f}  {val_acc:>7.4f}  {current_lr:>8.2e}  "
              f"{elapsed:>5.1f}s")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "val_loss":   val_loss,
                "val_acc":    val_acc,
                "config": {
                    "n_features": Config.N_FEATURES,
                    "hidden1":    Config.HIDDEN_SIZE,
                    "hidden2":    Config.HIDDEN_SIZE_2,
                    "fc_size":    Config.FC_SIZE,
                    "n_classes":  Config.N_CLASSES,
                },
            }, model_path)
            print(f"         ↳ saved best model (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1

        metrics.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
            "lr":         current_lr,
        })
        log_path.write_text(json.dumps(metrics, indent=2))

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs).")
            break

    print(f"\nBest val loss : {best_val_loss:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Log saved to  : {log_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ALERT FatigueClassifier")
    parser.add_argument("--val-fold",   type=int,   default=4)
    parser.add_argument("--epochs",     type=int,   default=Config.MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=Config.BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=Config.LR)
    parser.add_argument("--patience",   type=int,   default=Config.PATIENCE)
    args = parser.parse_args()

    train(
        val_fold=args.val_fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )
