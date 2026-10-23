"""
trainer.py
==========
Production-grade training + validation engine for the ALERT fatigue detection system.

Features & Optimizations
------------------------
1. Hardware: Native PyTorch Automatic Mixed Precision (AMP) with CUDA GradScaler
   tailored for NVIDIA RTX 3050 6GB GPU Tensor Cores.
2. Data Preprocessing: Dynamic camera-angle neutralization (relative median pose)
   and outlier clipping via preprocess_window.
3. Class-Imbalance Handling: Configurable Focal Loss (gamma=2.0) with class weighting
   to resolve Class 1 (Low Vigilant) under-detection.
4. Generalization: Stride subsampling (e.g. stride=2 or 4) to eliminate 75% window
   overlap memorization, plus temporal data augmentations.
5. Metrics: Full per-epoch tracking of Loss, Accuracy, and per-class / macro F1.
6. Artifacts: Checkpoints persist model weights, config, and normalizer statistics.
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

from core.model import FatigueClassifier, build_model
from training.augment import AugmentedWindowDataset
from training.losses import get_criterion
from training.normalizer import FeatureNormalizer
from training.preprocess import preprocess_window, clip_window
from utils.config import Config

# Backward-compatibility alias for modules importing _clip_window from trainer
_clip_window = preprocess_window


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

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
        subsample_stride: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        npz_paths        : list of .npz cache files to load
        normalizer       : fitted FeatureNormalizer (applied in __getitem__)
        subsample_stride : keep every Nth window per file to reduce temporal
                           redundancy. stride=1 -> all windows;
                           stride=2 -> 50% overlap; stride=4 -> 0% overlap.
        """
        self.windows: list[np.ndarray] = []
        self.labels:  list[int]        = []

        for p in npz_paths:
            data = np.load(p)
            wins = data["windows"]              # (M, 60, 8)
            label = int(data["label"])
            wins = wins[::subsample_stride]
            self.windows.append(wins)
            self.labels.extend([label] * len(wins))

        self._windows = np.concatenate(self.windows, axis=0)   # (N', 60, 8)
        self._labels  = np.array(self.labels, dtype=np.int64)  # (N',)
        self._normalizer       = normalizer
        self._subsample_stride = subsample_stride

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._windows[idx].copy()           # (60, 8) float32
        x = preprocess_window(x)                # 1. Outlier clip + relative pose centering
        if self._normalizer is not None:
            x = self._normalizer.transform_window(x)  # 2. Z-score normalize
        y = torch.tensor(self._labels[idx], dtype=torch.long)
        return torch.from_numpy(x), y

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for WeightedRandomSampler."""
        counts = np.bincount(self._labels, minlength=Config.N_CLASSES).astype(float)
        counts = np.where(counts == 0, 1.0, counts)
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
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    balance: bool = False,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    sampler = None
    if balance:
        if hasattr(dataset, "class_weights"):
            w = dataset.class_weights()
            sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
            shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def compute_f1_scores(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> tuple[list[float], float]:
    """Compute per-class F1 and macro F1."""
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = (2 * tp + fp + fn)
        f1 = float(2 * tp / denom) if denom > 0 else 0.0
        f1s.append(round(f1, 4))
    macro_f1 = float(np.mean(f1s))
    return f1s, round(macro_f1, 4)


# ---------------------------------------------------------------------------
# Training + Validation Pass
# ---------------------------------------------------------------------------

def _run_epoch(
    model: FatigueClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW | None,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float, list[float], float]:
    """
    Run one epoch with automatic mixed precision (AMP) on CUDA.
    Returns: (avg_loss, accuracy, per_class_f1, macro_f1)
    """
    training = optimizer is not None
    model.train(training)
    use_amp = (device.type == "cuda")

    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                logits, _ = model(x)
                loss = criterion(logits, y)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += loss.item() * len(y)
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y.detach().cpu().numpy())

    y_true = np.concatenate(all_targets) if all_targets else np.array([])
    y_pred = np.concatenate(all_preds) if all_preds else np.array([])

    avg_loss = total_loss / max(len(y_true), 1)
    accuracy = float((y_true == y_pred).mean()) if len(y_true) > 0 else 0.0
    per_class_f1, macro_f1 = compute_f1_scores(y_true, y_pred, Config.N_CLASSES)

    return avg_loss, accuracy, per_class_f1, macro_f1


# ---------------------------------------------------------------------------
# Main Training Routine
# ---------------------------------------------------------------------------

def train(
    val_fold:         int   = 4,
    arch:             str   = "bigru",
    epochs:           int   = Config.MAX_EPOCHS,
    batch_size:       int   = Config.BATCH_SIZE,
    lr:               float = Config.LR,
    patience:         int   = Config.PATIENCE,
    train_stride:     int   = 2,
    loss_type:        str   = "focal",
    focal_gamma:      float = 2.0,
    class_weights:    list[float] | None = None,
    cache_dir:        Path  = Path(Config.CACHE_DIR),
    model_path:       Path  = Path(Config.MODEL_PATH),
    log_path:         Path  = Path(Config.TRAINING_LOG),
) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("ALERT — Fatigue Detection Training Engine (CUDA + AMP)")
    print("=" * 70)
    print(f"Architecture  : {arch.upper()}")
    print(f"Device        : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Mixed Prec.   : {'Enabled (torch.amp.autocast)' if device.type == 'cuda' else 'Disabled'}")
    print(f"Loss Function : {loss_type.upper()} (gamma={focal_gamma})")
    print(f"Train Stride  : {train_stride} (subsample overlap reduction)")

    # Default cost-sensitive class weights: [Alert=1.0, LowVigilant=2.0, Drowsy=1.2]
    if class_weights is None:
        class_weights = [1.0, 2.0, 1.2]
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print(f"Class Weights : Alert={class_weights[0]}, LowVig={class_weights[1]}, Drowsy={class_weights[2]}")

    # --- Data Splits ---
    train_files, val_files = _split_cache_files(cache_dir, val_fold)
    if not train_files:
        raise RuntimeError(
            f"No training .npz files found in {cache_dir}.\n"
            "Run: python scripts/extract_cache.py"
        )
    print(f"Train Videos  : {len(train_files)} | Val Videos: {len(val_files)} (Holdout: Fold {val_fold})")

    # Fit normalizer on preprocessed training windows (stride=1, full distribution)
    print("\nFitting feature normalizer on preprocessed training windows...")
    raw_train_ds = CachedWindowDataset(train_files, normalizer=None, subsample_stride=1)
    preprocessed_train_windows = np.stack(
        [preprocess_window(raw_train_ds._windows[i].copy())
         for i in range(len(raw_train_ds))]
    )

    normalizer = FeatureNormalizer()
    normalizer.fit(preprocessed_train_windows)
    stats_path = model_path.parent / "feature_stats.npz"
    normalizer.save(stats_path)
    print(f"Normalizer saved to: {stats_path}")
    print(normalizer.summary())

    # Build datasets
    train_ds = CachedWindowDataset(train_files, normalizer=normalizer, subsample_stride=train_stride)
    val_ds   = CachedWindowDataset(val_files,   normalizer=normalizer, subsample_stride=1)

    effective_overlap = max(0, (Config.WINDOW_SIZE - Config.STRIDE * train_stride) / Config.WINDOW_SIZE * 100)
    print(f"\nTrain Windows : {len(train_ds):,} (~{effective_overlap:.0f}% overlap)")
    print(f"Val Windows   : {len(val_ds):,} (dense 75% overlap evaluation)")

    aug_train_ds = AugmentedWindowDataset(train_ds)
    pin_mem = (device.type == "cuda")
    train_loader = _make_loader(aug_train_ds, batch_size, shuffle=True, balance=True, pin_memory=pin_mem)
    val_loader   = _make_loader(val_ds, batch_size, shuffle=False, pin_memory=pin_mem)

    # --- Model Setup ---
    torch.manual_seed(Config.RANDOM_SEED)
    model = build_model(arch=arch, n_features=Config.N_FEATURES, n_classes=Config.N_CLASSES).to(device)
    print(f"Model Params  : {model.count_parameters():,} (< 250k edge compliant)")

    criterion = get_criterion(
        loss_name=loss_type,
        class_weights=weights_tensor,
        gamma=focal_gamma,
        label_smoothing=0.05,
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.COSINE_T_MAX, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # --- Training Loop ---
    best_macro_f1 = 0.0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    metrics: list[dict] = []

    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"{'Epoch':>5}  {'Tr Loss':>8}  {'Tr Acc':>7}  {'Val Loss':>8}  "
        f"{'Val Acc':>7}  {'F1-LowV':>8}  {'Macro-F1':>8}  {'LR':>8}  {'Time':>5}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        train_loss, train_acc, _, _ = _run_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler
        )
        val_loss, val_acc, val_f1s, val_macro_f1 = _run_epoch(
            model, val_loader, criterion, None, device, scaler=None
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.perf_counter() - t0

        f1_low_vig = val_f1s[1] if len(val_f1s) > 1 else 0.0
        print(
            f"{epoch:>5}  {train_loss:>8.4f}  {train_acc:>7.4f}  {val_loss:>8.4f}  "
            f"{val_acc:>7.4f}  {f1_low_vig:>8.4f}  {val_macro_f1:>8.4f}  "
            f"{current_lr:>8.2e}  {elapsed:>4.1f}s",
            end="",
        )

        # Save checkpoint based on best validation Macro-F1 (or loss tie-break)
        is_best = (val_macro_f1 > best_macro_f1) or (val_macro_f1 == best_macro_f1 and val_loss < best_val_loss)
        if is_best:
            best_macro_f1 = val_macro_f1
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_macro_f1,
                "val_per_class_f1": val_f1s,
                "config": {
                    "arch": arch,
                    "n_features": Config.N_FEATURES,
                    "n_classes": Config.N_CLASSES,
                    "parameters": model.count_parameters(),
                    "hidden1": Config.HIDDEN_SIZE,
                    "hidden2": Config.HIDDEN_SIZE_2,
                    "fc_size": Config.FC_SIZE,
                },
                "normalizer_stats": normalizer.to_dict(),
                "class_weights": class_weights,
                "loss_type": loss_type,
            }, model_path)
            print("  [BEST SAVED]")
        else:
            epochs_no_improve += 1
            print()

        metrics.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "val_macro_f1": round(val_macro_f1, 4),
            "val_f1_alert": round(val_f1s[0], 4) if len(val_f1s) > 0 else 0.0,
            "val_f1_low_vigilant": round(val_f1s[1], 4) if len(val_f1s) > 1 else 0.0,
            "val_f1_drowsy": round(val_f1s[2], 4) if len(val_f1s) > 2 else 0.0,
            "lr": current_lr,
            "time_sec": round(elapsed, 2),
        })
        log_path.write_text(json.dumps(metrics, indent=2))

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch} (no macro-F1 gain for {patience} epochs).")
            break

    print("\n" + "=" * 70)
    print(f"Training Complete.")
    print(f"Best Val Macro-F1 : {best_macro_f1:.4f}")
    print(f"Best Val Loss     : {best_val_loss:.4f}")
    print(f"Model Checkpoint  : {model_path}")
    print(f"Metrics Log       : {log_path}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ALERT Fatigue Classifier (PyTorch + CUDA AMP)")
    parser.add_argument("--arch", type=str, default="bigru", choices=["bigru", "tcn", "cnn_gru"], help="Candidate edge architecture")
    parser.add_argument("--val-fold", type=int, default=4, help="Validation holdout fold (default: 4)")
    parser.add_argument("--epochs", type=int, default=Config.MAX_EPOCHS, help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=Config.LR, help="Learning rate")
    parser.add_argument("--patience", type=int, default=Config.PATIENCE, help="Early stopping patience")
    parser.add_argument("--train-stride", type=int, default=2, help="Subsample train windows (1=all, 2=50% overlap, 4=0% overlap)")
    parser.add_argument("--loss", type=str, default="focal", choices=["focal", "ce"], help="Loss function")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma parameter")
    parser.add_argument("--class-weights", type=str, default="1.0,2.0,1.2", help="Comma-separated class weights (Alert,LowVig,Drowsy)")
    parser.add_argument("--model-path", type=str, default=None, help="Custom model checkpoint save path")
    parser.add_argument("--log-path", type=str, default=None, help="Custom training log save path")
    args = parser.parse_args()

    weights = [float(w.strip()) for w in args.class_weights.split(",")] if args.class_weights else None
    m_path = Path(args.model_path) if args.model_path else Path(Config.MODEL_PATH)
    l_path = Path(args.log_path) if args.log_path else Path(Config.TRAINING_LOG)

    train(
        val_fold=args.val_fold,
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        train_stride=args.train_stride,
        loss_type=args.loss,
        focal_gamma=args.gamma,
        class_weights=weights,
        model_path=m_path,
        log_path=l_path,
    )
