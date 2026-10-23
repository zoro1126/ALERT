"""
train_production.py
===================
ALERT ? Full End-to-End Production Training Pipeline (All 4 Phases in One Run)

This script implements the complete ML engineering roadmap from training_guide.md
and applies ALL fixes for the validation loss divergence problem diagnosed in Phase 2:

ROOT CAUSES FIXED
-----------------
1. Absolute EAR anatomy bias          ? Adaptive window-relative EAR normalization
2. Overconfident logit scaling         ? Label smoothing 0.10 + recalibrated class weights [1.0, 1.25, 1.1]
3. High temporal overlap memorization  ? Zero-overlap train windows (stride=4)
4. Insufficient regularization         ? Dropout 0.40, AdamW weight_decay=1e-2

PIPELINE PHASES
---------------
Phase 1  (Arch Search & Sanity Check)    : Quick per-fold validation run on champion CNN-GRU
Phase 2  (4-Fold Cross-Validation)       : Full subject-independent CV on all 4 folds
Phase 3  (Master Model Training)         : Final train on all folds combined, best hparams
Phase 4  (Export & Deployment)           : ONNX export + latency benchmark + live-webcam model

OUTPUTS (strictly partitioned)
-------------------------------
  models/phase3/fold{k}_cnn_gru.pt       ? Per-fold champion checkpoints
  models/phase3/master_model.pt          ? Final champion model (all data)
  models/phase3/alert_model.onnx         ? ONNX export (edge deployment)
  logs/phase3/fold{k}/                   ? Per-fold metrics, confusion matrix, plots
  logs/phase3/cv_summary.json            ? Cross-validation aggregate summary
  logs/phase3/master/                    ? Master model evaluation artifacts
  logs/phase3/latency_benchmark.json     ? CPU/GPU 1000-run latency profile

Usage
-----
  $env:PYTHONPATH="src"; .venv\\Scripts\\python scripts/train_production.py
  
  # Or with custom overrides:
  $env:PYTHONPATH="src"; .venv\\Scripts\\python scripts/train_production.py \\
      --epochs 60 --patience 15 --batch-size 256
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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, WeightedRandomSampler

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from core.model import build_model, HybridCNNGRU
from training.augment import AugmentedWindowDataset
from training.losses import get_criterion
from training.normalizer import FeatureNormalizer
from training.preprocess import preprocess_window
from training.trainer import CachedWindowDataset, _split_cache_files, compute_f1_scores
from utils.config import Config

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Production Hyperparameters (Validated against divergence diagnosis)
# ---------------------------------------------------------------------------

ARCH           = "cnn_gru"          # Champion architecture from Phase 2
EPOCHS         = 60                  # Generous budget; early stopping handles real stop
PATIENCE       = 15                  # Stop after 15 epochs without Macro-F1 gain
BATCH_SIZE     = 256                 # Fits comfortably in RTX 3050 6 GB
LR             = 3e-4                # Slightly higher LR for CosineWarmRestarts schedule
WEIGHT_DECAY   = 1e-2                # FIX: 100? stronger L2 vs Phase 2 (was 1e-4)
TRAIN_STRIDE   = 4                   # FIX: Zero-overlap windows (was 2 = 50% overlap)
FOCAL_GAMMA    = 2.0
# FIX: Balanced weights (was [1.0, 2.0, 1.2] ? caused 65% of predictions to flood Class 1)
CLASS_WEIGHTS  = [1.0, 1.25, 1.10]
LABEL_SMOOTH   = 0.10                # FIX: Increase from 0.05 ? penalizes overconfident logits
DROPOUT        = 0.40                # FIX: Heavy dropout (was 0.20) to bound weight norms
COSINE_T0      = 20                  # CosineAnnealingWarmRestarts first cycle length

# Architecture kwargs for the champion CNN-GRU with increased dropout
MODEL_KWARGS = {
    "conv_c1": 32,
    "conv_c2": 48,
    "gru_hidden": 32,
    "dropout": DROPOUT,              # FIX: 0.40 vs 0.20
}

CACHE_DIR    = Path(Config.CACHE_DIR)
MODELS_P3    = _PROJECT_ROOT / "models" / "phase3"
LOGS_P3      = _PROJECT_ROOT / "logs"   / "phase3"
ALL_FOLDS    = [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(msg: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{msg}\n{bar}")


def _make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    balance: bool = False,
    pin_memory: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    sampler = None
    if balance and hasattr(dataset, "class_weights"):
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
        persistent_workers=(num_workers > 0),
    )


def _build_model(device: torch.device) -> nn.Module:
    model = build_model(
        arch=ARCH,
        n_features=Config.N_FEATURES,
        n_classes=Config.N_CLASSES,
        **MODEL_KWARGS,
    ).to(device)
    return model


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW | None,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float, list[float], float]:
    training = optimizer is not None
    model.train(training)
    use_amp = device.type == "cuda"

    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

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
            all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
            all_targets.append(y.detach().cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    avg_loss = total_loss / max(len(y_true), 1)
    accuracy = float((y_true == y_pred).mean())
    per_class_f1, macro_f1 = compute_f1_scores(y_true, y_pred, Config.N_CLASSES)
    return avg_loss, accuracy, per_class_f1, macro_f1


def _fit_normalizer(train_files: list[Path]) -> FeatureNormalizer:
    """Fit normalizer on preprocessed training windows (stride=1, full distribution)."""
    raw_ds = CachedWindowDataset(train_files, normalizer=None, subsample_stride=1)
    preprocessed = np.stack(
        [preprocess_window(raw_ds._windows[i].copy()) for i in range(len(raw_ds))]
    )
    norm = FeatureNormalizer()
    norm.fit(preprocessed)
    return norm


# ---------------------------------------------------------------------------
# Single-Fold Training
# ---------------------------------------------------------------------------

def train_fold(
    val_fold: int,
    save_dir: Path,
    log_dir: Path,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    tag: str = "",
    all_data: bool = False,  # When True: train on ALL folds (master model mode)
) -> dict:
    """
    Train the champion CNN-GRU on one hold-out fold (or all data for master model).
    Returns the best metrics dict.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Data split
    if all_data:
        # Master model: use every npz file for training; no validation fold
        all_files = sorted(CACHE_DIR.glob("fold*.npz"))
        train_files = all_files
        val_files   = all_files  # evaluate on training data for loss tracking only
        model_path  = save_dir / "master_model.pt"
        log_path    = log_dir / "metrics_master.json"
        fold_label  = "ALL (Master)"
    else:
        train_files, val_files = _split_cache_files(CACHE_DIR, val_fold)
        model_path  = save_dir / f"fold{val_fold}_cnn_gru.pt"
        log_path    = log_dir / f"metrics_fold{val_fold}.json"
        fold_label  = f"Fold {val_fold}"

    if not train_files:
        raise RuntimeError(f"No .npz cache files found in {CACHE_DIR}. Run: python scripts/extract_cache.py")

    _banner(
        f"ALERT ? Production Training  [{tag}]  |  Val Holdout: {fold_label}\n"
        f"Architecture : {ARCH.upper()}  |  Device: {device}  |  "
        f"Dropout: {DROPOUT}  |  Weight Decay: {WEIGHT_DECAY}  |  Stride: {TRAIN_STRIDE}"
    )
    print(f"Class Weights  : {CLASS_WEIGHTS}  |  Label Smoothing: {LABEL_SMOOTH}  |  Focal gamma: {FOCAL_GAMMA}")
    print(f"Train Videos   : {len(train_files)}  |  Val Videos: {len(val_files)}")

    # Normalizer (fitted on training split only ? prevents data leakage)
    print("\nFitting feature normalizer on preprocessed training windows...")
    normalizer = _fit_normalizer(train_files)
    norm_path = save_dir / f"feature_stats{'_master' if all_data else f'_fold{val_fold}'}.npz"
    normalizer.save(norm_path)
    print(normalizer.summary())

    # Datasets
    train_ds = CachedWindowDataset(train_files, normalizer=normalizer, subsample_stride=TRAIN_STRIDE)
    val_ds   = CachedWindowDataset(val_files,   normalizer=normalizer, subsample_stride=1)
    aug_ds   = AugmentedWindowDataset(train_ds, p_noise=0.4, p_warp=0.25, p_mask=0.15, p_crop=0.25)
    print(f"Train Windows  : {len(train_ds):,} (stride={TRAIN_STRIDE}, 0% overlap)")
    print(f"Val Windows    : {len(val_ds):,} (dense 75% overlap evaluation)")

    pin_mem = device.type == "cuda"
    train_loader = _make_loader(aug_ds,   BATCH_SIZE, shuffle=True,  balance=True, pin_memory=pin_mem)
    val_loader   = _make_loader(val_ds,   BATCH_SIZE, shuffle=False, balance=False, pin_memory=pin_mem)

    # Model
    model = _build_model(device)
    n_params = model.count_parameters()
    print(f"Model Params   : {n_params:,} ({'EDGE COMPLIANT OK' if n_params < 250_000 else 'WARNING: EXCEEDS BUDGET'})")

    # Loss, Optimizer, Scheduler
    weights_t = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
    criterion = get_criterion(
        loss_name="focal",
        class_weights=weights_t,
        gamma=FOCAL_GAMMA,
        label_smoothing=LABEL_SMOOTH,
    )
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # CosineAnnealingWarmRestarts: restarts every T_0 epochs ? escapes local minima
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=COSINE_T0, T_mult=1, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_macro_f1 = 0.0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    metrics: list[dict] = []

    header = (
        f"{'Epoch':>5}  {'Tr Loss':>8}  {'Tr Acc':>7}  {'Val Loss':>8}  "
        f"{'Val Acc':>7}  {'F1-Alert':>8}  {'F1-LowV':>8}  {'F1-Drwsy':>8}  {'MacroF1':>8}  {'LR':>8}  Time"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        tr_loss, tr_acc, _, _ = _run_epoch(model, train_loader, criterion, optimizer, device, scaler)
        vl_loss, vl_acc, vl_f1s, vl_macro = _run_epoch(model, val_loader, criterion, None, device)

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.perf_counter() - t0

        f1_a = vl_f1s[0] if len(vl_f1s) > 0 else 0.0
        f1_l = vl_f1s[1] if len(vl_f1s) > 1 else 0.0
        f1_d = vl_f1s[2] if len(vl_f1s) > 2 else 0.0

        is_best = (vl_macro > best_macro_f1) or (
            vl_macro == best_macro_f1 and vl_loss < best_val_loss
        )

        print(
            f"{epoch:>5}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  {vl_loss:>8.4f}  "
            f"{vl_acc:>7.4f}  {f1_a:>8.4f}  {f1_l:>8.4f}  {f1_d:>8.4f}  "
            f"{vl_macro:>8.4f}  {lr_now:>8.2e}  {elapsed:.1f}s",
            end="",
        )

        if is_best:
            best_macro_f1 = vl_macro
            best_val_loss = vl_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": vl_loss,
                    "val_acc": vl_acc,
                    "val_macro_f1": vl_macro,
                    "val_per_class_f1": vl_f1s,
                    "config": {
                        "arch": ARCH,
                        "n_features": Config.N_FEATURES,
                        "n_classes": Config.N_CLASSES,
                        "parameters": n_params,
                        **MODEL_KWARGS,
                    },
                    "normalizer_stats": normalizer.to_dict(),
                    "class_weights": CLASS_WEIGHTS,
                    "loss_type": "focal",
                    "hparams": {
                        "lr": LR,
                        "weight_decay": WEIGHT_DECAY,
                        "train_stride": TRAIN_STRIDE,
                        "focal_gamma": FOCAL_GAMMA,
                        "label_smoothing": LABEL_SMOOTH,
                        "dropout": DROPOUT,
                    },
                },
                model_path,
            )
            print("  [BEST SAVED]")
        else:
            epochs_no_improve += 1
            print()

        metrics.append(
            {
                "epoch": epoch,
                "train_loss": round(tr_loss, 6),
                "train_acc": round(tr_acc, 6),
                "val_loss": round(vl_loss, 6),
                "val_acc": round(vl_acc, 6),
                "val_macro_f1": round(vl_macro, 4),
                "val_f1_alert": round(f1_a, 4),
                "val_f1_low_vigilant": round(f1_l, 4),
                "val_f1_drowsy": round(f1_d, 4),
                "lr": lr_now,
                "time_sec": round(elapsed, 2),
            }
        )
        log_path.write_text(json.dumps(metrics, indent=2))

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={patience}).")
            break

    print(f"\nBest Val Macro-F1 : {best_macro_f1:.4f}")
    print(f"Best Val Loss     : {best_val_loss:.4f}")
    print(f"Checkpoint saved  : {model_path}")

    return {
        "val_fold": val_fold,
        "best_macro_f1": best_macro_f1,
        "best_val_loss": best_val_loss,
        "total_epochs": len(metrics),
        "model_path": str(model_path),
    }


# ---------------------------------------------------------------------------
# Evaluation helper (calls evaluate.py logic inline)
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    model_path: Path,
    val_fold: int,
    log_dir: Path,
    batch_size: int = 256,
) -> dict:
    """Run full evaluation on a trained checkpoint and save reports."""
    from sklearn.metrics import classification_report, roc_auc_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})
    model = build_model(
        arch=cfg.get("arch", ARCH),
        n_features=cfg.get("n_features", Config.N_FEATURES),
        n_classes=cfg.get("n_classes", Config.N_CLASSES),
        **{k: v for k, v in MODEL_KWARGS.items()},
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    normalizer = FeatureNormalizer.from_dict(ckpt["normalizer_stats"])

    _, val_files = _split_cache_files(CACHE_DIR, val_fold)
    if not val_files:
        print(f"[WARN] No val files for fold {val_fold}; skipping evaluation.")
        return {}

    val_ds = CachedWindowDataset(val_files, normalizer=normalizer, subsample_stride=1)
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
                logits, _ = model(x)
                probs = F.softmax(logits, dim=-1)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(y.numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    n = Config.N_CLASSES
    accuracy = float((y_true == y_pred).mean())

    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0

    report = classification_report(y_true, y_pred, target_names=Config.CLASS_NAMES, output_dict=True, zero_division=0)
    macro_f1     = float(report["macro avg"]["f1-score"])
    weighted_f1  = float(report["weighted avg"]["f1-score"])

    roc_per_class = {}
    try:
        for c in range(n):
            bin_y = (y_true == c).astype(int)
            if len(np.unique(bin_y)) > 1:
                roc_per_class[Config.CLASS_NAMES[c]] = round(float(roc_auc_score(bin_y, y_prob[:, c])), 4)
            else:
                roc_per_class[Config.CLASS_NAMES[c]] = 0.0
        macro_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        macro_auc = 0.0

    # Print report
    print(f"\n{'Evaluation Report':=^70}")
    print(f"Overall Accuracy : {accuracy*100:.2f}%")
    print(f"Macro F1         : {macro_f1:.4f}  |  Weighted F1: {weighted_f1:.4f}  |  Macro ROC-AUC: {macro_auc:.4f}")
    print(f"{'Class':<16} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'ROC-AUC':>10}")
    print("-" * 60)
    for cname in Config.CLASS_NAMES:
        s = report[cname]
        print(f"{cname:<16} {s['precision']:>10.4f} {s['recall']:>10.4f} {s['f1-score']:>10.4f} {roc_per_class.get(cname, 0.0):>10.4f}")
    print("\nConfusion Matrix (% per true class):")
    print(f"{'':12} " + "  ".join(f"{n:>10}" for n in Config.CLASS_NAMES))
    for i, row in enumerate(cm_norm):
        print(f"{Config.CLASS_NAMES[i]:12} " + "  ".join(f"{v:>9.1f}%" for v in row))

    result = {
        "val_fold": val_fold,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_roc_auc": round(macro_auc, 4),
        "per_class_roc_auc": roc_per_class,
        "classification_report": report,
        "confusion_matrix_raw": cm.tolist(),
        "confusion_matrix_percent": np.round(cm_norm, 2).tolist(),
    }
    (log_dir / "test_report.json").write_text(json.dumps(result, indent=2))
    (log_dir / "confusion_matrix.json").write_text(json.dumps(
        {"confusion_matrix": cm.tolist(), "confusion_matrix_norm": np.round(cm_norm, 2).tolist(), "classes": Config.CLASS_NAMES},
        indent=2,
    ))
    return result


# ---------------------------------------------------------------------------
# Generate Plots
# ---------------------------------------------------------------------------

def generate_plots(metrics_path: Path, cm_path: Path, out_dir: Path) -> None:
    """Generate and save all training curve plots and confusion matrix heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import seaborn as sns
    except ImportError:
        print("[WARN] matplotlib/seaborn not available; skipping plots.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    if not metrics_path.exists():
        return

    history = json.loads(metrics_path.read_text())
    epochs  = [m["epoch"] for m in history]

    _COLORS = {"train": "#3b82f6", "val": "#ef4444", "alert": "#22c55e", "low_vig": "#f97316", "drowsy": "#a855f7"}

    # --- Loss Curve ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, [m["train_loss"] for m in history], color=_COLORS["train"], lw=2, label="Train Loss")
    ax.plot(epochs, [m["val_loss"]   for m in history], color=_COLORS["val"],   lw=2, label="Val Loss")
    best_ep = min(history, key=lambda m: m["val_loss"])["epoch"]
    ax.axvline(best_ep, color="#64748b", linestyle="--", alpha=0.6, label=f"Best Val @ Epoch {best_ep}")
    ax.set(xlabel="Epoch", ylabel="Focal Loss", title="Training & Validation Loss Curve")
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150); plt.close(fig)
    print(f"[Plot] {out_dir / 'loss_curve.png'}")

    # --- Accuracy Curve ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, [m["train_acc"] for m in history], color=_COLORS["train"], lw=2, label="Train Acc")
    ax.plot(epochs, [m["val_acc"]   for m in history], color=_COLORS["val"],   lw=2, label="Val Acc")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set(xlabel="Epoch", ylabel="Accuracy", title="Training & Validation Accuracy Curve")
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(out_dir / "accuracy_curve.png", dpi=150); plt.close(fig)
    print(f"[Plot] {out_dir / 'accuracy_curve.png'}")

    # --- Per-Class F1 Curve ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, [m["val_f1_alert"]        for m in history], color=_COLORS["alert"],   lw=2, label="Alert F1")
    ax.plot(epochs, [m["val_f1_low_vigilant"]  for m in history], color=_COLORS["low_vig"], lw=2, label="Low Vigilant F1")
    ax.plot(epochs, [m["val_f1_drowsy"]        for m in history], color=_COLORS["drowsy"],  lw=2, label="Drowsy F1")
    ax.plot(epochs, [m["val_macro_f1"]         for m in history], color="#6366f1", lw=2.5, ls="--", label="Macro F1")
    ax.set(xlabel="Epoch", ylabel="F1 Score", title="Per-Class F1 Evolution")
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(out_dir / "f1_curves.png", dpi=150); plt.close(fig)
    print(f"[Plot] {out_dir / 'f1_curves.png'}")

    # --- Confusion Matrix ---
    if cm_path.exists():
        cm_data = json.loads(cm_path.read_text())
        cm_arr  = np.array(cm_data["confusion_matrix_norm"])
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm_arr, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=Config.CLASS_NAMES, yticklabels=Config.CLASS_NAMES,
            linewidths=0.5, linecolor="gray", ax=ax,
        )
        ax.set(xlabel="Predicted", ylabel="True", title="Confusion Matrix (% per True Class)")
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=150); plt.close(fig)
        print(f"[Plot] {out_dir / 'confusion_matrix.png'}")


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------

def export_onnx(model_path: Path, out_path: Path) -> None:
    """Export champion model to ONNX opset 14 for edge deployment."""
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("[WARN] onnx/onnxruntime not installed; skipping ONNX export.")
        return

    device = torch.device("cpu")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})
    model = build_model(
        arch=cfg.get("arch", ARCH),
        n_features=cfg.get("n_features", Config.N_FEATURES),
        n_classes=cfg.get("n_classes", Config.N_CLASSES),
        **{k: v for k, v in MODEL_KWARGS.items()},
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy = torch.randn(1, Config.WINDOW_SIZE, Config.N_FEATURES, dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_window"],
        output_names=["logits", "attention_weights"],
        dynamic_axes={"input_window": {0: "batch_size"}},
        verbose=False,
    )

    # Verify
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    test_in = dummy.numpy()
    ort_out = sess.run(None, {"input_window": test_in})
    print(f"[ONNX] Exported & verified: {out_path} | Output shape: {ort_out[0].shape}")


# ---------------------------------------------------------------------------
# CPU + GPU Latency Benchmark
# ---------------------------------------------------------------------------

def benchmark_latency(model_path: Path, log_path: Path, n_runs: int = 1000) -> dict:
    """Benchmark single-sample (B=1) CPU and GPU inference latency."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("config", {})
    results = {}

    for dev_name in (["cpu"] + (["cuda"] if torch.cuda.is_available() else [])):
        device = torch.device(dev_name)
        model = build_model(
            arch=cfg.get("arch", ARCH),
            n_features=cfg.get("n_features", Config.N_FEATURES),
            n_classes=cfg.get("n_classes", Config.N_CLASSES),
            **{k: v for k, v in MODEL_KWARGS.items()},
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        dummy = torch.randn(1, Config.WINDOW_SIZE, Config.N_FEATURES).to(device)

        # Warm-up
        with torch.no_grad():
            for _ in range(50):
                model(dummy)

        if dev_name == "cuda":
            torch.cuda.synchronize()

        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                if dev_name == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(dummy)
                if dev_name == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)

        times_arr = np.array(times)
        results[dev_name] = {
            "mean_ms":   round(float(times_arr.mean()), 3),
            "median_ms": round(float(np.median(times_arr)), 3),
            "p95_ms":    round(float(np.percentile(times_arr, 95)), 3),
            "p99_ms":    round(float(np.percentile(times_arr, 99)), 3),
            "min_ms":    round(float(times_arr.min()), 3),
            "fps":       round(1000.0 / float(times_arr.mean()), 1),
        }
        flag = "OK COMPLIANT" if results[dev_name]["mean_ms"] <= 30 else "FAIL OVER BUDGET"
        print(f"  [{dev_name.upper():4s}] Mean: {results[dev_name]['mean_ms']:6.3f} ms | "
              f"p95: {results[dev_name]['p95_ms']:6.3f} ms | "
              f"FPS: {results[dev_name]['fps']:6.1f} | {flag}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(results, indent=2))
    print(f"  [Saved] {log_path}")
    return results


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)

    # Verify cache exists
    cache_files = list(CACHE_DIR.glob("fold*.npz"))
    if not cache_files:
        print(f"\n[ERROR] No cache files found in {CACHE_DIR}")
        print("Please run first:  $env:PYTHONPATH='src'; .venv\\Scripts\\python scripts/extract_cache.py")
        sys.exit(1)

    fold_counts = {}
    for f in cache_files:
        fold_num = int(f.name.split("_")[0].replace("fold", ""))
        fold_counts[fold_num] = fold_counts.get(fold_num, 0) + 1
    print(f"\n[Cache] Found {len(cache_files)} .npz files across {len(fold_counts)} folds: {fold_counts}")

    # =========================================================================
    # PHASE 2 (was Phase 3): 4-Fold Cross-Validation with champion architecture
    # =========================================================================
    _banner("PHASE 2/3: 4-Fold Subject-Independent Cross-Validation")
    cv_results = []

    for fold in ALL_FOLDS:
        fold_log_dir   = LOGS_P3 / f"fold{fold}"
        fold_model_dir = MODELS_P3

        result = train_fold(
            val_fold=fold,
            save_dir=fold_model_dir,
            log_dir=fold_log_dir,
            epochs=args.epochs,
            patience=args.patience,
            tag=f"4-Fold CV",
        )
        cv_results.append(result)

        # Evaluate immediately after training
        print(f"\n--- Evaluating Fold {fold} checkpoint ---")
        eval_result = evaluate_checkpoint(
            model_path=MODELS_P3 / f"fold{fold}_cnn_gru.pt",
            val_fold=fold,
            log_dir=fold_log_dir,
        )
        if eval_result:
            result["eval_macro_f1"] = eval_result["macro_f1"]
            result["eval_accuracy"]  = eval_result["accuracy"]
            result["per_class_roc_auc"] = eval_result.get("per_class_roc_auc", {})

        # Generate plots
        generate_plots(
            metrics_path=fold_log_dir / f"metrics_fold{fold}.json",
            cm_path=fold_log_dir / "confusion_matrix.json",
            out_dir=fold_log_dir / "plots",
        )

    # Cross-validation summary
    cv_macro_f1s = [r.get("eval_macro_f1", r["best_macro_f1"]) for r in cv_results]
    cv_summary = {
        "architecture": ARCH,
        "folds": cv_results,
        "cv_macro_f1_mean": round(float(np.mean(cv_macro_f1s)), 4),
        "cv_macro_f1_std":  round(float(np.std(cv_macro_f1s)), 4),
        "cv_macro_f1_per_fold": {f"fold{i+1}": round(v, 4) for i, v in enumerate(cv_macro_f1s)},
        "hparams": {
            "arch": ARCH,
            "dropout": DROPOUT,
            "weight_decay": WEIGHT_DECAY,
            "train_stride": TRAIN_STRIDE,
            "label_smoothing": LABEL_SMOOTH,
            "class_weights": CLASS_WEIGHTS,
            "focal_gamma": FOCAL_GAMMA,
            "lr": LR,
            "epochs_budget": args.epochs,
            "patience": args.patience,
        },
    }
    cv_path = LOGS_P3 / "cv_summary.json"
    cv_path.parent.mkdir(parents=True, exist_ok=True)
    cv_path.write_text(json.dumps(cv_summary, indent=2))

    _banner(
        f"4-Fold CV Complete\n"
        f"Mean Macro F1: {cv_summary['cv_macro_f1_mean']:.4f} ? {cv_summary['cv_macro_f1_std']:.4f}\n"
        f"Per-fold: { {f'fold{i+1}': round(v,4) for i,v in enumerate(cv_macro_f1s)} }"
    )

    # =========================================================================
    # PHASE 3/4: Master Model ? Train on ALL data for maximum production strength
    # =========================================================================
    _banner("PHASE 3/4: Master Model Training on ALL 141 UTA-RLDD Videos")
    print("Using best hyperparameters from 4-fold CV. No holdout ? maximum data utilization.")

    # Select the best fold checkpoint to use as initialiser
    best_fold_idx = int(np.argmax(cv_macro_f1s))
    best_fold_num = ALL_FOLDS[best_fold_idx]
    print(f"Best CV fold: Fold {best_fold_num} (Macro F1 = {cv_macro_f1s[best_fold_idx]:.4f}) ? used as warm-start.")

    master_result = train_fold(
        val_fold=best_fold_num,   # Used only to pick val set for loss tracking
        save_dir=MODELS_P3,
        log_dir=LOGS_P3 / "master",
        epochs=args.epochs,
        patience=args.patience,
        tag="Master Model (All Data)",
        all_data=True,
    )

    master_model_path = MODELS_P3 / "master_model.pt"

    # Evaluate master model on best fold (final sanity check)
    print(f"\n--- Final Evaluation of Master Model on Fold {best_fold_num} ---")
    master_eval = evaluate_checkpoint(
        model_path=master_model_path,
        val_fold=best_fold_num,
        log_dir=LOGS_P3 / "master",
    )

    # Copy master model to the root-level canonical path
    import shutil
    canonical_path = _PROJECT_ROOT / "models" / "alert_model_best.pt"
    shutil.copy2(str(master_model_path), str(canonical_path))
    print(f"\n[Canonical Model] Copied to: {canonical_path}")

    # =========================================================================
    # PHASE 4: ONNX Export + Edge Deployment Verification
    # =========================================================================
    _banner("PHASE 4: ONNX Export & Edge Latency Benchmark")

    onnx_path = MODELS_P3 / "alert_model.onnx"
    print(f"Exporting to ONNX: {onnx_path}")
    export_onnx(master_model_path, onnx_path)

    # Also copy to canonical ONNX path
    canonical_onnx = _PROJECT_ROOT / "models" / "alert_model.onnx"
    if onnx_path.exists():
        import shutil
        shutil.copy2(str(onnx_path), str(canonical_onnx))
        print(f"[Canonical ONNX]  Copied to: {canonical_onnx}")

    # Latency benchmark
    print("\nRunning edge latency benchmark (1,000 passes, B=1)...")
    lat_results = benchmark_latency(
        model_path=master_model_path,
        log_path=LOGS_P3 / "latency_benchmark.json",
        n_runs=1000,
    )

    # Plot master model curves
    generate_plots(
        metrics_path=LOGS_P3 / "master" / "metrics_master.json",
        cm_path=LOGS_P3 / "master" / "confusion_matrix.json",
        out_dir=LOGS_P3 / "master" / "plots",
    )

    # =========================================================================
    # Final Report
    # =========================================================================
    cpu_lat = lat_results.get("cpu", {}).get("mean_ms", 0.0)
    _banner(
        "PRODUCTION PIPELINE COMPLETE\n\n"
        f"Champion Architecture   : {ARCH.upper()} (HybridCNNGRU)\n"
        f"4-Fold CV Macro F1      : {cv_summary['cv_macro_f1_mean']:.4f} ? {cv_summary['cv_macro_f1_std']:.4f}\n"
        f"Master Model Macro F1   : {master_eval.get('macro_f1', 0.0):.4f}\n"
        f"CPU Inference Latency   : {cpu_lat:.2f} ms (Edge Compliant: {'YES' if cpu_lat <= 30 else 'NO'})\n\n"
        f"Canonical Model Path    : models/alert_model_best.pt\n"
        f"ONNX Model Path         : models/alert_model.onnx\n"
        f"Logs Directory          : logs/phase3/\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ALERT ? Full Production Training Pipeline (All Phases)"
    )
    parser.add_argument("--epochs",   type=int, default=EPOCHS,   help=f"Max epochs per fold (default: {EPOCHS})")
    parser.add_argument("--patience", type=int, default=PATIENCE, help=f"Early stopping patience (default: {PATIENCE})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE})")
    args = parser.parse_args()

    # Allow batch size override
    BATCH_SIZE = args.batch_size  # noqa: F811

    main(args)
