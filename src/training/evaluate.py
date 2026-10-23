"""
evaluate.py
===========
Rigorous evaluation and metric generator for trained ALERT fatigue models.

Key Capabilities
----------------
1. Eliminates Normalizer Mismatch: Automatically extracts the exact training-fitted
   feature normalizer from the model checkpoint (or models/feature_stats.npz).
2. Comprehensive Metrics:
   - Overall Accuracy, Macro F1, Weighted F1
   - Per-class Precision, Recall, F1, and Support (Alert, Low Vigilant, Drowsy)
   - Confusion Matrix (both integer counts and row-normalized percentages)
   - Multi-class One-vs-Rest ROC-AUC per class and Macro-AUC
3. Export Artifacts: Saves structured JSON reports to logs/ for publication/dashboard.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from core.model import FatigueClassifier
from training.normalizer import FeatureNormalizer
from training.trainer import CachedWindowDataset, _split_cache_files
from utils.config import Config


def _compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def evaluate(
    val_fold:   int  = 4,
    model_path: Path = Path(Config.MODEL_PATH),
    cache_dir:  Path = Path(Config.CACHE_DIR),
    log_dir:    Path = Path(Config.LOG_DIR),
    batch_size: int  = 256,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}.\n"
            "Train the model first: python -m training.trainer"
        )

    print("=" * 70)
    print(f"ALERT — Model Evaluation Engine (Holdout: Fold {val_fold})")
    print("=" * 70)
    print(f"Loading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    model = FatigueClassifier(
        n_features=cfg.get("n_features", Config.N_FEATURES),
        hidden1=cfg.get("hidden1", Config.HIDDEN_SIZE),
        hidden2=cfg.get("hidden2", Config.HIDDEN_SIZE_2),
        fc_size=cfg.get("fc_size", Config.FC_SIZE),
        n_classes=cfg.get("n_classes", Config.N_CLASSES),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    epoch_trained = ckpt.get("epoch", "?")
    best_loss = ckpt.get("val_loss", 0.0)
    best_macro = ckpt.get("val_macro_f1", 0.0)
    print(f"Model trained for {epoch_trained} epochs | Best Val Loss: {best_loss:.4f} | Best Val Macro-F1: {best_macro:.4f}")

    # --- Load Matched Feature Normalizer ---
    normalizer = None
    if "normalizer_stats" in ckpt and isinstance(ckpt["normalizer_stats"], dict):
        normalizer = FeatureNormalizer.from_dict(ckpt["normalizer_stats"])
        print("[Normalizer] Loaded directly from checkpoint metadata.")
    else:
        stats_path = model_path.parent / "feature_stats.npz"
        if stats_path.exists():
            normalizer = FeatureNormalizer.load(stats_path)
            print(f"[Normalizer] Loaded from external file: {stats_path}")
        else:
            raise RuntimeError(
                f"No normalizer stats found in checkpoint or at {stats_path}.\n"
                "Models cannot be evaluated reliably without normalizer statistics."
            )

    # --- Data Loading ---
    _, val_files = _split_cache_files(cache_dir, val_fold)
    if not val_files:
        raise RuntimeError(f"No validation files found for fold {val_fold} in {cache_dir}")

    val_ds = CachedWindowDataset(val_files, normalizer=normalizer, subsample_stride=1)
    loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Validation set: {len(val_files)} videos | {len(val_ds):,} windows")

    # --- Inference ---
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
                logits, _ = model(x)
                probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    # --- Core Metrics ---
    n_classes = Config.N_CLASSES
    accuracy = float((y_true == y_pred).mean())
    cm = _compute_confusion_matrix(y_true, y_pred, n_classes)

    # Row-normalized confusion matrix (percentages per true class)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0

    # Scikit-learn Classification Report
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=Config.CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    macro_f1 = float(report_dict["macro avg"]["f1-score"])
    weighted_f1 = float(report_dict["weighted avg"]["f1-score"])

    # Multi-class ROC-AUC (One-vs-Rest)
    roc_auc_per_class = {}
    try:
        for c in range(n_classes):
            bin_y = (y_true == c).astype(int)
            if len(np.unique(bin_y)) > 1:
                auc = float(roc_auc_score(bin_y, y_prob[:, c]))
                roc_auc_per_class[Config.CLASS_NAMES[c]] = round(auc, 4)
            else:
                roc_auc_per_class[Config.CLASS_NAMES[c]] = 0.0
        macro_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception as e:
        macro_auc = 0.0

    # --- Display Results ---
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(f"Overall Accuracy : {accuracy * 100:.2f}%")
    print(f"Macro F1 Score   : {macro_f1:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print(f"Macro ROC-AUC    : {macro_auc:.4f}")
    print("-" * 70)
    print(f"{'Class':<16} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10} {'ROC-AUC':>10}")
    print("-" * 70)
    for c_name in Config.CLASS_NAMES:
        c_stats = report_dict[c_name]
        auc_val = roc_auc_per_class.get(c_name, 0.0)
        print(f"{c_name:<16} {c_stats['precision']:>10.4f} {c_stats['recall']:>10.4f} "
              f"{c_stats['f1-score']:>10.4f} {int(c_stats['support']):>10d} {auc_val:>10.4f}")
    print("-" * 70)

    print("\nCONFUSION MATRIX (Raw Counts):")
    lbl_header = "True \\ Pred"
    hdr = f"{lbl_header:<15}" + "".join(f"{name:>12}" for name in Config.CLASS_NAMES)
    print(hdr)
    print("-" * len(hdr))
    for i, row in enumerate(cm):
        print(f"{Config.CLASS_NAMES[i]:<15}" + "".join(f"{val:>12d}" for val in row))

    print("\nCONFUSION MATRIX (Normalized Accuracy % per True Class):")
    print(hdr)
    print("-" * len(hdr))
    for i, row in enumerate(cm_norm):
        print(f"{Config.CLASS_NAMES[i]:<15}" + "".join(f"{val:>11.1f}%" for val in row))

    # --- Save JSON Results ---
    log_dir.mkdir(parents=True, exist_ok=True)
    report_output = {
        "val_fold": val_fold,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_roc_auc": round(macro_auc, 4),
        "per_class_roc_auc": roc_auc_per_class,
        "classification_report": report_dict,
        "confusion_matrix_raw": cm.tolist(),
        "confusion_matrix_percent": np.round(cm_norm, 2).tolist(),
    }

    test_report_path = log_dir / "test_report.json"
    cm_path = log_dir / "confusion_matrix.json"

    test_report_path.write_text(json.dumps(report_output, indent=2))
    cm_path.write_text(json.dumps({
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_norm": np.round(cm_norm, 2).tolist(),
        "classes": Config.CLASS_NAMES,
    }, indent=2))

    print(f"\n[Saved Artifact] Full evaluation report: {test_report_path}")
    print(f"[Saved Artifact] Confusion matrix:      {cm_path}")
    print("=" * 70)

    return report_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ALERT Fatigue Detection Model")
    parser.add_argument("--val-fold", type=int, default=4, help="Held-out validation fold")
    parser.add_argument("--model", type=str, default=str(Config.MODEL_PATH), help="Path to model .pt")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for evaluation")
    args = parser.parse_args()

    evaluate(val_fold=args.val_fold, model_path=Path(args.model), batch_size=args.batch_size)
