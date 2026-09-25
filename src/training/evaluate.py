"""
evaluate.py
===========
Loads the best saved model checkpoint and evaluates it on a held-out fold.

Usage (from project root)
--------------------------
  python -m training.evaluate                # uses val-fold 4 by default
  python -m training.evaluate --val-fold 3

Outputs
-------
  Prints per-class precision / recall / F1 and overall accuracy.
  Saves  logs/confusion_matrix.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from core.model import FatigueClassifier
from training.trainer import CachedWindowDataset, _split_cache_files
from utils.config import Config


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def _per_class_metrics(cm: np.ndarray) -> list[dict]:
    n = cm.shape[0]
    results = []
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        results.append({
            "class":     Config.CLASS_NAMES[c],
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "support":   int(cm[c, :].sum()),
        })
    return results


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    val_fold:   int  = 4,
    model_path: Path = Path(Config.MODEL_PATH),
    cache_dir:  Path = Path(Config.CACHE_DIR),
    log_dir:    Path = Path(Config.LOG_DIR),
) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load checkpoint ---
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run: python -m training.trainer first."
        )
    ckpt = torch.load(model_path, map_location=device)
    cfg  = ckpt.get("config", {})

    model = FatigueClassifier(
        n_features = cfg.get("n_features", Config.N_FEATURES),
        hidden1    = cfg.get("hidden1",    Config.HIDDEN_SIZE),
        hidden2    = cfg.get("hidden2",    Config.HIDDEN_SIZE_2),
        fc_size    = cfg.get("fc_size",    Config.FC_SIZE),
        n_classes  = cfg.get("n_classes",  Config.N_CLASSES),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}  "
          f"(val_loss={ckpt.get('val_loss', 0):.4f})")

    # --- Data ---
    _, val_files = _split_cache_files(cache_dir, val_fold)
    if not val_files:
        raise RuntimeError(f"No val files for fold {val_fold} in {cache_dir}")

    val_ds = CachedWindowDataset(val_files)
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=128, shuffle=False, num_workers=2
    )

    # --- Inference ---
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits, _ = model(x)
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    # --- Metrics ---
    n_classes = Config.N_CLASSES
    accuracy  = (y_true == y_pred).mean()
    cm        = _confusion_matrix(y_true, y_pred, n_classes)
    per_class = _per_class_metrics(cm)
    macro_f1  = np.mean([m["f1"] for m in per_class])

    print(f"\nOverall accuracy : {accuracy:.4f}")
    print(f"Macro F1         : {macro_f1:.4f}")
    print(f"\n{'Class':<15} {'Precision':>9} {'Recall':>7} {'F1':>6} {'Support':>8}")
    print("-" * 50)
    for m in per_class:
        print(f"{m['class']:<15} {m['precision']:>9.4f} {m['recall']:>7.4f} "
              f"{m['f1']:>6.4f} {m['support']:>8}")

    print(f"\nConfusion matrix (rows=true, cols=pred):")
    header = "         " + "".join(f"{n[:7]:>9}" for n in Config.CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        label = Config.CLASS_NAMES[i][:7]
        print(f"{label:<9}" + "".join(f"{v:>9}" for v in row))

    # --- Save ---
    log_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "val_fold":   val_fold,
        "accuracy":   round(float(accuracy), 4),
        "macro_f1":   round(float(macro_f1), 4),
        "per_class":  per_class,
        "confusion_matrix": cm.tolist(),
    }
    out_path = log_dir / "confusion_matrix.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-fold", type=int, default=4)
    args = parser.parse_args()
    evaluate(val_fold=args.val_fold)
