"""
plot_metrics.py
===============
Publication-quality visualization generator for ALERT training and evaluation metrics.

Outputs Generated
-----------------
1. logs/plots/loss_curve.png        — Train vs Validation Loss across epochs
2. logs/plots/accuracy_curve.png    — Train vs Validation Accuracy across epochs
3. logs/plots/f1_curves.png         — Per-Class F1 (Alert, Low Vigilant, Drowsy) and Macro-F1
4. logs/plots/confusion_matrix.png  — Dual-annotated Confusion Matrix (Counts + %)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def generate_training_plots(
    metrics_file: Path = _PROJECT_ROOT / "logs" / "training_metrics.json",
    cm_file: Path = _PROJECT_ROOT / "logs" / "confusion_matrix.json",
    output_dir: Path = _PROJECT_ROOT / "logs" / "plots",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # --- 1. Loss & Accuracy Curves ---
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            history = json.load(f)

        if history:
            epochs = [m["epoch"] for m in history]
            train_loss = [m["train_loss"] for m in history]
            val_loss = [m["val_loss"] for m in history]
            train_acc = [m["train_acc"] * 100 for m in history]
            val_acc = [m["val_acc"] * 100 for m in history]

            # Loss Curve
            plt.figure(figsize=(8, 5), dpi=300)
            plt.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=2.2)
            plt.plot(epochs, val_loss, label="Val Loss", color="#ff7f0e", linewidth=2.2, linestyle="--")
            plt.xlabel("Epoch", fontsize=11, fontweight="bold")
            plt.ylabel("Loss", fontsize=11, fontweight="bold")
            plt.title("ALERT — Training & Validation Loss", fontsize=13, fontweight="bold", pad=12)
            plt.legend(frameon=True, facecolor="white", framealpha=0.9)
            plt.tight_layout()
            loss_path = output_dir / "loss_curve.png"
            plt.savefig(loss_path)
            plt.close()
            print(f"[Plot Generated] {loss_path}")

            # Accuracy Curve
            plt.figure(figsize=(8, 5), dpi=300)
            plt.plot(epochs, train_acc, label="Train Accuracy (%)", color="#2ca02c", linewidth=2.2)
            plt.plot(epochs, val_acc, label="Val Accuracy (%)", color="#d62728", linewidth=2.2, linestyle="--")
            plt.xlabel("Epoch", fontsize=11, fontweight="bold")
            plt.ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
            plt.title("ALERT — Training & Validation Accuracy", fontsize=13, fontweight="bold", pad=12)
            plt.legend(frameon=True, facecolor="white", framealpha=0.9)
            plt.tight_layout()
            acc_path = output_dir / "accuracy_curve.png"
            plt.savefig(acc_path)
            plt.close()
            print(f"[Plot Generated] {acc_path}")

            # F1 Scores Progression
            if "val_macro_f1" in history[0]:
                plt.figure(figsize=(9, 5), dpi=300)
                macro_f1 = [m["val_macro_f1"] for m in history]
                f1_alert = [m.get("val_f1_alert", 0.0) for m in history]
                f1_lowvig = [m.get("val_f1_low_vigilant", 0.0) for m in history]
                f1_drowsy = [m.get("val_f1_drowsy", 0.0) for m in history]

                plt.plot(epochs, macro_f1, label="Macro F1", color="#333333", linewidth=2.5)
                plt.plot(epochs, f1_alert, label="Class 0 (Alert) F1", color="#1f77b4", linewidth=1.8, linestyle=":")
                plt.plot(epochs, f1_lowvig, label="Class 1 (Low Vigilant) F1", color="#ff7f0e", linewidth=2.2)
                plt.plot(epochs, f1_drowsy, label="Class 2 (Drowsy) F1", color="#d62728", linewidth=1.8, linestyle="-.")

                plt.xlabel("Epoch", fontsize=11, fontweight="bold")
                plt.ylabel("F1 Score", fontsize=11, fontweight="bold")
                plt.title("ALERT — Validation F1 Score Progression (Class 1 Focus)", fontsize=13, fontweight="bold", pad=12)
                plt.ylim(-0.05, 1.05)
                plt.legend(frameon=True, facecolor="white", framealpha=0.9, loc="lower right")
                plt.tight_layout()
                f1_path = output_dir / "f1_curves.png"
                plt.savefig(f1_path)
                plt.close()
                print(f"[Plot Generated] {f1_path}")

    # --- 2. Confusion Matrix Heatmap ---
    if cm_file.exists():
        with open(cm_file, "r") as f:
            cm_data = json.load(f)

        raw_cm = np.array(cm_data["confusion_matrix"])
        classes = cm_data.get("classes", ["Alert", "Low Vigilant", "Drowsy"])

        row_sums = raw_cm.sum(axis=1, keepdims=True)
        norm_cm = np.divide(raw_cm.astype(float), row_sums, out=np.zeros_like(raw_cm, dtype=float), where=row_sums != 0) * 100.0

        # Create dual-annotation labels: "count\n(percent%)"
        annot_matrix = np.empty_like(raw_cm, dtype=object)
        for i in range(raw_cm.shape[0]):
            for j in range(raw_cm.shape[1]):
                annot_matrix[i, j] = f"{raw_cm[i, j]:,}\n({norm_cm[i, j]:.1f}%)"

        plt.figure(figsize=(7, 6), dpi=300)
        sns.heatmap(
            norm_cm,
            annot=annot_matrix,
            fmt="",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            cbar_kws={"label": "Recall (%)"},
            linewidths=1.0,
            linecolor="white",
        )
        plt.xlabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=8)
        plt.ylabel("True Class", fontsize=11, fontweight="bold", labelpad=8)
        plt.title("ALERT — Normalized Confusion Matrix", fontsize=13, fontweight="bold", pad=14)
        plt.tight_layout()
        cm_path = output_dir / "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        print(f"[Plot Generated] {cm_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate ALERT metric plots")
    parser.add_argument("--metrics", type=str, default=str(_PROJECT_ROOT / "logs" / "training_metrics.json"), help="Path to training_metrics.json")
    parser.add_argument("--cm", type=str, default=str(_PROJECT_ROOT / "logs" / "confusion_matrix.json"), help="Path to confusion_matrix.json")
    parser.add_argument("--output-dir", type=str, default=str(_PROJECT_ROOT / "logs" / "plots"), help="Directory to save generated PNG plots")
    args = parser.parse_args()

    generate_training_plots(
        metrics_file=Path(args.metrics),
        cm_file=Path(args.cm),
        output_dir=Path(args.output_dir),
    )
