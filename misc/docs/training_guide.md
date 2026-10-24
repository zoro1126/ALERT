# ALERT — Model Training & Engineering Guide

> **Authoritative Handbook for Dataset Preparation, Feature Extraction, Edge-Compliant Model Design, and Experimental Validation on the UTA-RLDD Dataset.**

---

## ⚠️ Master Rule: Zero Fake / Synthetic Data (Non-Negotiable)

```
╔════════════════════════════════════════════════════════════════════════════════════════╗
║                                 ABSOLUTE DIRECTIVE                                     ║
║                                                                                        ║
║  1. NEVER train, fine-tune, or pre-train on synthetic, mocked, or AI-generated data.  ║
║  2. NEVER fabricate or simulate training curves, evaluation metrics, or test outputs. ║
║  3. ALL models must be trained and validated EXCLUSIVELY on the authentic,            ║
║     real-world recordings of the UTA-RLDD dataset.                                    ║
║  4. Ground truth video files and genuine physical landmark extractions are the ONLY    ║
║     permissible source of data in this project.                                        ║
╚════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## Table of Contents

1. [Developer Mandate & Scope of Freedom](#1-developer-mandate--scope-of-freedom)
2. [Target Hardware Setup (RTX 3050 Laptop / CUDA)](#2-target-hardware-setup-rtx-3050-laptop--cuda)
3. [UTA-RLDD Dataset Architecture](#3-uta-rldd-dataset-architecture)
4. [Feature Extraction & Caching Strategies](#4-feature-extraction--caching-strategies)
5. [Current Pipeline Diagnosis & Known Failure Modes](#5-current-pipeline-diagnosis--known-failure-modes)
6. [Edge Hardware Compliance & Constraints](#6-edge-hardware-compliance--constraints)
7. [Candidate Architectures for Edge Deployment](#7-candidate-architectures-for-edge-deployment)
8. [Comprehensive Evaluation & Metric Generation](#8-comprehensive-evaluation--metric-generation)
9. [Training Protocols & Execution Commands](#9-training-protocols--execution-commands)
10. [Export & Deployment Pipeline (ONNX / Edge)](#10-export--deployment-pipeline-onnx--edge)
11. [Troubleshooting & Best Practices](#11-troubleshooting--best-practices)

---

## 1. Developer Mandate & Scope of Freedom

As the lead machine learning engineer on this module, **you have full authorization and freedom to modify, refactor, or completely rewrite any component** of the training and modeling pipeline:

- **Feature Engineering & Extraction:** You may redesign how features are extracted, computed, filtered, or aggregated from MediaPipe facial landmarks and head poses. You can modify [`src/features/feature_extractor.py`](src/features/feature_extractor.py) and [`scripts/extract_cache.py`](scripts/extract_cache.py).
- **Cache Formats & Storage:** You are free to change the cache format from `.npz` to `.pt`, `.h5`, `.parquet`, or memory-mapped binary tensors if it accelerates IO or training throughput.
- **Preprocessing & Normalization:** You can replace global z-score normalization with per-subject baseline subtraction, rolling window normalization, min-max scaling, or temporal filtering (e.g., Butterworth low-pass, Savitzky-Golay).
- **Model Architectures:** You are encouraged to design and benchmark new architectures (TCN, 1D-CNN+GRU, MobileNetV3-1D, Transformer/Informer, Quantized GRU) in [`src/core/model.py`](src/core/model.py) or new model modules, provided they comply with edge hardware constraints.
- **Loss Functions & Optimization:** You can experiment with Focal Loss, Class-Balanced Loss, Label Smoothing, AdamW, Lookahead, SAM, Cosine Annealing with Warm Restarts, etc.
- **Configuration & Hyperparameters:** You have full authority to modify [`src/utils/config.py`](src/utils/config.py) and CLI flags.

**Your Single Objective:** Train a robust, reliable, and deployable driver fatigue classification model that generalizes well across unseen subjects on the UTA-RLDD dataset.

---

## 2. Target Hardware Setup (RTX 3050 Laptop / CUDA)

This guide is tailored for development on an **NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM) paired with a modern multi-core CPU**.

### 2.1 PyTorch with CUDA Acceleration

Do not use the CPU-only build. Install the CUDA 12.1 or 11.8 PyTorch wheel:

```bash
# 1. Activate your virtual environment
source .venv/bin/activate          # Linux
# or
.venv\Scripts\activate           # Windows

# 2. Install PyTorch with CUDA 12.1 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Verify CUDA detection and GPU name
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### 2.2 GPU Acceleration Best Practices for 6 GB VRAM

1. **Automatic Mixed Precision (AMP):**
   Wrap the training forward pass with `torch.amp.autocast('cuda')` and `torch.cuda.amp.GradScaler()`. This cuts VRAM usage by ~40% and speeds up GRU/linear layer computations on Tensor Cores.
2. **Batch Size Tuning:**
   On a 6 GB RTX 3050, you can comfortably increase batch size to `256` or `512` for 60-frame 8D feature windows.
3. **DataLoader Optimization:**
   Set `pin_memory=True` and `num_workers=4` for asynchronous CPU-to-GPU memory transfer:
   ```python
   DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=True, num_workers=4)
   ```
4. **PyTorch 2.0+ Compilation (Optional):**
   ```python
   model = torch.compile(model)  # Optimizes CUDA kernels and fuses operations
   ```

---

## 3. UTA-RLDD Dataset Architecture

### 3.1 Overview

- **Full Name:** University of Texas at Arlington Real-Life Drowsiness Dataset
- **Subjects:** 60 participants (diverse gender, ethnicity, age, facial hair, glasses)
- **Recordings:** 141 multi-minute videos (~10 minutes each, recorded at 30 FPS, ~18,000 frames/video)
- **Total Frames:** Over 2.5 million raw video frames across all folds
- **Validation Protocol:** Strict **Subject-Independent Cross-Validation** (subjects in validation fold never appear in training folds).

### 3.2 Label Classes

| Class Index | Label Name | Behavioral Manifestation | UTA-RLDD Video Suffix |
| :---: | :--- | :--- | :---: |
| **0** | **Alert** | Normal blink dynamics, vigilant gaze, erect posture | `0.mov` / `0.mp4` |
| **1** | **Low Vigilant** | Early fatigue: slower saccades, reduced blink rate, slight head drift | `5.mov` / `5.mp4` |
| **2** | **Drowsy** | Severe fatigue: microsleeps, prolonged eye closures, yawns, head nods | `10.mov` / `10.mp4` |

> **Crucial Note on Class 1 (Low Vigilant):** In UTA-RLDD literature, Class 1 is consistently the hardest class to classify (baseline F1 often sits around 0.40–0.50), because early fatigue features overlap with relaxed alert states. Special loss weighting or focal losses are highly recommended.

### 3.3 Directory Layout

```
uta-reallife-drowsiness-dataset/
├── Fold1_part1/
│   ├── 01/
│   │   ├── 0.mov        # Subject 01 — Alert
│   │   ├── 5.mov        # Subject 01 — Low Vigilant
│   │   └── 10.mov       # Subject 01 — Drowsy
│   ├── 02/ ...
├── Fold1_part2/         # Subjects 07-12
├── Fold2_part1/         # Subjects 13-24
├── Fold3_part1/         # Subjects 25-36 (Note: Subject 32 has 10_1.mp4, 10_2.mp4)
└── Fold4_part1/         # Subjects 37-48
```

---

## 4. Feature Extraction & Caching Strategies

### 4.1 Current Extraction Method

The extraction script [`scripts/extract_cache.py`](scripts/extract_cache.py) performs:
1. Video downsampling from 30 FPS to 15 FPS (sample every 2nd frame).
2. MediaPipe FaceLandmarker detection (468 3D landmarks).
3. Geometric calculation of 8 hand-crafted signals per frame:
   - `[0] EAR_L` (Left Eye Aspect Ratio)
   - `[1] EAR_R` (Right Eye Aspect Ratio)
   - `[2] EAR_avg` (Mean of left and right EAR)
   - `[3] MAR` (Mouth Aspect Ratio)
   - `[4] PERCLOS` (Percentage of eyelid closure over rolling buffer)
   - `[5] Pitch` (Head pitch in degrees via `cv2.solvePnP`)
   - `[6] Yaw` (Head yaw in degrees via `cv2.solvePnP`)
   - `[7] Blink_rate` (Detected blinks per second)
4. Segmentation into overlapping sliding windows: `window_size = 60` frames (4.0s @ 15 FPS), `stride = 15` frames (1.0s step).
5. Saving each video as `cache/fold{fold}_{subject}_{label}.npz`.

### 4.2 Why and How to Re-Extract (If You Choose To)

You have full freedom to re-extract the dataset with improved representations. With your RTX 3050 and fast CPU, extraction will be significantly faster than CPU-only runs.

#### High-Value Feature Enhancements:
1. **Delta-Pose ($\Delta\text{Yaw}$, $\Delta\text{Pitch}$, $\Delta\text{Roll}$):**
   Raw `Yaw` carries a massive ~30° subject-dependent camera angle bias. Replacing absolute pose with first-order and second-order temporal differences ($\text{pose}_t - \text{pose}_{t-1}$) removes static camera bias while preserving head drops (nodding).
2. **Subject-Calibrated Baselines:**
   Compute the neutral EAR and MAR from the first 30 seconds of each video (or from the Alert video of each subject), and express features as ratios:
   $$\text{EAR}_{\text{rel}} = \frac{\text{EAR}_t}{\text{EAR}_{\text{baseline}}}$$
3. **Head Motion Energy:**
   Add kinetic energy or rotational velocity magnitude: $\sqrt{(\Delta \text{pitch})^2 + (\Delta \text{yaw})^2}$.
4. **Differential Eye Features:**
   Add blink duration (frame count of consecutive closure) and eyelid velocity (derivative of EAR).

---

## 5. Current Pipeline Diagnosis & Known Failure Modes

Understanding why earlier baseline attempts plateaued will save you dozens of hours:

### 5.1 Observation from Baseline Training Run

When training the standard BiGRU model on fold 4 validation:
```
Epoch  Train Loss  Train Acc  Val Loss  Val Acc        LR
    1      1.0996     0.3614    1.0842   0.3449  9.99e-06
    5      1.0500     0.4431    1.0718   0.4134  9.78e-06  <-- Min Val Loss
   10      1.0267     0.4665    1.0742   0.4150  9.14e-06
   20      1.0023     0.4969    1.0851   0.4114  6.89e-06
   30      0.9861     0.5145    1.0962   0.4118  4.11e-06
Early stopping at epoch 30 (patience 25).
```

### 5.2 Root Causes Diagnosed

1. **Systematic Head Pose Bias (Camera Angle Overfitting):**
   - In the training folds, mean Yaw is ~30.3° with std 33.5°.
   - Each subject's camera position is fixed, meaning absolute Yaw acts as a "subject fingerprint" rather than a fatigue signal.
   - The model learns to associate specific camera angles with specific subjects, failing completely when evaluated on unseen subjects in fold 4.
2. **Class Confusion (Alert vs. Low Vigilant):**
   - The boundary between Alert (0) and Low Vigilant (1) in feature space is subtle. The model tends to collapse predictions into Class 0 and Class 2, leaving Class 1 under-detected.
3. **Window Stride Redundancy:**
   - A stride of 15 creates 75% frame overlap between adjacent windows. Consecutive windows are nearly identical, causing the model to rapidly memorize sequence sequences if stride subsampling is not used.

---

## 6. Edge Hardware Compliance & Constraints

The final ALERT model must run inside vehicles on **edge hardware** (e.g., Raspberry Pi 4/5, NVIDIA Jetson Nano/Orin Nano, low-power Intel/AMD laptop CPUs, or embedded Android/ARM SoCs).

### Mandatory Edge Budgets

| Metric | Edge Target | Hard Ceiling | Rationale |
| :--- | :---: | :---: | :--- |
| **Inference Latency** | **$\le 5$ ms** (GPU/NPU) / **$\le 15$ ms** (CPU) | **$30$ ms** | Must maintain $\ge 30$ FPS real-time processing |
| **Parameter Count** | **$50\text{k} - 250\text{k}$** | **$1.0\text{M}$** | Minimizes memory bandwidth & cache misses |
| **Model Disk Size** | **$< 2$ MB** | **$< 10$ MB** | Over-the-air firmware update compliance |
| **Peak Runtime RAM** | **$< 50$ MB** | **$< 150$ MB** | Runs alongside MediaPipe and UI processes |
| **Quantization** | INT8 / FP16 ready | ONNX compatible | Seamless deployment via ONNXRuntime / TensorRT |

---

## 7. Candidate Architectures for Edge Deployment

You are encouraged to implement and benchmark alternatives to the baseline BiGRU.

```
                           INPUT: (Batch, 60, F)
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           ▼                         ▼                         ▼
   [Option A: BiGRU+Attn]     [Option B: TCN 1D]        [Option C: CNN-GRU]
    - 2-layer BiGRU (64, 32)   - Dilated causal 1D Conv  - 1D Conv (kernel 3, 5)
    - Temporal Attention       - Residual connections    - MaxPool 1D
    - Dense (64) + Dropout     - Receptive field = 60    - Single GRU (48)
           │                         │                         │
           └─────────────────────────┼─────────────────────────┘
                                     ▼
                            Logits: (Batch, 3)
```

### Option A: Refined BiGRU + Multi-Head Attention (Current Baseline)
- **Strengths:** Excellent temporal modeling of forward and backward micro-sleep events.
- **Tuning:** Reduce hidden size to `hidden1=48, hidden2=24` to avoid overfitting and accelerate inference.

### Option B: Temporal Convolutional Network (TCN / Dilated 1D-CNN)
- **Strengths:** Fully convolutional, highly parallelizable on GPUs and NPUs, zero recurrent state overhead.
- **Layers:** 3–4 dilated Conv1D layers (dilation factors: 1, 2, 4, 8) with kernel size 3, BatchNorm, and ReLU.
- **Latency:** Often 2–3× faster than GRUs on edge CPUs.

### Option C: Hybrid 1D-CNN + Lightweight GRU
- **Strengths:** 1D Conv acts as a local feature extractor across consecutive frames (e.g., detecting rapid eyelid descent), feeding high-level temporal features into a small single-layer GRU (32 units).

---

## 8. Comprehensive Evaluation & Metric Generation

To guarantee true model quality and avoid misleading accuracy numbers, **you must log and generate all standard evaluation artifacts**.

### 8.1 Required Metrics Per Model & Per Fold

1. **Loss Curves:** Training loss and Validation loss plotted across all epochs.
2. **Accuracy Curves:** Training accuracy and Validation accuracy plotted across all epochs.
3. **Confusion Matrix:** Raw counts and row-normalized percentages showing true vs. predicted classes.
4. **Per-Class Metrics:**
   - Precision, Recall, and F1-score for each class (Alert, Low Vigilant, Drowsy).
   - Macro F1 and Weighted F1.
5. **ROC Curves & AUC:** One-vs-Rest ROC curve and AUC score for each of the 3 classes.
6. **Inference Latency Benchmark:** Average execution time over 1,000 runs on CPU and GPU.

### 8.2 Logging Directory Structure

All metrics and generated plots should be saved in `logs/`:

```
logs/
├── training_metrics.json          # Per-epoch loss, acc, lr history
├── test_report.json               # Final precision, recall, f1, support
├── confusion_matrix.png           # Visual heatmap of confusion matrix
├── loss_curve.png                 # Train vs Val loss graph
├── accuracy_curve.png             # Train vs Val accuracy graph
├── roc_curves.png                 # One-vs-Rest ROC plots
└── model_comparison.json         # Leaderboard comparing different candidate models
```

### 8.3 Automated Metric & Plot Generation Script

Ensure your pipeline generates publication-quality plots. Use or adapt the following script:

```python
# scripts/plot_metrics.py
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def generate_training_plots(metrics_file: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "r") as f:
        history = json.load(f)

    epochs = [m["epoch"] for m in history]
    train_loss = [m["train_loss"] for m in history]
    val_loss   = [m["val_loss"] for m in history]
    train_acc  = [m["train_acc"] for m in history]
    val_acc    = [m["val_acc"] for m in history]

    # 1. Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=2)
    plt.plot(epochs, val_loss,   label="Val Loss",   color="#ff7f0e", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training and Validation Loss Curve")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=300)
    plt.close()

    # 2. Accuracy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="Train Accuracy", color="#2ca02c", linewidth=2)
    plt.plot(epochs, val_acc,   label="Val Accuracy",   color="#d62728", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy Curve")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=300)
    plt.close()

    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    generate_training_plots(Path("logs/training_metrics.json"), Path("logs/plots"))
```

---

## 9. Training Protocols & Execution Commands

### 9.1 Environment Initialization

```bash
# Activate virtual environment
source .venv/bin/activate        # Linux
# or
.venv\Scripts\activate           # Windows

# Set Python path
export PYTHONPATH=src            # Linux/macOS
# or
set PYTHONPATH=src               # Windows CMD
```

### 9.2 Cache Extraction (If Modifying Features)

```bash
# Single GPU / fast extraction on all 4 folds
PYTHONPATH=src python3 scripts/extract_cache.py --force

# Or run parallel workers per fold:
PYTHONPATH=src python3 scripts/extract_cache.py --fold 1 &
PYTHONPATH=src python3 scripts/extract_cache.py --fold 2 &
PYTHONPATH=src python3 scripts/extract_cache.py --fold 3 &
PYTHONPATH=src python3 scripts/extract_cache.py --fold 4 &
```

### 9.3 Model Training Commands

```bash
# 1. Standard CUDA training with Fold 4 as Validation
PYTHONPATH=src python3 src/training/trainer.py \
    --val-fold 4 \
    --epochs 50 \
    --batch-size 256 \
    --lr 1e-4 \
    --train-stride 2 \
    --patience 20

# 2. Fully Independent Windows (0% train overlap, best for generalisation)
PYTHONPATH=src python3 src/training/trainer.py \
    --val-fold 4 \
    --train-stride 4 \
    --batch-size 128 \
    --lr 5e-5 \
    --patience 25

# 3. 4-Fold Cross-Validation Routine (Run all 4 folds to test true generalisation)
for fold in 1 2 3 4; do
    echo "=== Running Training on Val Fold $fold ==="
    PYTHONPATH=src python3 src/training/trainer.py \
        --val-fold $fold \
        --epochs 40 \
        --batch-size 256 \
        --lr 1e-4
done
```

### 9.4 Full Model Evaluation

```bash
# Evaluate best checkpoint and generate full classification report
PYTHONPATH=src python3 src/training/evaluate.py --val-fold 4

# Generate visual curve plots
python3 scripts/plot_metrics.py
```

---

## 10. Export & Deployment Pipeline (ONNX / Edge)

Once your best model is trained, export it to **ONNX** format for high-speed edge inference:

### 10.1 Exporting to ONNX

```python
# scripts/export_onnx.py
import torch
from pathlib import Path
from core.model import FatigueClassifier

def export():
    model = FatigueClassifier()
    ckpt = torch.load("models/alert_model_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy_input = torch.randn(1, 60, 8, dtype=torch.float32)
    onnx_path = Path("models/alert_model.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_window"],
        output_names=["logits", "attention_weights"],
        dynamic_axes={"input_window": {0: "batch_size"}},
    )
    print(f"Exported ONNX model successfully to {onnx_path}")

if __name__ == "__main__":
    export()
```

### 10.2 Verifying with ONNX Runtime

```bash
pip install onnxruntime
python3 -c "
import onnxruntime as ort, numpy as np
session = ort.InferenceSession('models/alert_model.onnx')
x = np.random.randn(1, 60, 8).astype(np.float32)
out = session.run(None, {'input_window': x})
print('ONNX inference output shape:', out[0].shape)
"
```

---

## 11. Troubleshooting & Best Practices

| Symptom | Root Cause | Fix / Solution |
| :--- | :--- | :--- |
| **CUDA out of memory (OOM)** | Batch size too large for 6 GB VRAM | Reduce `--batch-size` to `128` or `64`. Enable `torch.amp.autocast`. |
| **Validation loss diverges while train loss drops** | Subject camera-angle memorization or high overlap | Use `--train-stride 4`. Switch absolute Yaw/Pitch to $\Delta\text{Yaw}$ / $\Delta\text{Pitch}$. Apply Dropout = 0.4. |
| **Class 1 (Low Vigilant) F1 score is near zero** | Class boundary overlap and imbalance | Implement **Focal Loss** ($\gamma=2.0$) or adjust `CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 1.2]))`. |
| **DataLoader worker crashed / frozen** | Forkserver issue on multi-worker setup | Set `num_workers=0` or `num_workers=2` in `_make_loader()`. |
| **Normalizer stats missing warning at inference** | `models/feature_stats.npz` not generated | Always verify `trainer.py` saves `feature_stats.npz`. Never copy stats between different fold splits. |

---

*ALERT Engineering Documentation — Maintained for Core Model Development & Edge Optimization.*
