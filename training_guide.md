# ALERT — Training Guide

> **Complete reference for dataset preparation, feature caching, model architecture, and training the BiGRU-Attention FatigueClassifier.**

---

## Table of Contents

1. [Prerequisites & Environment Setup](#1-prerequisites--environment-setup)
2. [Dataset Structure (UTA-RLDD)](#2-dataset-structure-uta-rldd)
3. [Feature Extraction & Cache Pipeline](#3-feature-extraction--cache-pipeline)
4. [Cache File Format](#4-cache-file-format)
5. [Model Architecture](#5-model-architecture)
6. [Training Pipeline](#6-training-pipeline)
7. [Hyperparameter Reference](#7-hyperparameter-reference)
8. [Useful Training Commands](#8-useful-training-commands)
9. [Evaluation](#9-evaluation)
10. [Running Live Inference After Training](#10-running-live-inference-after-training)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Prerequisites & Environment Setup

### System Requirements

| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| **OS** | Linux, macOS, Windows 11 | Linux / Ubuntu 22.04 |
| **Python** | 3.10+ | 3.11 |
| **RAM** | 4 GB | 8+ GB |
| **CPU** | Any x86-64 quad-core | Intel 11th Gen+ (AVX-512 / VNNI) |
| **GPU** | Not required | Optional CUDA GPU for 10x speedup |
| **Disk** | 2 GB | 5 GB (for dataset + cache) |

### Installation (Linux / macOS)

```bash
# 1. Clone the repository
git clone https://github.com/zoro1126/ALERT.git
cd ALERT

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install PyTorch (CPU-only, platform-agnostic)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Verify
python3 -c "import torch, mediapipe, cv2; print('OK')"
```

### Installation (Windows 11)

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## 2. Dataset Structure (UTA-RLDD)

### Overview

- **Full Name:** University of Texas at Arlington Real-Life Drowsiness Dataset
- **Subjects:** 60 participants (diverse gender, ethnicity, age, accessories)
- **Total Videos:** 141 multi-minute recordings (each ~10 minutes, ~18,000 frames @ 30 FPS)
- **Classes:** 3 (balanced: 47 videos per class)
- **Validation Scheme:** Subject-independent 4-fold cross-validation

### Label Mapping (from video filename stem)

| Filename | Fatigue Level | Class Index |
| :---: | :--- | :---: |
| `0.mov` / `0.mp4` | Alert — fully vigilant, normal blink rate | **0** |
| `5.mov` / `5.mp4` | Low Vigilant — early fatigue, intermittent yawns, slower blinks | **1** |
| `10.mov` / `10.mp4` | Drowsy — micro-sleeps, prolonged closures, head nodding | **2** |

### Folder Layout

```
uta-reallife-drowsiness-dataset/
├── Fold1_part1/
│   └── Fold1_part1/
│       ├── 01/
│       │   ├── 0.mov        # Subject 01 → Alert
│       │   ├── 5.mov        # Subject 01 → Low Vigilant
│       │   └── 10.mov       # Subject 01 → Drowsy
│       ├── 02/ ...
│       └── 12/
├── Fold1_part2/             # Subjects 07-12 (continued)
├── Fold2_part1/ ...         # Subjects 13-24
├── Fold3_part1/ ...         # Subjects 25-36
└── Fold4_part1/ ...         # Subjects 37-48
```

> **Note:** Subject 32 in Fold 3 has its drowsy video split as `10_1.mp4` and `10_2.mp4`. These two files are skipped by the scanner (known limitation — total video count is 141, not 144).

### Dataset Scanner Code

The scanner lives in [`src/training/dataset.py`](src/training/dataset.py). It walks all folders under `Config.DATASET_ROOT`, parses the subject ID and label from path components, and assigns fold numbers based on parent folder name.

```python
from training.dataset import scan_dataset, dataset_summary

records = scan_dataset()
print(dataset_summary(records))
# Total videos: 141 | Alert: 47 | Low Vigilant: 47 | Drowsy: 47
```

---

## 3. Feature Extraction & Cache Pipeline

### Why Pre-Caching?

Running MediaPipe FaceLandmarker on a 10-minute 30 FPS video takes **~4–6 minutes per video** on CPU. With 141 videos, that is **9–14 hours per training run** if done live. Pre-caching compresses this to a **one-time 3.5-hour batch job**, after which every subsequent training run loads from 28.8 MB of compressed `.npz` files.

### The Extraction Pipeline (per video)

```
Raw Video (e.g., 18,000 frames @ 30 FPS)
│
▼ Step 1: Downsample
│   sample_every = round(native_fps / 15)  --> keep every 2nd frame
│   Result: ~9,000 frames @ 15 FPS
│
▼ Step 2: MediaPipe Tasks FaceLandmarker (IMAGE mode)
│   - Stateless per-frame inference (no tracking state between frames)
│   - Uses XNNPACK CPU delegate (fastest on-device acceleration)
│   - Extracts 468 3D normalized facial landmark points
│   - Skips frames where no face is detected
│
▼ Step 3: FeatureExtractor (geometric computation)
│   For each frame with detected face → compute 8D float32 vector:
│
│   [0] EAR_L      = Eye Aspect Ratio, left eye  (6 landmarks)
│   [1] EAR_R      = Eye Aspect Ratio, right eye (6 landmarks)
│   [2] EAR_avg    = (EAR_L + EAR_R) / 2
│   [3] MAR        = Mouth Aspect Ratio           (8 landmarks)
│   [4] PERCLOS    = % frames eye closed in rolling buffer
│   [5] head_pitch = 3D head pitch (degrees, solvePnP)
│   [6] head_yaw   = 3D head yaw (degrees, solvePnP)
│   [7] blink_rate = blinks per second in rolling 60-frame window
│
│   Result per video: numpy array of shape (N_frames_detected, 8), dtype float32
│
▼ Step 4: Sliding Window Segmentation
│   window_size = 60  frames  (4.0s of temporal context @ 15 FPS)
│   stride      = 15  frames  (1.0s step between consecutive windows)
│   starts = range(0, N - 60 + 1, 15)
│   windows = stack([features[s : s+60] for s in starts])
│   Result: numpy array of shape (M_windows, 60, 8), dtype float32
│
▼ Step 5: Compressed Storage
    np.savez_compressed(
        cache/fold{fold}_{subject}_{label}.npz,
        windows = (M, 60, 8) float32,
        label   = int32 scalar (0, 1, or 2),
        fold    = int32 scalar (1, 2, 3, or 4),
        subject = bytes (e.g., b"01")
    )
```

### EAR Formula (Eye Aspect Ratio)

```
         ||p2 - p6|| + ||p3 - p5||
EAR  =  ───────────────────────────
               2 × ||p1 - p4||

Where p1..p6 are the 6 eye landmark points (vertical/horizontal distances).
EAR ≈ 0.25–0.40  → eye open
EAR < threshold  → eye partially or fully closed
```

### Head Pose (solvePnP)

Six 3D canonical face anchor points (nose tip, chin, left/right eye corners, left/right mouth corners) are projected against their 2D landmark positions using OpenCV `cv2.solvePnP` to recover real-world Euler angles (Pitch, Yaw, Roll) via Rodrigues rotation decomposition.

### Running Cache Extraction

```bash
# Full extraction (all folds, ~3.5 hours on i7-7820HQ, ~2 hours on i5 11th Gen):
python3 scripts/extract_cache.py

# Single fold only (run 4 terminals in parallel for full speed):
python3 scripts/extract_cache.py --fold 1
python3 scripts/extract_cache.py --fold 2
python3 scripts/extract_cache.py --fold 3
python3 scripts/extract_cache.py --fold 4

# Dry run (no files written, just list what would be processed):
python3 scripts/extract_cache.py --dry-run

# Force re-extract (overwrite existing .npz files):
python3 scripts/extract_cache.py --force
```

### Extraction Output Summary

| Fold | Videos | Windows | File Size |
| :---: | :---: | :---: | :---: |
| 1 | 36 | ~21,780 | ~7.4 MB |
| 2 | 36 | ~21,540 | ~7.3 MB |
| 3 | 33 | ~20,939 | ~7.1 MB |
| 4 | 36 | ~20,683 | ~7.0 MB |
| **Total** | **141** | **~84,942** | **~28.8 MB** |

---

## 4. Cache File Format

Each `.npz` file in `cache/` follows the naming convention:

```
cache/fold{fold}_{subject}_{label}.npz
Example: cache/fold1_01_0.npz   (Fold 1, Subject 01, Alert class)
         cache/fold4_48_2.npz   (Fold 4, Subject 48, Drowsy class)
```

### Contents of each `.npz`

```python
import numpy as np

data = np.load("cache/fold1_01_0.npz")

data["windows"]  # shape: (M, 60, 8), dtype: float32
                 #  M  = number of windows from this video (~600-700)
                 #  60 = frames per window (4 seconds @ 15 FPS)
                 #  8  = feature dimensions

data["label"]    # scalar int32: 0 = Alert, 1 = Low Vigilant, 2 = Drowsy
data["fold"]     # scalar int32: 1, 2, 3, or 4
data["subject"]  # bytes: e.g. b"01"
data["n_frames"] # scalar int32: total frames extracted from the raw video
```

### Loading All Cache Files for Training

```python
from pathlib import Path
import numpy as np

cache_dir = Path("cache")
train_files = [p for p in cache_dir.glob("fold*.npz") if not p.name.startswith("fold4_")]
val_files   = [p for p in cache_dir.glob("fold4_*.npz")]

# Each file has its own M windows; concatenate all together
all_windows = np.concatenate([np.load(f)["windows"] for f in train_files])
# shape: (N_total_train, 60, 8)
```

---

## 5. Preprocessing Pipeline

> **Why this matters:** Raw features span wildly different scales (EAR ≈ 0.27 vs. Pitch ≈ ±30°). Without normalization, GRU gradients are dominated by Pitch/Yaw and the model overfits to per-subject camera-angle biases instead of learning fatigue signals.

The following two-step pipeline is applied **identically** at training time (inside `CachedWindowDataset.__getitem__`) and at inference time (inside `RealTimeClassifier._infer()`), ensuring no train/inference mismatch.

### Step 1 — Outlier Clipping (`_clip_window`)

Applied first, before normalization, to remove gimbal-lock artifacts:

| Feature | Clip Range | Reason |
| :--- | :--- | :--- |
| EAR_L / EAR_R / EAR_avg | `[0.0, 0.6]` | Physically bounded ~0–0.5; spikes to 20.8 are MediaPipe landmark collapses |
| MAR | `[0.0, 0.8]` | Max yawn ≈ 0.7; raw outliers reached 8.6 |
| Pitch | `[-60°, +60°]` | solvePnP gimbal lock wraps to ±179° at sharp head turns |
| Yaw | `[-90°, +90°]` | Safety net; data already within range |
| Blink rate | `[0.0, 4.0]` | p99.9 of training data; 10/s values are counting artifacts |

### Step 2 — Z-score Normalization (`FeatureNormalizer`)

- Statistics (mean, std) are computed from the **training split only** (no val leakage).
- Computed on **already-clipped** windows so the stats reflect the actual distribution the model sees.
- Saved to `models/feature_stats.npz` at the end of each training run.
- Loaded automatically by `RealTimeClassifier` from the same directory as `alert_model_best.pt`.

```
Raw window (60, 8)
  → _clip_window()              clamp outliers per feature
  → FeatureNormalizer           (x - mean) / std  per feature column
  → AugmentedWindowDataset      noise / warp / mask / crop  [train only]
  → FatigueClassifier (GRU)
```

### `models/feature_stats.npz` — Content

```python
import numpy as np
stats = np.load("models/feature_stats.npz")
stats["mean"]  # (8,) float64 — per-feature mean of clipped training windows
stats["std"]   # (8,) float64 — per-feature std  of clipped training windows
```

> **Important:** If you re-extract the cache or change the training split, re-train so a fresh `feature_stats.npz` is generated. Never share a stats file between different dataset splits.

---

## 5. Model Architecture

### FatigueClassifier (`src/core/model.py`)

```
Input: Tensor of shape (Batch, 60, 8)
  │   B = batch size, 60 = time steps, 8 = feature dims
  │
  ▼
┌─────────────────────────────────────────┐
│   Layer 1: Bidirectional GRU            │
│   hidden_size = 64  →  output = 128     │
│   dropout = 0.3 (between GRU layers)    │
└─────────────────────────────────────────┘
  │   Output: (B, 60, 128)
  ▼
┌─────────────────────────────────────────┐
│   Layer 2: Bidirectional GRU            │
│   hidden_size = 32  →  output = 64      │
└─────────────────────────────────────────┘
  │   Output: (B, 60, 64)
  ▼
┌─────────────────────────────────────────┐
│   Temporal Self-Attention               │
│   e_t = v^T * tanh(W * h_t + b)        │
│   α_t = softmax(e_t)                    │
│   c   = Σ (α_t * h_t)                  │
└─────────────────────────────────────────┘
  │   Output: (B, 64)  — attended context vector
  ▼
┌─────────────────────────────────────────┐
│   Classification Head                   │
│   Linear(64 → 64) → ReLU               │
│   Dropout(0.2)                          │
│   Linear(64 → 3)  → Softmax            │
└─────────────────────────────────────────┘
  │   Output: (B, 3)  — [P(Alert), P(Low_Vigilant), P(Drowsy)]
  ▼
Predicted Class = argmax(probabilities)
```

### Model Statistics

| Property | Value |
| :--- | :--- |
| **Total Parameters** | 63,939 |
| **Model File Size** | ~0.25 MB (`.pt`) |
| **Input Shape** | `(Batch, 60, 8)` |
| **Output Classes** | 3 (Alert, Low Vigilant, Drowsy) |
| **Inference Time (CPU)** | ~2.1 ms per 60-frame window |

---

## 6. Training Pipeline

### `src/training/trainer.py` — What Happens Inside

```
1. Load Cache
   ├── Split .npz files by fold: train vs. validation
   ├── CachedWindowDataset: loads all windows into RAM (~300 MB)
   └── Compute class weights for WeightedRandomSampler

2. Data Augmentation (training split only)
   AugmentedWindowDataset applies on-the-fly:
   ├── Gaussian Noise   (p=0.5): adds N(0, 0.01) to all features
   ├── Time Warp        (p=0.3): DTW-style elastic temporal distortion
   ├── Channel Mask     (p=0.2): zeros out one random feature column
   └── Temporal Crop    (p=0.3): crop 80-100%, interpolate back to 60 frames

3. Model & Optimizer
   ├── FatigueClassifier (BiGRU + Attention)
   ├── Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
   └── Scheduler: CosineAnnealingLR (T_max=50, eta_min=1e-6)

4. Training Loop (per epoch)
   ├── Forward pass → CrossEntropyLoss
   ├── Backward pass → gradient clip (max_norm=1.0) → AdamW step
   ├── Validation pass (no augmentation, no dropout)
   └── Save best checkpoint if val_loss improves

5. Early Stopping
   └── Stop if val_loss does not improve for 20 consecutive epochs

6. Outputs
   ├── models/alert_model_best.pt  (best checkpoint by val_loss)
   └── logs/training_metrics.json  (per-epoch train/val loss + accuracy)
```

### Checkpoint Format

```python
torch.save({
    "epoch":       epoch,
    "model_state": model.state_dict(),
    "val_loss":    val_loss,
    "val_acc":     val_acc,
    "config": {
        "n_features": 8,
        "hidden1":    64,
        "hidden2":    32,
        "fc_size":    64,
        "n_classes":  3,
    }
}, "models/alert_model_best.pt")
```

---

## 7. Hyperparameter Reference

All hyperparameters live in a single place: [`src/utils/config.py`](src/utils/config.py).

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `TARGET_FPS` | `15` | Downsample FPS for feature extraction |
| `WINDOW_SIZE` | `60` | Frames per temporal window (4 seconds) |
| `STRIDE` | `15` | Sliding window step (1 second) |
| `N_FEATURES` | `8` | Feature dimensions per frame |
| `HIDDEN_SIZE` | `64` | BiGRU Layer 1 hidden units |
| `HIDDEN_SIZE_2` | `32` | BiGRU Layer 2 hidden units |
| `FC_SIZE` | `64` | Classification head dense size |
| `DROPOUT_RNN` | `0.3` | Dropout between GRU layers |
| `DROPOUT_FC` | `0.2` | Dropout before final linear layer |
| `BATCH_SIZE` | `128` | Mini-batch size |
| `LR` | `1e-3` | Initial AdamW learning rate |
| `WEIGHT_DECAY` | `1e-4` | AdamW L2 regularization |
| `MAX_EPOCHS` | `50` | Maximum training epochs |
| `PATIENCE` | `20` | Early stopping patience (epochs) |
| `COSINE_T_MAX` | `50` | CosineAnnealing LR period |
| `RANDOM_SEED` | `42` | Global random seed |

---

## 8. Useful Training Commands

### Prerequisites for All Commands

```bash
cd /path/to/ALERT
source .venv/bin/activate        # Linux/macOS
# OR
.venv\Scripts\activate           # Windows

set PYTHONPATH=src               # Windows CMD
# OR
export PYTHONPATH=src            # Linux/macOS (set once per session)
```

### Core Training Command

```bash
# Standard training — fold 4 as validation (recommended)
PYTHONPATH=src python3 src/training/trainer.py --val-fold 4
# Uses: LR=1e-4, batch=128, train-stride=2 (50% overlap, ~32K windows)

# Fully independent windows (0% overlap, ~16K windows — slowest to memorise)
PYTHONPATH=src python3 src/training/trainer.py --val-fold 4 --train-stride 4

# All windows / old behaviour (NOT recommended — severe overfitting)
PYTHONPATH=src python3 src/training/trainer.py --val-fold 4 --train-stride 1

# Custom learning rate and patience
PYTHONPATH=src python3 src/training/trainer.py --val-fold 4 --lr 5e-5 --patience 15

# Use a different fold as validation
PYTHONPATH=src python3 src/training/trainer.py --val-fold 1

# Limit epochs for a quick smoke test
PYTHONPATH=src python3 src/training/trainer.py --val-fold 4 --epochs 5
```

### --train-stride Reference

| `--train-stride` | Windows loaded | Effective overlap | Use case |
| :---: | :---: | :---: | :--- |
| `1` | 64K | 75% | ⚠ Not recommended — severe overfitting observed |
| `2` *(default)* | 32K | 50% | Recommended — 4× slower divergence, good balance |
| `4` | 16K | 0% | Most independent samples; slower but best generalisation |

### Training Behaviour Benchmark

Before and after the pipeline fixes (normalization + clipping + stride + LR):

| Epoch | Old train | Old val | Old gap | New train | New val | New gap |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.877 | 1.489 | +0.61 | 1.031 | 1.086 | **+0.06** |
| 2 | 0.696 | 1.894 | +1.20 | 0.975 | 1.130 | **+0.15** |
| 3 | 0.584 | 2.118 | +1.53 | 0.947 | 1.188 | **+0.24** |
| 4 | 0.519 | 2.292 | +1.77 | 0.927 | 1.223 | **+0.29** |
| 5 | 0.477 | 2.318 | +1.84 | 0.905 | 1.278 | **+0.37** |

**Divergence rate: +0.31σ/epoch → +0.08σ/epoch (4× improvement)**. Full 50-epoch runs are expected to show val loss stabilising around epoch 15–25.

### Cache Extraction Commands

```bash
# Full dataset extraction (all 141 videos, ~3.5 hours sequential)
PYTHONPATH=src python3 scripts/extract_cache.py

# Parallel extraction — run each in a separate terminal:
PYTHONPATH=src python3 scripts/extract_cache.py --fold 1   # Terminal 1
PYTHONPATH=src python3 scripts/extract_cache.py --fold 2   # Terminal 2
PYTHONPATH=src python3 scripts/extract_cache.py --fold 3   # Terminal 3
PYTHONPATH=src python3 scripts/extract_cache.py --fold 4   # Terminal 4

# Dry run (preview without writing files)
PYTHONPATH=src python3 scripts/extract_cache.py --dry-run

# Force re-extraction (overwrite existing)
PYTHONPATH=src python3 scripts/extract_cache.py --force
```

### Inspect the Cache

```bash
# Count total windows across all folds
python3 -c "
from pathlib import Path; import numpy as np
files = list(Path('cache').glob('fold*.npz'))
total = sum(np.load(f)['windows'].shape[0] for f in files)
print(f'Total files: {len(files)}, Total windows: {total}')
"

# Check cache breakdown by fold/label
python3 -c "
from pathlib import Path; import numpy as np
from collections import Counter
breakdown = Counter()
for f in Path('cache').glob('fold*.npz'):
    d = np.load(f)
    breakdown[(int(str(f.stem).split('_')[0].replace('fold','')), int(d['label']))] += d['windows'].shape[0]
for k in sorted(breakdown):
    print(f'Fold {k[0]}, Label {k[1]}: {breakdown[k]} windows')
"
```

### Run Unit Tests

```bash
# Feature extractor tests (EAR, MAR, PERCLOS, head pose)
PYTHONPATH=src python3 -m pytest tests/test_feature_extractor.py -v

# Model architecture tests (gradients, attention, save/load)
PYTHONPATH=src python3 -m pytest tests/test_model.py -v
```

### Resume Training from Checkpoint

```python
# In Python — load checkpoint and inspect
import torch
ckpt = torch.load("models/alert_model_best.pt", weights_only=False)
print(f"Best epoch: {ckpt['epoch']}, Val Loss: {ckpt['val_loss']:.4f}, Val Acc: {ckpt['val_acc']:.4f}")
```

### Monitor Training Progress

At startup, the trainer prints the normalizer summary showing per-feature statistics computed on clipped training data:

```
Normalizer    : fitted on clipped data, saved to models/feature_stats.npz
Feature            Mean        Std
----------------------------------
EAR_L            0.2677     0.1536
EAR_R            0.2695     0.1534
EAR_avg          0.2695     0.1539
MAR              0.4416     0.0235
PERCLOS          0.6327     0.4560
Pitch           -0.6757    22.6459
Yaw             30.3677    33.5302
Blink_rate       0.1620     0.3907
```

Then the per-epoch training table:

```
Epoch  Train Loss  Train Acc  Val Loss  Val Acc        LR    Time
-----------------------------------------------------------------
    1      0.9520     0.5103    1.4044   0.2984  9.99e-04   68.5s
         ↳ saved best model (val_loss=1.4044)
    2      0.8134     0.6021    0.9873   0.5213  9.96e-04   67.1s
         ↳ saved best model (val_loss=0.9873)
```

Epoch metrics are also written in real-time to `logs/training_metrics.json`.

---

## 9. Evaluation

### Generate Confusion Matrix and Per-Class F1 Scores

```bash
PYTHONPATH=src python3 src/training/evaluate.py --val-fold 4
```

Outputs:
- **Terminal:** Precision, Recall, F1 per class (Alert, Low Vigilant, Drowsy)
- **`logs/confusion_matrix.json`:** Full confusion matrix for visualization

### Expected Target Performance on UTA-RLDD

| Metric | Threshold POC (Current) | BiGRU+Attention (Target) |
| :--- | :---: | :---: |
| **Alert F1** | ~0.70 (heuristic) | ~0.90+ |
| **Low Vigilant F1** | ~0.45 (hardest class) | ~0.78+ |
| **Drowsy F1** | ~0.68 (heuristic) | ~0.91+ |
| **Macro F1** | ~0.61 | **~0.86+** |

Low Vigilant is historically the most challenging class (early-stage subtle fatigue), consistent with prior literature on UTA-RLDD.

---

## 10. Running Live Inference After Training

Once `models/alert_model_best.pt` is produced:

```bash
# Full model inference (with BiGRU predictions displayed live)
python3 src/run_alert.py --camera 1

# With fresh baseline calibration (recommended for new sessions or new users)
python3 src/run_alert.py --camera 1 --recalibrate

# Threshold-only demo (no model needed — works before training)
python3 src/run_alert.py --camera 1 --demo

# Without FPS overlay in corner
python3 src/run_alert.py --camera 1 --no-fps

# Specify a custom trained model path
python3 src/run_alert.py --camera 1 --model models/alert_model_best.pt
```

> **Camera Index Note:** On most Linux machines with a virtual webcam (e.g., Iriun), use `--camera 1` for the physical integrated webcam. On Windows, typically `--camera 0`.

---

## 11. Troubleshooting

### `[RealTimeClassifier] WARNING: models/feature_stats.npz not found`

The normalizer stats file is generated automatically when you train the model. If you see this warning:
```bash
# Re-train to generate feature_stats.npz:
PYTHONPATH=src python3 src/training/trainer.py --val-fold 4
```
The model will still run but predictions will be unreliable because raw un-normalized features are fed to the GRU.

### `No module named 'cv2'` / `No module named 'mediapipe'`
```bash
# Make sure venv is activated:
source .venv/bin/activate     # or .venv\Scripts\activate
```

### `No training .npz files found in cache/`
```bash
# Run cache extraction first:
PYTHONPATH=src python3 scripts/extract_cache.py
```

### `Cannot open camera index 0` on Linux
```bash
# List available cameras:
ls /dev/video*
# Use the correct index (usually 1 if /dev/video0 is virtual):
python3 src/run_alert.py --camera 1 --demo
```

### `No module named 'pygame.mixer'` Warning
This is a harmless warning. The pygame installation in the venv is missing the SDL mixer module. Audio alerts will silently degrade to no sound. The visual HUD and all detection logic continue to work normally.

### Training is very slow / high memory usage
- Reduce batch size: `--batch-size 64`
- Reduce number of DataLoader workers (edit `num_workers=2` to `num_workers=0` in `trainer.py`)
- Close other RAM-heavy applications

### `KeyboardInterrupt` traceback on Ctrl-C during `run_alert.py`
This is a known MediaPipe Tasks cleanup issue on Python 3.14 when interrupted mid-session. The exit is still clean — no data is lost. Expected behavior.

---

*Generated: September 2026 — ALERT v0.1.0 — CE_AIML_31*
