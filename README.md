# ALERT — Adaptive Landmark Eye-state & Reaction-Time

> **Real-time driver drowsiness detection · Software-only · CPU · <2 GB RAM**

ALERT is a research-grade fatigue detection system built on MediaPipe FaceMesh and a BiGRU + Attention model trained on the UTA Real-Life Drowsiness Dataset. It runs entirely on CPU with a standard webcam and requires no EEG, CAN bus, or specialised hardware.

---

## Architecture

```
Webcam frame (BGR)
      │
      ▼
MediaPipe FaceLandmarker  (468-point FaceMesh, Tasks API)
      │
      ▼
FeatureExtractor  ──────────────────────────────────────────────────
  EAR_left · EAR_right · EAR_avg   (Soukupová & Čech 2016)
  MAR                               (Mouth Aspect Ratio — yawn proxy)
  PERCLOS                           (% frames eye closed in 60-frame window)
  head_pitch · head_yaw             (cv2.solvePnP, 6-point PnP)
  blink_rate                        (blinks/sec over rolling window)
      │  (8 features per frame)
      ▼
RealTimeClassifier  — rolling deque (60 frames), infers every 15 frames
      │
      ▼
FatigueClassifier (PyTorch)
  BiGRU(64) → BiGRU(32) → TemporalAttention → Dense(64) → Dense(3)
      │  (logits + attention weights)
      ▼
AlertManager  ──  NORMAL → WARNING → CRITICAL
      │            + procedural audio tones (pygame)
      ▼
OpenCV HUD  — status bar, feature panel, prob bars, attention strip
```

---

## Directory Structure

```
ALERT/
├── src/
│   ├── core/
│   │   ├── feature_extractor.py   # EAR, MAR, PERCLOS, head pose, blink rate
│   │   ├── model.py               # BiGRU + Attention (PyTorch)
│   │   ├── classifier.py          # rolling window + real-time inference
│   │   ├── calibrator.py          # adaptive EAR threshold calibration
│   │   └── alert_manager.py       # state machine + audio
│   ├── training/
│   │   ├── dataset.py             # UTA-RLDD scanner + fold split
│   │   ├── frame_extractor.py     # video → (N, 8) feature matrix
│   │   ├── augment.py             # temporal augmentations
│   │   ├── trainer.py             # training + validation loop
│   │   └── evaluate.py            # confusion matrix + F1 report
│   ├── ui/
│   │   ├── overlay.py             # OpenCV HUD drawing functions
│   │   └── dashboard.py           # main detection loop
│   └── run_alert.py               # CLI entry point
├── scripts/
│   └── extract_cache.py           # batch feature extraction → .npz cache
├── models/                        # saved .pt checkpoints (git-ignored)
├── cache/                         # .npz window cache (git-ignored)
├── logs/                          # training metrics + confusion matrix
├── tests/
│   ├── test_feature_extractor.py
│   └── test_model.py
├── requirements.txt
└── setup.py
```

---

## Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install PyTorch (CPU build)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install remaining dependencies
pip install -r requirements.txt
```

---

## Dataset

Place the **UTA Real-Life Drowsiness Dataset** at:

```
ALERT/uta-reallife-drowsiness-dataset/
  Fold1_part1/Fold1_part1/<subject>/{0,5,10}.mov
  Fold2_part1/...
  ...
```

Label mapping: `0.mov` → Alert (0) · `5.mov` → Low Vigilant (1) · `10.MOV` → Drowsy (2)

Download: [UTA-RLDD on IEEE DataPort](https://ieee-dataport.org/open-access/utarldd-uta-real-life-drowsiness-dataset)

---

## Training Pipeline

### Step 1 — Extract feature cache

```bash
# Dry-run first to verify all videos are found
python scripts/extract_cache.py --dry-run

# Extract features for all videos (~2–4 hours for 141 videos)
python scripts/extract_cache.py

# Or one fold at a time:
python scripts/extract_cache.py --fold 1
```

This creates `cache/fold<N>_<subject>_<label>.npz` files containing
`(M, 60, 8)` float32 windows ready for training.

### Step 2 — Train

```bash
python -m training.trainer --val-fold 4 --epochs 100
```

Best checkpoint saved to `models/alert_model_best.pt`.
Training log saved to `logs/training_metrics.json`.

### Step 3 — Evaluate

```bash
python -m training.evaluate --val-fold 4
```

Prints per-class precision / recall / F1.
Saves `logs/confusion_matrix.json`.

---

## Running the Detector

```bash
# Default (webcam 0, loads models/alert_model_best.pt)
python src/run_alert.py

# Custom model or camera
python src/run_alert.py --model models/my_model.pt --camera 1

# Re-run calibration (10-second open-eye session)
python src/run_alert.py --recalibrate
```

### First-run calibration

On first launch, ALERT runs a 10-second calibration phase where you look directly at the camera with eyes wide open. This sets a personalised EAR threshold stored at `~/.alert/calibration.json`. Subsequent launches skip calibration automatically.

---

## HUD Guide

| Element | Description |
|---|---|
| **Top bar** | Alert state (NORMAL / WARNING / CRITICAL) + predicted class + confidence |
| **Left panel** | Live EAR, MAR, PERCLOS, head pitch/yaw, blink rate — red = above threshold |
| **Right bars** | Per-class softmax probabilities |
| **Bottom strip** | Attention weight heatmap — which of the 60 frames drove the prediction |
| **Warm-up bar** | Shown until the 60-frame buffer is full (first ~4 seconds) |

**Press Q to quit.**

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover EAR/MAR geometry, PERCLOS and blink state machine,
FrameFeatures serialisation, HeadPoseEstimator camera matrix,
attention weight normalisation, gradient flow, dropout stochasticity,
and model save/load roundtrip.

---

## Resource Usage

| Resource | Value |
|---|---|
| RAM (runtime) | ~800 MB (MediaPipe + PyTorch + OpenCV) |
| Model size | 0.25 MB on disk |
| Parameters | 63,939 |
| Inference rate | Every 15 frames (1 second at 15 FPS) |

---

## Research Foundations

| Paper | Contribution |
|---|---|
| Liu et al. (2022) IET Image Processing | Foundational BiGRU architecture for temporal fatigue modeling |
| Sensors MDPI (2023) | Benchmark review — PERCLOS + temporal context outperforms EAR alone |
| Soukupová & Čech (2016) | Eye Aspect Ratio (EAR) formula |
| Guo et al. (2020) | 6-point solvePnP head pose estimation |
| Um et al. (2017) ACM ICMI | Time-series data augmentation (gaussian noise, time warp) |
| Bahdanau et al. (2015) | Additive attention mechanism |

---

*ALERT is a software-only research prototype. Do not rely on it as your sole means of staying awake while driving.*
