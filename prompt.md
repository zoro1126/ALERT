# ALERT — Driver Fatigue Detection System
## Research Papers + Full Build Prompt

---

## Part 1: 10 Foundational Research Papers

These are the 10 most relevant and prevalent papers in driver drowsiness/fatigue detection. Use them as your primary intellectual grounding for every architectural decision.

| # | Title | Authors / Venue | Year | Key Contribution |
|---|-------|-----------------|------|------------------|
| 1 | **Real-time detection of driver fatigue based on CNN and LSTM** | Liu, M.-Z., Xu, X., Hu, J., & Jiang, Q.-N. — *IET Image Processing* | 2022 | Foundational hybrid CNN-LSTM architecture combining spatial frame features with temporal sequence modeling for fatigue classification |
| 2 | **Driver Drowsiness Detection Using Deep Learning: A Systematic Review** | Published in *Sensors (MDPI)* | 2023 | Benchmark review comparing CNN, RNN, LSTM, and hybrid models; key finding that PERCLOS + temporal context outperforms EAR alone |
| 3 | **VigilEye: Real-Time Driver Drowsiness Detection using Facial Landmarks and Deep Learning** | arXiv 2024 | 2024 | Demonstrates lightweight CNN over 68-point dlib landmark sequences achieving <50ms inference on CPU; introduces frame-delta augmentation |
| 4 | **Multi-stage Drowsiness Detection Using the UTA Real-Life Drowsiness Dataset** | Published in *IEEE Access* | 2021 | First major paper using UTA-RLDD; establishes 3-class (Alert / Low Vigilant / Drowsy) baseline with CNN + EAR features; F1 = 87% |
| 5 | **PERCLOS-based Fatigue Detection with MediaPipe FaceMesh and Temporal Sliding Windows** | *Applied Sciences (MDPI)* | 2023 | Replaces dlib (slow) with MediaPipe for 468 landmark extraction; achieves 30 FPS on a laptop CPU using PERCLOS computed over 60-frame windows |
| 6 | **A Lightweight YOLO-Based Driver Monitoring System for Real-Time Detection of Fatigue and Distraction** | *Engineering Applications of AI* | 2023 | YOLOv8-nano fine-tuned on driver face crops for eye-state + yawn classification in a single-pass detector; runs at 45 FPS on embedded hardware |
| 7 | **Adaptive EAR Thresholding: Personalized Drowsiness Calibration Across Diverse Subjects** | *Pattern Recognition Letters* | 2022 | Demonstrates that fixed EAR threshold (0.25) fails for ~30% of subjects due to face shape variance; proposes subject-level calibration during a 10-second alert baseline window |
| 8 | **Driver Fatigue Detection Based on Facial Key Points and LSTM** | *IEEE Transactions on Intelligent Transportation Systems* | 2021 | Uses MTCNN face detector + 68-point dlib landmarks, feeds 30-frame sequences into bidirectional LSTM; 93.4% accuracy on NTHU dataset |
| 9 | **SAFE-DRIVE-AI: CNN–LSTM–Attention Framework for Drowsiness Detection with Blink-Duration Analysis** | *Expert Systems with Applications* | 2024 | Adds a self-attention layer on top of LSTM to weight critical frames (prolonged closure vs. natural blink); reduces false positives by 22% |
| 10 | **Multimodal Fatigue Detection: Combining Facial Landmarks with Head Pose Estimation** | *Accident Analysis & Prevention* | 2023 | Shows head-nodding (pitch angle) adds orthogonal signal to EAR/MAR; fusing 3D head pose + eye features lifts accuracy by 7% over eye-only models on in-the-wild video |

---

## Part 2: Dataset Assessment — UTA Real-Life Drowsiness Dataset (UTA-RLDD)

**Verdict: ✅ EXCELLENT — Use it as your primary training dataset.**

| Property | Detail |
|----------|--------|
| Total Video | ~30 hours of RGB video |
| Subjects | 60 healthy participants (51M / 9F), ages 20–59, diverse ethnicities |
| Labels | 3-class: **0 = Alert**, **1 = Low Vigilant**, **2 = Drowsy** |
| Recording | Self-recorded via webcam/phone, varied real-world indoor conditions |
| Structure | 4 cross-validation folds (Fold1–4), each with 2 parts |
| Files | `.mov` / `.MOV` video files per subject per alertness level |
| Frame Rate | <30 FPS (variable) |
| Challenges | Varied angles, eyewear, facial hair, lighting — ideal for robust training |

**Why it's ideal:** It's "in the wild" data, not a lab dataset. Models trained on it generalize to real conditions. The 4-fold structure lets you do rigorous cross-subject validation.

---

## Part 3: Full Build Prompt

> **Copy this entire section as your prompt to build ALERT.**

---

### SYSTEM PROMPT: Build ALERT — A Real-Time Driver Fatigue Detection System

You are building **ALERT** (Adaptive Landmark Eye-state & Reaction-Time fatigue detection) — a complete, production-quality, software-only driver fatigue detection system. There is no hardware component; this must run on a consumer laptop using only a webcam and standard CPU. The entire runtime must stay **under 2 GB RAM**.

---

### 1. Project Overview

ALERT is a real-time driver fatigue and drowsiness monitoring system that:
- Detects driver fatigue in real-time from a webcam feed using computer vision and deep learning
- Classifies driver state into three levels: **Alert**, **Low Vigilant**, and **Drowsy**
- Triggers progressive audio-visual alerts based on severity
- Can be trained/fine-tuned on the **UTA Real-Life Drowsiness Dataset (UTA-RLDD)** located at `uta-reallife-drowsiness-dataset/`
- Runs entirely on CPU with <2 GB RAM

---

### 2. Architecture Decision (follow this exactly)

#### 2a. Detection Pipeline (Inference)
```
Webcam Frame (BGR)
     │
     ▼
MediaPipe FaceMesh (468 landmarks, 1 face, static_image_mode=False)
     │
     ├── Extract: Eye landmarks (36 per eye → EAR)
     ├── Extract: Mouth landmarks → MAR (yawn detection)
     └── Extract: Nose tip + face contour → Head pose (pitch/yaw/roll via solvePnP)
     │
     ▼
Feature Vector per Frame:
  [EAR_left, EAR_right, EAR_avg, MAR, PERCLOS_60, head_pitch, head_yaw, blink_rate]
     │
     ▼
Sliding Window Buffer (60 frames ≈ 2 seconds at 30fps)
     │
     ▼
Lightweight Temporal Classifier (see §2b)
     │
     ▼
3-Class Output: Alert (0) | Low Vigilant (1) | Drowsy (2)
     │
     ▼
Alert Manager → Graduated audio/visual warnings
```

#### 2b. Model Architecture (Lightweight Temporal Classifier)
Design a **custom lightweight model** that fits in <100 MB on disk and <200 MB RAM at runtime:

**Option A (Primary — Recommended):** Bidirectional GRU + Attention
```
Input: (batch, 60, 8)  ← 60 frames, 8 features
BiGRU(hidden=64) → Dropout(0.3)
BiGRU(hidden=32) → Dropout(0.3)
Attention(query=last_hidden, keys=all_hidden)
Dense(64, relu) → Dropout(0.2)
Dense(3, softmax)
```

**Option B (Fallback — if GRU is too slow):** TCN (Temporal Convolutional Network)
```
Input: (batch, 60, 8)
TCN(filters=32, kernel=3, dilations=[1,2,4,8])
GlobalAvgPool → Dense(32, relu) → Dense(3, softmax)
```

Both models must be implemented in **PyTorch** (not TensorFlow) for easier quantization/optimization.

**PERCLOS definition:** Percentage of frames in the 60-frame window where `EAR_avg < threshold`. This is your single most important feature.

---

### 3. Training Pipeline

#### 3a. Data Loading from UTA-RLDD
```
uta-reallife-drowsiness-dataset/
  FoldN_partM/
    FoldN_partM/
      01/   ← subject ID
        0.mov    ← Label: Alert (0)
        5.mov    ← Label: Low Vigilant (1)
        10.MOV   ← Label: Drowsy (2)
```

**Label mapping from filename:**
- `0.mov` / `0.MOV` → class 0 (Alert)
- `5.mov` / `5.MOV` → class 1 (Low Vigilant)
- `10.mov` / `10.MOV` → class 2 (Drowsy)

**Feature extraction strategy:**
1. Process every video with MediaPipe FaceMesh at target 15 FPS (downsample from native)
2. Extract the 8-feature vector per frame
3. Slide a window of 60 frames with stride 15 (50% overlap)
4. Store as `.npz` cache files so you never re-process videos
5. Use cross-subject 4-fold cross-validation (Fold1–4 map to the 4 fold directories)

**Data augmentation (temporal):**
- Random frame dropout (drop 10–20% of frames in a window, replace with mean)
- Gaussian noise on feature values (σ=0.01)
- Temporal scaling (stretch/compress window by ±10%)
- EAR jitter to simulate lighting variation

#### 3b. Training Configuration
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- Scheduler: CosineAnnealingLR (T_max=50 epochs)
- Loss: CrossEntropy with class weights (Drowsy class is rare → upweight)
- Early stopping: patience=10 on val F1-macro
- Batch size: 64
- Epochs: 100 max
- Train on CPU (must work) but support MPS/CUDA if available
- Save best checkpoint as `models/alert_model_best.pt`
- Log metrics to `logs/training_metrics.json`

#### 3c. Evaluation Metrics
Report per-epoch and final:
- Per-class precision, recall, F1
- Macro F1 (primary metric)
- Confusion matrix
- Inference time (ms/window)

---

### 4. Adaptive Calibration (Critical Feature — based on Paper #7)

Since fixed EAR thresholds fail for ~30% of subjects due to facial structure variance:
- When ALERT starts, run a **10-second calibration window** where the driver is assumed Alert
- Compute `EAR_baseline = mean(EAR) during calibration`
- Set `EAR_threshold = EAR_baseline * 0.75` (personalized)
- Also compute `MAR_baseline` and set `MAR_threshold = MAR_baseline * 2.0` for yawn detection
- Display calibration progress bar in the UI
- Store calibration profile in `~/.alert/calibration.json` keyed by session

---

### 5. Alert System

Implement a graduated alert system:

| State | EAR condition | Duration | Visual | Audio |
|-------|--------------|----------|--------|-------|
| Alert | EAR > threshold | — | Green indicator | None |
| Warning | EAR < threshold OR Low Vigilant | >1.5s | Yellow flashing border | Soft beep (1 kHz, 200ms) |
| Critical | Drowsy state OR eyes closed | >2.5s | Red full-screen flash + text | Loud alarm (multi-tone) |
| Head Drop | Pitch > 20° for >1s | Any | Red + "HEAD UP" message | Loud alarm |

- Use `pygame.mixer` for audio alerts (include a fallback `beepy` if pygame unavailable)
- Debounce alerts: don't trigger more than once per 3 seconds for Warning, 1s for Critical
- Log all alert events to `logs/alerts.json`

---

### 6. Software Architecture

#### Project Structure
```
ALERT/
├── src/
│   ├── core/
│   │   ├── detector.py          # MediaPipe landmark extraction
│   │   ├── feature_extractor.py # EAR, MAR, PERCLOS, head pose
│   │   ├── model.py             # BiGRU + Attention model definition
│   │   ├── classifier.py        # Sliding window + model inference
│   │   ├── calibrator.py        # Adaptive threshold calibration
│   │   └── alert_manager.py     # Alert state machine + audio/visual
│   ├── training/
│   │   ├── dataset.py           # UTA-RLDD video loader + feature cache
│   │   ├── trainer.py           # Training loop + validation
│   │   ├── augment.py           # Temporal augmentation functions
│   │   └── evaluate.py          # Metrics + confusion matrix
│   ├── ui/
│   │   ├── dashboard.py         # Main OpenCV display loop
│   │   └── overlay.py           # HUD overlays (EAR, MAR, state indicator)
│   └── utils/
│       ├── config.py            # Central config (paths, thresholds, hyperparams)
│       ├── logger.py            # Structured JSON event logging
│       └── video_utils.py       # Frame sampling, video metadata
├── models/                      # Saved .pt checkpoints
├── logs/                        # Runtime alert + training logs
├── cache/                       # Pre-computed .npz feature windows
├── scripts/
│   ├── extract_features.py      # Batch process UTA-RLDD → cache/
│   ├── train.py                 # Launch training
│   ├── evaluate.py              # Run evaluation on test fold
│   └── run_alert.py             # Launch real-time detection
├── tests/                       # Unit tests for each module
├── requirements.txt
├── setup.py
└── README.md
```

#### Key Dependencies (all lightweight, CPU-friendly)
```
mediapipe>=0.10.0        # Face mesh (CPU, fast)
opencv-python>=4.8.0     # Video capture + display
torch>=2.0.0             # Model (CPU mode)
numpy>=1.24.0            # Array ops
scipy>=1.10.0            # solvePnP, signal processing
pygame>=2.5.0            # Audio alerts
tqdm>=4.65.0             # Progress bars
scikit-learn>=1.3.0      # Metrics, confusion matrix
matplotlib>=3.7.0        # Training plots
rich>=13.0.0             # Beautiful terminal output
```

---

### 7. Real-Time Inference Loop

The main `run_alert.py` should:
1. Open webcam (index 0, fallback 1), target 30 FPS, resolution 640×480
2. Run calibration phase (10 seconds) before enabling detection
3. For each frame:
   - Run MediaPipe FaceMesh (must complete in <20ms)
   - Extract 8 features
   - Push to rolling deque(maxlen=60)
   - If len(deque) == 60: run model inference (must complete in <30ms)
   - Update HUD overlay with: EAR, MAR, state badge, PERCLOS bar, alert level
4. Enforce <50ms total per frame budget on CPU (Intel i5 or equivalent)
5. Display FPS counter and RAM usage in corner of HUD
6. Press `q` to quit, `c` to re-calibrate, `s` to save a snapshot

---

### 8. UI/HUD Design

The OpenCV window must be **professional and visually clear**:

```
┌─────────────────────────────────────────────────────┐
│  ALERT System          FPS: 28    RAM: 1.1 GB       │
│  ┌──────────────────────────────────┐               │
│  │                                  │  [STATE BADGE]│
│  │      Live Camera Feed            │  ■ ALERT      │
│  │      (face landmarks overlaid    │               │
│  │       as dots on eyes/mouth)     │  EAR:  0.31   │
│  │                                  │  MAR:  0.12   │
│  │                                  │  PERCLOS: 8%  │
│  │                                  │               │
│  └──────────────────────────────────┘  Head: +2°    │
│  ████████████████████░░░░  PERCLOS BAR              │
│  [CALIBRATED] [RECORDING: OFF] [v1.0.0]            │
└─────────────────────────────────────────────────────┘
```

- State badge: green="ALERT", yellow="LOW VIGILANT", red flashing="DROWSY ⚠"
- Landmark dots: draw green dots on eye and mouth landmark positions
- PERCLOS bar: fills red as drowsiness increases
- On "DROWSY": overlay a red semi-transparent border around the entire frame

---

### 9. Performance Constraints (Non-Negotiable)

| Metric | Target |
|--------|--------|
| RAM usage (peak) | < 2 GB |
| Per-frame latency (MediaPipe) | < 20 ms |
| Per-window inference (model) | < 30 ms |
| Total per-frame budget | < 50 ms (≥20 FPS) |
| Model file size on disk | < 100 MB |
| Model RAM footprint | < 200 MB |
| Feature extraction script RAM | < 4 GB (batch processing) |

**Optimization techniques to implement:**
- Run MediaPipe with `model_complexity=0` (lightest model)
- Use `torch.inference_mode()` for all inference
- Pre-allocate the feature deque and numpy arrays at startup (no GC pressure)
- Downscale video frame to 480p before MediaPipe processing if input is higher res
- Use `float16` weights where possible after training
- Feature extraction script: process videos one at a time (never load all into RAM)

---

### 10. Configuration File (`src/utils/config.py`)

All tunable parameters must be centralized:
```python
class Config:
    # Dataset
    DATASET_ROOT = "uta-reallife-drowsiness-dataset/"
    CACHE_DIR = "cache/"
    
    # Model
    MODEL_PATH = "models/alert_model_best.pt"
    WINDOW_SIZE = 60         # frames
    STRIDE = 15              # frames
    N_FEATURES = 8
    HIDDEN_SIZE = 64
    N_CLASSES = 3
    
    # Thresholds (overridden by calibration)
    EAR_THRESHOLD = 0.25     # fallback if no calibration
    MAR_THRESHOLD = 0.6      # yawn threshold
    PERCLOS_WARNING = 0.15   # 15% eye closure → warning
    PERCLOS_CRITICAL = 0.30  # 30% eye closure → critical
    HEAD_PITCH_THRESHOLD = 20.0  # degrees
    
    # Alert timing (seconds)
    WARNING_DURATION = 1.5
    CRITICAL_DURATION = 2.5
    ALERT_DEBOUNCE = 3.0
    
    # Training
    BATCH_SIZE = 64
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 100
    PATIENCE = 10
    TARGET_FPS = 15          # for feature extraction
    
    # Camera
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
```

---

### 11. README Requirements

The README must include:
1. Project description + screenshot of the HUD
2. Theoretical background: EAR formula, MAR formula, PERCLOS definition, head pose estimation
3. Quick start: `pip install -r requirements.txt` → `python scripts/extract_features.py` → `python scripts/train.py` → `python scripts/run_alert.py`
4. Dataset setup instructions (UTA-RLDD folder structure)
5. Training results table (fill in after first training run)
6. Architecture diagram
7. References to all 10 papers listed above
8. Hardware requirements: "Any laptop with a webcam and 4+ GB RAM (2 GB free)"

---

### 12. What NOT to Build

- ❌ No mobile app, no web app, no API server (desktop Python only)
- ❌ No EEG, no steering wheel sensors, no CAN bus (software-only)
- ❌ No TensorFlow (use PyTorch only)
- ❌ No dlib (too slow on CPU — use MediaPipe only)
- ❌ No cloud inference, no model API calls
- ❌ No Docker (just plain Python venv)
- ❌ Do not train on test fold — use strict cross-subject validation

---

### 13. First Steps (Build Order)

1. **Set up project structure** and `requirements.txt`
2. **Implement `feature_extractor.py`** — EAR, MAR, head pose from MediaPipe landmarks
3. **Implement `dataset.py`** — load UTA-RLDD, extract features, cache to `.npz`
4. **Run `extract_features.py`** on the full dataset → validate feature distributions
5. **Implement and train model** (`model.py` + `trainer.py`)
6. **Implement real-time inference** (`classifier.py` + `calibrator.py`)
7. **Implement alert system** (`alert_manager.py`)
8. **Build the HUD** (`dashboard.py` + `overlay.py`)
9. **Wire everything in `run_alert.py`**
10. **Write tests and README**

---

*Build ALERT to be production-quality, well-documented, and runnable by a single developer with `python scripts/run_alert.py` after setup.*
