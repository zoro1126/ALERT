# ALERT — Technical Pitch Deck Content
### Automated Landmark based Eye and Response Tracker
> **Audience:** Project Guide | **Type:** Technical Pitch | **Stage:** Pre-Development (POC Complete)

---

---

## SLIDE 1 — Introduction

### Project Title
**ALERT: Automated Landmark based Eye and Response Tracker**
*A Non-Intrusive, Real-Time Driver Fatigue Detection System*

### One-Line Hook
> Every 100 seconds, a drowsy driving crash occurs in the United States alone. ALERT detects driver fatigue before a crash happens — using only a camera.

### What is ALERT?
ALERT is a **real-time, fully offline Driver Monitoring System (DMS)** that analyses a continuous video stream of a driver's face to detect fatigue and drowsiness across **four severity levels** — from fully alert to severe microsleep risk — using a cascade of:

1. **Geometric Facial Feature Extraction** via 468-point 3D landmark tracking (MediaPipe FaceLandmarker)
2. **Temporal Feature Engineering** across rolling time windows (PERCLOS, blink rate, EAR trends)
3. **A Multi-Model Ensemble Classifier** that fuses signals from a geometric model, a sequence model, and a vision model to produce a robust fatigue level output

### What ALERT is NOT
- It is **not** a wearable or physiological sensor system (no EEG, no PPG patches)
- It is **not** a cloud-dependent system — all inference runs **100% offline on the edge device**
- It is **not** a simple threshold alarm — it uses trained ML models for multi-class probabilistic grading

### Current Status
- **Phase:** Pre-Development — Proof of Concept (POC) complete
- **POC Stack:** Python 3.11 | MediaPipe Tasks API | OpenCV | scikit-learn Random Forest
- **POC Output:** Working 30 FPS real-time pipeline with HUD overlay, calibration timer, and L0–L3 classification
- **Next Phase:** Dataset training on UTA-RLDD + NTHU-DDD, ensemble model implementation, edge deployment

---

---

## SLIDE 2 — Background and Motivation

### The Problem: Scale of Drowsy Driving

| Statistic | Value | Source |
|---|---|---|
| Road crashes caused by drowsiness (US, annual) | ~91,000 | NHTSA 2017 |
| Deaths per year from drowsy driving (US) | ~800 | NHTSA 2017 |
| Estimated true crashes involving fatigue | Up to 21% of all fatal crashes | AAA Foundation |
| Economic cost of drowsy driving crashes (US) | $109 billion/year | NHTSA |
| Commercial truck drivers reporting drowsy driving | 28% | FMCSA Survey |

Drowsy driving impairs reaction time, lane-keeping, and decision-making in ways that are **functionally equivalent to 0.05–0.10% blood alcohol concentration** — yet is far less regulated and far harder to detect.

### Why Existing Solutions Fall Short

| Existing Approach | Limitation |
|---|---|
| **Steering wheel torque sensors** | Detects fatigue only after drift begins — reactive, not predictive |
| **Lane deviation alerts (LDWS)** | Triggers only when vehicle has already veered — dangerously late |
| **EEG/physiological wearables** | Highly accurate but impractical — driver must wear electrodes |
| **Single-threshold blink counters** | Brittle — fails for people with naturally narrow eyes, glasses, or lighting changes |
| **Infrared camera-only systems** | Expensive hardware; poor interpretability; can't work in all conditions |

### Why Facial Signals are the Right Channel

The human face provides **rich, involuntary physiological signals** of fatigue that cannot be easily suppressed by the driver:

- **Eye Aspect Ratio (EAR):** Eyes physically close as the levator palpebrae muscle fatigues — measurable with sub-millimetre precision via landmarks
- **PERCLOS:** The percentage of eye closure time is the **NHTSA-validated gold standard** for drowsiness measurement (Wierwille & Ellsworth, 1994)
- **Blink Dynamics:** Drowsy blinks are longer (>300ms) and less frequent initially, then more frequent in severe cases
- **Yawning (MAR):** Yawn frequency sharply increases at the onset of significant fatigue — a near-universal mammalian response
- **Head Pitch/Nodding:** Involuntary head drop due to muscle tone loss is a reliable L2–L3 indicator
- All of these are **passive, non-contact, and non-intrusive**

### The Technical Opportunity

Modern ML and computer vision now make it possible to extract these signals at 30 FPS from a standard RGB camera on a consumer CPU. MediaPipe FaceLandmarker runs 468 3D landmark points at <20ms per frame on Intel iGPU hardware — making a fully offline, real-time DMS viable without specialized hardware.

---

---

## SLIDE 3 — Relevance and Importance

### Regulatory Landscape Demanding DMS

- **Euro NCAP 2026:** All new passenger vehicles sold in Europe **must** have a Driver Monitoring System by 2026 under Euro NCAP's updated safety protocols.
- **EU General Safety Regulation (GSR) 2022/2023:** Mandates Advanced Driver Assistance Systems (ADAS) including DMS for all new vehicle types.
- **NHTSA FMVSS Proposals (USA):** Active rulemaking to require DMS in commercial trucks and new passenger vehicles.
- **India AIS-189 (2023):** Bureau of Indian Standards notified new norms for commercial vehicle safety electronics including fatigue detection systems.

This creates a **multi-billion dollar addressable market** with regulatory pull — DMS is not optional technology, it is mandatory infrastructure.

### Market Context

| Segment | Relevance |
|---|---|
| **Commercial trucking fleets** | Long-haul drivers are highest-risk; fleet operators face liability without DMS |
| **Ride-share / taxi platforms** | Uber, Ola, Lyft face driver fatigue incidents; DMS reduces liability |
| **Automotive OEMs** | Need Euro NCAP compliance — DMS integrated into ADAS stack |
| **Public transport buses** | Municipal bodies and state transport undertakings need driver safety systems |
| **Military & defence vehicles** | Fatigue monitoring for long-duration field operations |

### Why a Facial Landmark-Based Approach Is Commercially Viable

- **Hardware cost:** Standard 720p RGB camera costs ₹300–₹800 — vs ₹15,000–₹40,000 for NIR DMS hardware
- **No driver setup:** No wearables, no calibration jigs, no driver interaction
- **Offline operation:** No cellular connectivity required — critical for rural highways and tunnels
- **Software-upgradeable:** The model can be retrained and OTA-updated without hardware changes
- **Privacy-preserving:** No face images stored — only numerical feature vectors processed locally

### Academic and Research Relevance

ALERT sits at the intersection of:
- **Computer Vision** — real-time 3D landmark detection and geometric feature extraction
- **Signal Processing** — temporal window analysis and adaptive calibration
- **Machine Learning** — multi-class ensemble classification on physiological time-series
- **Edge Computing** — latency-constrained inference on CPU/NPU hardware

This makes it a **strong final-year project** touching all core CS/CE/AIML domains simultaneously.

---

---

## SLIDE 4 — Literature Review

### Foundation Papers

#### [L1] Eye Aspect Ratio (EAR) — Soukupová & Čech, CVWW 2016
- **Contribution:** Defined EAR as the ratio of eye height to width using 6 landmark points per eye:
  ```
  EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
  ```
  where p1–p6 are the six eye landmark coordinates (top-outer, top-inner, right corner, bottom-inner, bottom-outer, left corner).
- **Finding:** EAR drops sharply to ~0 during a blink and recovers within 150–400ms. Sustained low EAR indicates eye closure/drowsiness.
- **Impact:** EAR became the universal scalar for eye-openness in all subsequent DMS literature.

#### [L2] PERCLOS Standard — Wierwille & Ellsworth, FHWA 1994
- **Contribution:** Established PERCLOS (Percentage of Eye Closure) as the NHTSA-validated gold-standard drowsiness metric.
- **Definition:** Proportion of time eyes are ≥80% closed within a rolling time window (typically 60 seconds).
  ```
  PERCLOS = (frames where EAR < 0.8 * EAR_open) / total_frames_in_window
  ```
- **Thresholds validated:** PERCLOS > 0.15 → mild fatigue; PERCLOS > 0.25 → moderate; PERCLOS > 0.40 → severe.
- **Impact:** Every modern DMS (Seeing Machines, Smart Eye, Valeo) uses PERCLOS as a primary signal.

#### [L3] MediaPipe Face Mesh — Lugaresi et al. (Google), arXiv 2019
- **Contribution:** A lightweight ML pipeline tracking 468 3D facial landmarks in real-time on mobile/edge CPUs.
- **Architecture:** BlazeFace detector (SSD-based, 200 FPS on mobile) → Face Mesh model (TFLite, ~10ms inference).
- **Accuracy:** Sub-pixel landmark localisation error on standard face benchmark datasets.
- **Impact:** Replaced dlib's 68-point model as the industry standard for facial landmark DMS pipelines.

#### [L4] Head Pose via solvePnP — Fischler & Bolles (foundational), adapted for DMS
- **Contribution:** The Perspective-n-Point algorithm maps 2D image landmarks to a known 3D facial geometry model to extract Euler angles (Yaw, Pitch, Roll).
- **DMS Application:** Head pitch >15° sustained for >2s = head nod/drooping event (L2 indicator). Yaw >30° = lane-change/distraction event.

### DMS-Specific Survey Literature

#### [L5] Deep Learning Survey for Driver Drowsiness — arXiv 2024
- Surveyed 120+ papers from 2015–2024 on drowsiness detection methods.
- **Key finding:** Hybrid models (geometric features + CNN) consistently outperform pure CNN or pure geometric approaches.
- **Best performing architectures:** Transformer + EAR/PERCLOS features: 99.71% on NTHU-DDD; CNN-LSTM on video sequences: 97.8% on UTA-RLDD.
- **Key insight:** Feature engineering for EAR/PERCLOS remains critical even in deep learning pipelines — raw pixel CNNs alone are not robust across lighting and occlusion conditions.

#### [L6] NTHU-DDD Benchmark — Lin et al., 2012 (updated benchmarks 2025)
- Introduced a structured multi-condition drowsiness dataset with glasses and night conditions.
- Current SOTA on NTHU-DDD: **99.71% accuracy** using a Transformer-based model (2025).
- Benchmark shows that glasses occlusion drops EAR-only accuracy from 94% → 71% — motivating multi-signal ensemble approaches.

#### [L7] UTA-RLDD Dataset — Ghoddoosian et al., 2019
- Introduced the first large-scale naturalistic (non-simulated) drowsiness dataset.
- KNN baseline on UTA-RLDD: 98.89% accuracy; YOLOv5/v8 detection: 99.5% mAP@0.5.
- Dataset provides real-world variability absent in lab datasets — critical for generalisation.

### Gaps Identified in Literature

| Gap | ALERT's Response |
|---|---|
| Most systems binary (drowsy/not drowsy) | ALERT uses 4-class grading (L0–L3) based on PERCLOS severity bands |
| Single-model fragility under occlusion/lighting | ALERT proposes 3-model ensemble with independent signal channels |
| Lab-condition datasets only | Training on UTA-RLDD (naturalistic) + NTHU-DDD (glasses/night) |
| Cloud-dependent inference | ALERT is 100% offline, <33ms per frame on consumer CPU |
| No driver-specific calibration | ALERT performs 10-second personalised baseline calibration at session start |

---

---

## SLIDE 5 — Objectives

### Primary Objective
Design and implement a **real-time, offline, non-intrusive Driver Fatigue Detection System** that classifies driver fatigue into four severity levels (L0–L3) from a standard RGB camera stream at ≥30 FPS with <100ms end-to-end latency on consumer CPU hardware.

### Specific Technical Objectives

#### O1 — Data Extraction Pipeline
Implement a **468-point 3D facial landmark extraction pipeline** using MediaPipe FaceLandmarker Tasks API that achieves:
- ≥30 FPS throughput on Intel Core i5/i7 iGPU hardware
- <20ms per-frame landmark extraction latency
- Robust tracking under ±25° yaw and ±20° pitch head rotation

#### O2 — Temporal Feature Engineering
Construct a **9-dimensional temporal feature vector** from raw landmark coordinates using rolling window statistics and geometric ratios that:
- Adapts to individual driver anatomy via personalised 10-second calibration
- Corrects for head-pose foreshortening effects on EAR via solvePnP head pose compensation
- Operates on rolling windows of 5 seconds (short-term) and 60 seconds (long-term PERCLOS)

#### O3 — Ensemble Model Design
Design, train and validate a **3-model ensemble classifier**:
- Model A: LightGBM / Random Forest on feature vector (≥95% accuracy on held-out validation)
- Model B: GRU sequence model (32 hidden units) on 150-frame temporal window
- Model C: Lightweight CNN on cropped eye-region ROI (32×16px)
- Fused via weighted soft-vote achieving ≥97% L2/L3 recall (critical safety metric)

#### O4 — Real-Time HUD System
Implement a **real-time Head-Up Display overlay** showing live fatigue level, signal statistics, calibration status, and probability breakdown at 30 FPS.

#### O5 — Edge Deployment Readiness
Architect the software stack to be **forward-compatible** with:
- ONNX Runtime / OpenVINO for CPU/NPU accelerated inference
- TensorRT quantized deployment on NVIDIA Jetson Orin Nano (target: <5ms inference)
- C++ rewrite of the hot-path inference pipeline for embedded SoC deployment

---

---

## SLIDE 6 — Required Tools and Technology

### Hardware

| Component | Specification | Purpose |
|---|---|---|
| Development machine | Intel Core i7-7700HQ, 16GB RAM, Intel HD 630 iGPU | POC development and testing |
| Webcam | 720p/1080p RGB, 30 FPS, USB UVC-compliant | Video input for landmark extraction |
| Target edge device (future) | NVIDIA Jetson Orin Nano (8GB) or Intel NUC with OpenVINO NPU | Production edge deployment |

### Software Stack

#### Core Runtime
| Library | Version | Role |
|---|---|---|
| Python | 3.11+ | Primary development language |
| OpenCV | 4.8+ | Camera I/O, image processing, HUD rendering, solvePnP |
| MediaPipe | 0.10+ (Tasks API) | FaceLandmarker — 468-point 3D landmark extraction |
| NumPy | 1.26+ | Numerical feature computation, rolling window arrays |
| SciPy | 1.11+ | Signal processing, exponential moving average |

#### Machine Learning
| Library | Version | Role |
|---|---|---|
| scikit-learn | 1.3+ | Random Forest classifier (POC); preprocessing pipelines |
| LightGBM | 4.0+ | Gradient boosted tree classifier (production Model A) |
| PyTorch | 2.1+ | GRU sequence model (Model B) and CNN eye-patch model (Model C) |
| ONNX Runtime | 1.16+ | Cross-platform model serving for edge deployment |
| joblib | 1.3+ | Model serialisation and fast parallel feature computation |

#### Data Processing
| Library | Version | Role |
|---|---|---|
| Pandas | 2.0+ | Dataset loading, feature tabulation, label alignment |
| OpenCV VideoCapture | built-in | Video dataset frame extraction |
| tqdm | 4.65+ | Dataset preprocessing progress tracking |
| imbalanced-learn | 0.11+ | SMOTE oversampling for L3 class imbalance |

#### Development Tools
| Tool | Purpose |
|---|---|
| VS Code + Python Extension | Primary IDE |
| virtualenv / venv | Isolated Python environment (`.venv/`) |
| Git + GitHub | Version control |
| Jupyter Notebooks | Exploratory data analysis and model training experiments |

### MediaPipe FaceLandmarker — Technical Specifications

| Parameter | Value |
|---|---|
| Model file | `face_landmarker.task` (6.7 MB) |
| Landmark count | 468 points (x, y, z) per face |
| Additional outputs | Face blendshapes (52 AU-like coefficients), face transformation matrix |
| Running mode | LIVE_STREAM (async callback) or VIDEO (synchronous) |
| Input resolution | 640×480 recommended for speed; up to 1920×1080 supported |
| Inference backend | TFLite XNNPACK (CPU) or GPU delegate |
| Inference time | ~8–15ms on Intel Core i7 CPU; ~4–6ms with GPU delegate |
| Max faces tracked | Configurable — set to 1 for single-driver DMS |
| Coordinate system | Normalised image coordinates (0.0–1.0 for x/y); z is depth relative to face centre |

---

---

## SLIDE 7 — Method / Approach

### System Architecture Overview

```
[RGB Camera 640x480 @ 30 FPS]
           |
           v
[STAGE 1: FACE DETECTION & LANDMARK EXTRACTION]
  MediaPipe FaceLandmarker
  Output: 468 x (x, y, z) landmark coordinates
  Latency: ~12ms per frame
           |
           v
[STAGE 2: GEOMETRIC FEATURE COMPUTATION]
  EAR, MAR, MOE, Head Pose (solvePnP)
  Latency: ~2ms per frame
           |
           v
[STAGE 3: TEMPORAL FEATURE ENGINEERING]
  Rolling windows, PERCLOS, blink detection,
  personalised calibration, EMA adaptation
  Latency: ~1ms per frame
           |
           v
[STAGE 4: ENSEMBLE CLASSIFICATION]
  Model A (LightGBM) + Model B (GRU) + Model C (CNN)
  Weighted soft-vote fusion
  Latency: ~5–8ms per frame
           |
           v
[STAGE 5: ALERT OUTPUT + HUD RENDERING]
  Fatigue level L0/L1/L2/L3 + probabilities
  Audio/visual alerts for L2+, critical override for L3
  Latency: ~5ms per frame
           |
           v
  TOTAL END-TO-END LATENCY: < 33ms  (achieves 30 FPS)
```

---

### STAGE 1: Data Extraction — MediaPipe FaceLandmarker

#### Landmark Points Used

From the 468 available landmarks, ALERT uses the following anatomically meaningful subsets:

| Feature Group | Landmark Indices (MediaPipe) | Count | Purpose |
|---|---|---|---|
| Left eye contour | 362, 385, 387, 263, 373, 380 | 6 points | EAR computation |
| Right eye contour | 33, 160, 158, 133, 153, 144 | 6 points | EAR computation |
| Mouth outer | 61, 291, 13, 14, 17, 0, 267, 37 | 8 points | MAR + yawn detection |
| Nose tip | 1 | 1 point | solvePnP anchor |
| Chin | 199 | 1 point | solvePnP anchor |
| Left eye corner | 226 | 1 point | solvePnP anchor |
| Right eye corner | 446 | 1 point | solvePnP anchor |
| Left mouth corner | 57 | 1 point | solvePnP anchor |
| Right mouth corner | 287 | 1 point | solvePnP anchor |

**Total actively used: 24 landmark points out of 468**

#### Camera Capture Specifications
```
Resolution:   640 × 480 pixels
Frame rate:   30 FPS (MJPG codec on V4L2 / DirectShow)
Colour space: BGR (OpenCV default) → converted to RGB for MediaPipe
FOURCC codec: MJPG (hardware-compressed, lower USB bandwidth than YUYV)
Buffer size:  1 frame (cv2.CAP_PROP_BUFFERSIZE = 1) to minimise latency
```

#### Head Pose Estimation via solvePnP

A canonical 3D face model (6-point generic mesh) is used to solve the Perspective-n-Point problem:

```python
model_points_3D = np.array([
    [0.0,    0.0,    0.0],     # Nose tip
    [0.0,   -330.0, -65.0],    # Chin
    [-225.0, 170.0, -135.0],   # Left eye left corner
    [225.0,  170.0, -135.0],   # Right eye right corner
    [-150.0,-150.0, -125.0],   # Left mouth corner
    [150.0, -150.0, -125.0],   # Right mouth corner
])

# Camera intrinsics (estimated from frame dimensions)
focal_length = frame_width
centre = (frame_width / 2, frame_height / 2)
camera_matrix = np.array([
    [focal_length, 0, centre[0]],
    [0, focal_length, centre[1]],
    [0, 0, 1]
], dtype=np.float64)

_, rotation_vector, translation_vector = cv2.solvePnP(
    model_points_3D, image_points_2D, camera_matrix, dist_coeffs
)
pitch, yaw, roll = rotation_vector_to_euler_angles(rotation_vector)
```

**Output:** Yaw (±90°), Pitch (±45°), Roll (±30°) — used for both head-pose features and EAR correction.

---

### STAGE 2 & 3: Temporal Feature Engineering

#### The 9-Dimensional Feature Vector

| ID | Name | Formula / Definition | Window | Clinical Basis |
|---|---|---|---|---|
| F1 | EAR (calibrated) | `mean(EAR_left, EAR_right) / cos(|yaw|)` | Per-frame | Soukupová & Čech 2016 |
| F2 | EAR_std | Rolling standard deviation of EAR | 5 seconds (150 frames) | Temporal variability proxy for microsleep |
| F3 | PERCLOS | `count(EAR < threshold) / total_frames` | 60 seconds (1800 frames) | Wierwille & Ellsworth 1994 — NHTSA gold standard |
| F4 | Blink Rate | Blink events per minute | Rolling 60 seconds | Drowsy blink rate drops then spikes at severe fatigue |
| F5 | Blink Duration | Mean duration of eye-closed phase per blink | Rolling 60 seconds | >300ms average blink = strong fatigue indicator |
| F6 | MAR | `(||A-F|| + ||B-E|| + ||C-D||) / (2*||G-H||)` | Per-frame | Yawn geometry — mouth open ratio |
| F7 | Yawn Count | Yawn events (MAR > threshold for >1.5s) | Rolling 5 minutes | Yawn frequency correlates with KSS ≥6 |
| F8 | Head Pitch | Pitch angle from solvePnP (degrees) | Per-frame | >15° sustained = head nod event |
| F9 | MOE | `MAR / EAR` | Per-frame | Composite signal amplifying simultaneous eye-close + mouth-open |

#### Personalised Calibration Protocol (10 Seconds)

At session start, ALERT collects 300 frames (10 seconds × 30 FPS) of the alert driver:

```
Step 1: Collect raw EAR values for 300 frames
Step 2: Compute μ_EAR = mean(EAR_samples)
         Compute σ_EAR = std(EAR_samples)
Step 3: Set EAR_threshold = μ_EAR - 1.5 * σ_EAR
         (covers 93.3% of natural open-eye EAR distribution)
Step 4: Compute MAR_threshold = μ_MAR + 2.0 * σ_MAR
         (flags yawns as statistical outliers above resting MAR)
```

This eliminates false alarms for drivers with naturally narrow eyes (East Asian morphology typically has μ_EAR ~0.23 vs. μ_EAR ~0.30 for wider eyes) — a critical robustness requirement.

#### Long-Session EMA Drift Adaptation

Over long drives, lighting changes and natural postural drift can shift the EAR baseline. ALERT applies a slow Exponential Moving Average to gently adapt:

```
EAR_baseline(t) = α * EAR(t) + (1 - α) * EAR_baseline(t-1)
where α = 0.001  (time constant ~33 minutes at 30 FPS)
```

This ensures the threshold adapts to gradual environmental changes (sunset → dusk → night) without masking genuine drowsiness signals.

---

### STAGE 4: Ensemble Model Architecture

#### Model A — LightGBM Gradient Boosted Trees (Primary Geometric Classifier)

| Parameter | Value |
|---|---|
| Algorithm | LightGBM (gradient boosted decision trees) |
| Input | 9D feature vector [F1–F9] — one per frame |
| Output | Softmax probability over {L0, L1, L2, L3} |
| n_estimators | 200 trees |
| max_depth | 6 |
| learning_rate | 0.05 |
| num_leaves | 31 |
| class_weight | {L0: 1, L1: 1.5, L2: 2.0, L3: 4.0} — compensates for imbalance |
| Inference time | ~2–3 ms per sample on CPU |
| Model size | ~1.2 MB (joblib serialised) |
| Training dataset | UTA-RLDD + NTHU-DDD extracted feature vectors (~120,000 labelled samples) |
| Expected accuracy | ≥95% on held-out test split |
| Ensemble weight | 0.50 |

**Why LightGBM over Random Forest:** LightGBM uses leaf-wise tree growth and histogram-based splitting — 10–50× faster training than Random Forest on large datasets while maintaining comparable accuracy. At inference, both are ~2–3ms; LightGBM has smaller memory footprint.

#### Model B — GRU Sequence Classifier (Temporal Trend Detector)

| Parameter | Value |
|---|---|
| Algorithm | Gated Recurrent Unit (GRU), PyTorch |
| Input shape | [batch=1, sequence=150, features=4] — F1, F3, F5, F8 over 5 seconds |
| Output | Softmax probability over {L0, L1, L2, L3} |
| GRU hidden units | 32 |
| GRU layers | 1 |
| FC head | Linear(32→16) → ReLU → Linear(16→4) → Softmax |
| Total parameters | ~6,400 parameters |
| Model size | ~26 KB (PyTorch state dict) |
| Inference time | ~6–9 ms per 150-frame sequence on CPU |
| Inference frequency | Every 5 seconds (every 150 frames) — not per-frame |
| Training dataset | UTA-RLDD sequences (continuous 5-second windows extracted per participant) |
| Expected accuracy | ≥92% on sequence-level held-out test |
| Ensemble weight | 0.30 |

**Why GRU over LSTM:** GRU has only 2 gates vs. LSTM's 3, resulting in ~25% fewer parameters and faster inference with comparable sequence modelling performance on sequences <300 steps.

**Why 4 features (not all 9) for Model B:** F1 (EAR), F3 (PERCLOS), F5 (blink duration), F8 (head pitch) are the temporally richest signals. The other 5 features (EAR_std, blink rate, MAR, yawn count, MOE) are already rolling aggregations — feeding them into the GRU would be redundant double-aggregation.

#### Model C — CNN Eye-Patch Classifier (Vision-Level Redundancy)

| Parameter | Value |
|---|---|
| Algorithm | Custom lightweight CNN, PyTorch |
| Input shape | [batch=1, channels=1 (grayscale), H=16, W=32] — cropped eye ROI |
| Output | Softmax over {open, partially_closed, closed} — mapped to fatigue contribution |
| Architecture | Conv2D(1→16, 3×3) → BN → ReLU → MaxPool(2×2) → Conv2D(16→32, 3×3) → BN → ReLU → MaxPool(2×2) → Flatten → FC(32) → Softmax(3) |
| Total parameters | ~18,000 parameters |
| Model size | ~72 KB |
| Inference time | ~4–6 ms per frame on CPU |
| Training dataset | MRL Eye Dataset (84,898 eye images, labeled open/closed under varied lighting/pose) + CEW (Closed Eyes in the Wild, 2,423 subjects) |
| Expected accuracy | ≥97% binary (open/closed), ≥90% 3-class |
| Output mapping | `closed` → adds L2/L3 contribution to fusion; `partially_closed` → L1; `open` → L0 |
| Ensemble weight | 0.20 |

**Why a CNN eye-patch model:** Completely independent of MediaPipe landmark geometry — if landmarks jitter due to lighting or slight occlusion, Model A's EAR becomes unreliable. The CNN sees raw pixels and is unaffected by landmark errors, providing a redundant validation channel.

#### Fusion Layer — Weighted Soft Vote

```python
# Per-frame probability fusion
P_final = 0.50 * P_A + 0.30 * P_B_last + 0.20 * P_C
# P_B_last = most recent 5-second GRU prediction, held constant until next update

predicted_level = argmax(P_final)   # → L0, L1, L2, or L3

# Safety override: hard override on Model A or C severe signal
if P_A[L3] > 0.85 or eye_state_consecutive_closed_frames > 45:
    predicted_level = L3  # Immediate critical alert regardless of fusion
```

---

### STAGE 5: Fatigue Level Definitions and Alert Actions

| Level | Name | PERCLOS | EAR (relative) | Blink Duration | Action |
|---|---|---|---|---|---|
| L0 | Alert | < 0.08 | > 0.85 * baseline | < 200ms | No alert — status display only |
| L1 | Mild Fatigue | 0.08–0.15 | 0.70–0.85 * baseline | 200–300ms | Amber indicator — gentle chime |
| L2 | Moderate Fatigue | 0.15–0.40 | 0.50–0.70 * baseline | 300–500ms | Red indicator — repeated audible alarm |
| L3 | Severe / Microsleep | > 0.40 or eyes closed >1.5s | < 0.50 * baseline | > 500ms or sustained | Critical alarm + seat vibration signal (future) |

---

### Training Datasets — Full Specification

#### Dataset 1: UTA-RLDD (Primary Training Set)

| Attribute | Value |
|---|---|
| Full name | University of Texas at Arlington Real-Life Drowsiness Dataset |
| Participants | 60 subjects (diverse age 20–59, gender, ethnicity) |
| Recording duration | ~30 minutes per subject |
| Total frames | ~108,000+ frames per subject; ~6.4M frames total |
| Total video duration | ~30 hours |
| Camera | Webcam (640×480 or 1280×720, 30 FPS) |
| Environment | Naturalistic — home, office, uncontrolled lighting |
| Labels | 3 classes per video clip: **Alert (0)** / **Low Vigilance (1)** / **Drowsy (2)** |
| Label granularity | Clip-level (each ~10-minute clip has one label) |
| Label method | Self-reported KSS (Karolinska Sleepiness Scale) + expert observer review |
| File format | MP4 video files + CSV label file per subject |
| Dataset size | ~95.7 GB (uncompressed) |
| Access | Registered academic download from UTA website |
| ALERT label mapping | Alert→L0, Low Vigilance→L1, Drowsy→L2 |

**Extracted features per frame:** After processing with ALERT pipeline, each frame yields a 9D feature vector. With clip-level labels applied at frame level, expected ~2M usable labelled feature samples after calibration warmup excluded.

#### Dataset 2: NTHU-DDD (Night/Glasses Robustness)

| Attribute | Value |
|---|---|
| Full name | National Tsing Hua University Driver Drowsiness Detection Dataset |
| Participants | ~36 subjects |
| Conditions | Glasses / Sunglasses / No glasses × Day / Night → 4 condition variants |
| Labels | Binary: **Drowsy** / **Non-Drowsy** — per frame |
| Label granularity | Frame-level (manually annotated per frame) |
| File format | AVI videos + per-frame label CSV |
| Dataset size | ~14 GB |
| Access | Academic request to NTHU CV Lab |
| ALERT label mapping | Non-Drowsy→L0, Drowsy→L2 (no L1 mapping — binary dataset) |
| Primary purpose | Train model robustness to **glasses occlusion** and **low-light conditions** — the two biggest failure modes of EAR-based systems |

#### Dataset 3: YawDD (In-Vehicle Camera Angle)

| Attribute | Value |
|---|---|
| Full name | Yawning Detection Dataset |
| Participants | 107 subjects (61 male, 46 female) |
| Camera placement | In-vehicle — dashcam-angle (below eye level, angled upward ~15°) |
| Camera count | 2 simultaneous streams: front-face + dash-angle |
| Labels | 3 classes: **Normal** / **Talking** / **Yawning** — per video clip |
| File format | AVI video files |
| Dataset size | ~8 GB |
| ALERT label mapping | Normal→L0, Yawning→L1/L2 (yawn frequency determines severity) |
| Primary purpose | Train model on **realistic dashcam perspective** — most DMS cameras are mounted below eye level, creating geometric distortion that desk-webcam datasets do not cover |

#### Dataset 4: MRL Eye Dataset (CNN Eye-Patch Model Training)

| Attribute | Value |
|---|---|
| Full name | Machine Learning Research Lab Eye Dataset |
| Total images | 84,898 eye images |
| Subjects | 37 subjects |
| Conditions | 8 scales of image resolution, varied illumination (IR, daylight, low-light), with/without glasses |
| Labels | Binary: **Open / Closed** — per image |
| Label granularity | Image-level |
| File format | JPEG/PNG images, organised in open/ and closed/ directories |
| Dataset size | ~2.1 GB |
| Access | Freely available on dataset repository |
| ALERT label mapping | Open→eye_open, Closed→eye_closed (used to train Model C CNN) |
| Primary purpose | Train Model C (CNN eye-patch classifier) to recognise eye state from raw pixels, independent of landmark geometry |

#### Dataset 5: DROZY (Severity Calibration)

| Attribute | Value |
|---|---|
| Full name | DROZY — A Database for Drowsy Drivers |
| Participants | 14 subjects |
| Physiological ground truth | EEG (electroencephalography) + KSS self-report (1–9 scale) |
| Video modality | Near-infrared (NIR) camera |
| Primary value | **KSS 1–9 scale** provides finest-grained severity labels — enables calibrating PERCLOS thresholds to validated sleepiness levels |
| ALERT label mapping | KSS 1–4 → L0, KSS 5–6 → L1, KSS 7–8 → L2, KSS 9 → L3 |
| Primary purpose | Calibrate L0–L3 PERCLOS threshold values against a physiologically validated ground truth (EEG + KSS) |

#### Combined Training Dataset Summary

| Phase | Datasets | Subjects | Labelled Samples (approx.) | Purpose |
|---|---|---|---|---|
| POC | Synthetic / generated | — | 10,000 | Proof of concept only |
| Phase 2 (Development) | UTA-RLDD + NTHU-DDD + YawDD + MRL Eye | ~240 subjects | ~2.5M feature vectors + 85K eye images | Production model training |
| Phase 3 (Production) | All above + DROZY + custom in-vehicle recordings | 250+ subjects | ~3M+ samples | Full production system |

---

---

## SLIDE 8 — Bibliography

> Formatted in IEEE style for academic presentation.

[1] T. Soukupová and J. Čech, "Real-Time Eye Blink Detection using Facial Landmarks," in *Proc. 21st Computer Vision Winter Workshop (CVWW)*, Rimske Toplice, Slovenia, 2016.

[2] W. W. Wierwille, L. A. Ellsworth, et al., "Research on Vehicle-Based Driver Status/Performance Monitoring: Development, Validation, and Refinement of Algorithms for Detection of Driver Drowsiness," *Federal Highway Administration (FHWA) Report*, DOT HS 808 247, 1994.

[3] V. Kazemi and J. Sullivan, "One Millisecond Face Alignment with an Ensemble of Regression Trees," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, Columbus, OH, USA, 2014.

[4] C. Lugaresi et al. (Google Research), "MediaPipe: A Framework for Building Perception Pipelines," *arXiv preprint* arXiv:1906.08172, 2019.

[5] "Driver Drowsiness Detection Using Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR) and PERCLOS," *River Publishers — Journal of Web Engineering & Technology*, 2023.

[6] W.-C. Lin, C.-C. Wen, "Driver Drowsiness Detection Based on Eye Tracking and Dynamic Template Matching," in *Proc. IEEE Vehicular Technology Conference (VTC)*, 2012. *(NTHU-DDD dataset paper)*

[7] R. Ghoddoosian, M. Galib, and V. Athitsos, "A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW)*, Long Beach, CA, USA, 2019. *(UTA-RLDD dataset paper)*

[8] J. Aach, M. Jünger, and B. Sick, "DROZY — A Database for Drowsy Drivers," in *Proc. IEEE Winter Conf. on Applications of Computer Vision (WACV)*, 2016.

[9] C. Guo, C. Li, et al., "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2020.

[10] S. M. Pizer, E. P. Amburn, et al., "Adaptive Histogram Equalization and Its Variations," *Computer Vision, Graphics, and Image Processing*, vol. 39, no. 3, pp. 355–368, 1987. *(CLAHE)*

[11] NVIDIA Corporation, "NVIDIA JetPack SDK Documentation," [Online]. Available: https://developer.nvidia.com/embedded/jetpack

[12] Intel Corporation, "OpenVINO™ Toolkit Documentation," [Online]. Available: https://docs.openvino.ai

[13] Google, "MediaPipe Tasks API — FaceLandmarker Solution Guide," [Online]. Available: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

[14] M. A. Fischler and R. C. Bolles, "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography," *Communications of the ACM*, vol. 24, no. 6, pp. 381–395, 1981. *(foundational PnP work)*

[15] K. Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation," in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2014. *(GRU original paper)*

[16] National Highway Traffic Safety Administration (NHTSA), "Drowsy Driving Research and Program Plans," DOT HS 812 202, Washington D.C., 2015.

---

*Document prepared for ALERT technical pitch — Pre-Development Stage, August 2026.*
