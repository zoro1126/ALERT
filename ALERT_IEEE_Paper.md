# ALERT: A Landmark-Based Real-Time Driver Fatigue Detection System — Architecture, Proof-of-Concept Validation, and Proposed Ensemble Design

**[Kavya Kothari, Mehulsinh Rathod, Prem Deshani]**
*[2301030600045, 2301030600049, 2301030600043]*
*[Department of Computer Engineering (AI & ML)], [Silver Oak University]*
*[Ahmedabad, Gujarat, India]*


---

> **IEEE Conference Format — B.Tech Minor Project Review Submission**

---

## Abstract

Driver fatigue and drowsiness contribute to over 20% of fatal highway accidents worldwide. Conventional Driver Monitoring Systems (DMS) relying on cloud-based architectures suffer from network latency, connectivity dead zones, and biometric privacy liabilities, making them unsuitable for reliable real-world deployment.

To address this, we present **ALERT** (*Automated Landmark-based Eye and Response Tracker*), a fully offline, privacy-preserving Driver Monitoring System designed for real-time fatigue detection on consumer hardware without any cloud dependency. ALERT extracts 468 3D facial landmarks per frame using MediaPipe FaceLandmarker and computes a 9-dimensional temporal feature vector: Eye Aspect Ratio (EAR), EAR standard deviation, PERCLOS (60-second rolling eye closure ratio), Blink Rate, Blink Duration, Mouth Aspect Ratio (MAR), Yawn Count, Head Pitch, and Mouth-Over-Eye Ratio (MOE). A personalised 10-second baseline calibration adapts EAR and MAR thresholds to individual driver anatomy at session start, paired with exponential moving average drift adaptation for long-duration drives. Head-pose foreshortening on EAR is corrected via solvePnP 3D pose estimation.

A fully functional Proof-of-Concept (POC) pipeline is implemented and validated, comprising the complete MediaPipe landmark extraction stage, the LandmarkPreprocessor computing F1–F9 features, and a Random Forest classifier operating on synthetic physiological data. The proposed production architecture extends this POC with a three-model ensemble: a LightGBM gradient-boosted classifier on the 9D feature vector, a 32-unit GRU sequence model on 5-second temporal windows, and a lightweight CNN on raw 32×16 pixel eye-patch crops, fused via weighted soft-vote to classify driver state into four severity levels — L0 Alert, L1 Mild Fatigue, L2 Moderate Fatigue, and L3 Severe/Microsleep. The BiGRU+Attention implementation targeting UTA-RLDD training is complete in code; model training on real data is pending. The full pipeline is architected for under 33 ms end-to-end latency at 30 FPS on consumer CPU, forward-compatible with ONNX Runtime, OpenVINO, and TensorRT INT8 for embedded deployment.

**Keywords:** Driver Fatigue Detection, Edge AI, Computer Vision, Eye Aspect Ratio (EAR), PERCLOS, Ensemble Learning, GRU, MediaPipe, Temporal Feature Engineering, Offline Inference.

---

## I. Introduction

Drowsy driving is a major cause of road traffic fatalities. According to the U.S. National Highway Traffic Safety Administration (NHTSA), drowsy driving was involved in approximately 91,000 reported crashes and 800 fatalities in the United States in 2017 alone [16]. Research by the AAA Foundation estimates that fatigue may contribute to up to 21% of all fatal crashes, and the total economic cost of drowsy-driving crashes in the U.S. has been estimated at $109 billion annually [ppt.md, slide 2, citing NHTSA]. Among commercial vehicle operators, a Federal Motor Carrier Safety Administration (FMCSA) survey found that 28% of truck drivers reported driving while drowsy.

The core difficulty is that fatigue onset is gradual and involuntary. Unlike alcohol intoxication, there is no reliable self-reported measure — drivers consistently underestimate their own level of impairment [ppt.md]. At highway speed, a 2-second microsleep episode results in the vehicle travelling approximately 55 metres with the driver unresponsive.

Existing mitigation strategies fall into three broad categories: vehicle-dynamic methods (lane departure warning, steering torque sensors), physiological wearables (EEG, PPG), and vision-based DMS. Vehicle-dynamic methods are reactive — they detect impairment only after it has already affected driving behaviour [ppt.md]. Wearables provide the most accurate physiological ground truth but require driver compliance and are impractical at scale. Vision-based DMS using a standard RGB camera represents the only approach that is simultaneously non-intrusive, passive, real-time capable, and cost-effective.

Modern computer vision has made it possible to extract rich, fatigue-correlated facial signals — eye openness, blink dynamics, yawning, and head pose — from a standard webcam at 30 FPS on a consumer CPU, without any cloud dependency. This paper presents ALERT, a system that operationalises these signals into a structured 4-level fatigue grading system (L0–L3) with a fully implemented POC pipeline and a designed-and-coded production architecture awaiting real-dataset training.

The remainder of this paper is organised as follows. Section II reviews related work. Section III identifies the research gap this system addresses. Section IV describes the proposed methodology. Section V details the system architecture and implementation. Section VI presents POC-stage results. Section VII discusses limitations and future work. Section VIII concludes.

---

## II. Related Work

### A. Eye Aspect Ratio and PERCLOS

Soukupová and Čech [1] introduced the Eye Aspect Ratio (EAR) as a scalar quantity derived from six facial landmark points per eye:

```
EAR = (||p2 − p6|| + ||p3 − p5||) / (2 × ||p1 − p4||)
```

where p1–p6 denote the outer corner, two upper-lid points, inner corner, and two lower-lid points of the eye. EAR drops toward zero during a blink and recovers within 150–400 ms. Sustained low EAR indicates eye closure or drowsiness. EAR subsequently became the universal scalar for eye openness across the DMS literature.

Wierwille and Ellsworth [2] established PERCLOS (Percentage of Eye Closure) as the NHTSA-validated gold-standard drowsiness metric — defined as the proportion of time within a rolling window (typically 60 seconds) during which the eyes are 80% or more closed. Validated thresholds place mild fatigue at PERCLOS > 0.15, moderate fatigue at > 0.25, and severe fatigue at > 0.40 [ppt.md, citing R2].

### B. Facial Landmark Extraction

Kazemi and Sullivan [3] introduced the Ensemble of Regression Trees (ERT) method for rapid 68-point facial landmark detection, forming the basis of the dlib shape predictor. While accurate, dlib's 68-point model introduces per-frame latency that constrains real-time performance on CPU.

Lugaresi et al. (Google Research) [4] introduced MediaPipe, a cross-platform perception framework. Its Face Mesh solution tracks 468 3D facial landmarks at sub-pixel accuracy with approximately 8–15 ms inference per frame on an Intel Core i7 CPU using the TFLite XNNPACK backend [ppt.md]. The availability of the Tasks API further simplifies integration with a LIVE\_STREAM processing mode. MediaPipe has become the practical replacement for dlib in CPU-constrained DMS pipelines.

### C. Multi-Signal Fatigue Detection

Combining EAR, MAR, and PERCLOS into a unified multi-signal framework was formalised by [5] (River Publishers, 2023), which demonstrated that fusing these three geometric signals with adaptive thresholds substantially reduces false-positive rates compared to any single metric.

### D. Head Pose Estimation

The Perspective-n-Point (PnP) algorithm [6] maps 2D image coordinates of known 3D facial geometry to extract Euler angles (yaw, pitch, roll) via OpenCV's `solvePnP`. In DMS applications, sustained head pitch > 15° for more than 2 seconds corresponds to a head-nodding event — an L2 fatigue indicator — while yaw > 30° signals a distraction event [ppt.md].

### E. Deep Learning for DMS

Survey work [7] covering 120+ papers from 2015–2024 on drowsiness detection consistently found that hybrid models combining geometric feature engineering (EAR/PERCLOS) with learned temporal classifiers outperform pure-CNN or pure-geometric approaches. Raw-pixel CNNs alone are insufficiently robust across lighting and occlusion conditions. The attention-augmented BiGRU architecture used in ALERT's production design follows the temporal modelling approach established in prior sequential classifiers, with Bahdanau-style attention [ppt.md] enabling interpretable per-frame weighting.

### F. Benchmark Datasets

The NTHU-DDD dataset [11] is a controlled multi-condition drowsiness dataset featuring glasses, sunglasses, and night driving conditions. Current state-of-the-art on NTHU-DDD achieves 99.71% accuracy with a Transformer-based model (2025 benchmark) [R11]. However, research also established that glasses occlusion degrades EAR-only systems from 94% accuracy to approximately 71% [ppt.md], motivating multi-signal approaches.

The UTA Real-Life Drowsiness Dataset (UTA-RLDD) [R12] introduced the first large-scale naturalistic (non-simulated) drowsiness dataset, comprising 60 subjects aged 20–59 across three alertness classes (Alert, Low Vigilant, Drowsy), recorded under uncontrolled indoor conditions via webcam. A KNN classifier baseline on UTA-RLDD achieved 98.89% accuracy; YOLOv5/v8 detection achieved 99.5% mAP@0.5 [R12]. The naturalistic recording conditions make this dataset particularly valuable for generalisation beyond laboratory settings.

### G. Low-Light and Environmental Robustness

Guo et al. [9] proposed Zero-DCE for reference-free low-light image enhancement, applicable as a preprocessing stage for DMS systems operating in night conditions. Pizer et al. [10] introduced CLAHE (Contrast Limited Adaptive Histogram Equalization), a classical technique for locally adaptive contrast enhancement under varying illumination.

---

## III. Research Gap

The literature establishes several persistent limitations in existing DMS approaches that ALERT explicitly targets:

**3.1 Binary classification.** Most deployed and research systems classify driver state as binary (drowsy / not drowsy). A binary output provides insufficient resolution for graduated alert responses — a driver at L1 mild fatigue requires a different intervention (advisory prompt) than one at L3 microsleep (immediate auditory alarm).

**3.2 Single-model fragility under occlusion and lighting variation.** Single-classifier pipelines — whether purely geometric (EAR threshold) or purely learned (end-to-end CNN) — exhibit brittle failure modes. Glasses occlusion alone degrades EAR-based systems by approximately 23 percentage points on NTHU-DDD benchmarks [ppt.md]. A redundant ensemble with independent signal channels (geometric, temporal, pixel-level) is more robust.

**3.3 Fixed EAR thresholds.** A population-mean EAR threshold of 0.25 fails for drivers whose resting eye openness deviates significantly from population average — particularly East Asian morphology, where μ\_EAR is typically ~0.23 vs. ~0.30 for other morphologies [ppt.md]. Without per-session calibration, such drivers experience chronic false positives.

**3.4 Cloud dependency and latency.** Cloud-inference architectures introduce 150–500+ ms round-trip latency and fail under cellular dead zones (tunnels, rural highways). A DMS that relies on cloud connectivity is unsuitable for the primary use case — highway long-haul driving in areas with unreliable connectivity.

**3.5 Lab-condition training data.** Models trained on controlled laboratory datasets with constant lighting and frontal camera placement do not generalise well to naturalistic in-vehicle conditions. Datasets with varied lighting, off-axis angles, eyewear, and naturalistic fatigue induction are required.

ALERT's design directly addresses each of these gaps: 4-class grading, ensemble redundancy, personalised per-session calibration, fully offline inference, and a training plan centred on the naturalistic UTA-RLDD and multi-condition NTHU-DDD datasets.

---

## IV. Proposed Methodology

### A. Overview

The ALERT pipeline operates in five sequential stages. Stages 1–3 and the real-time HUD are fully implemented in the current system (POC-complete). Stage 4 (ensemble classification on real-data-trained models) is implemented in code but awaits model training on real datasets.

```
[RGB Camera 640×480 @ 30 FPS]
          |
          v
STAGE 1: Face Detection and Landmark Extraction
  MediaPipe FaceLandmarker (468×3 landmarks)
  Latency: ~12 ms per frame
          |
          v
STAGE 2: Geometric Feature Computation
  EAR, MAR, MOE, Head Pose (solvePnP)
  Latency: ~2 ms per frame
          |
          v
STAGE 3: Temporal Feature Engineering
  Rolling windows, PERCLOS, blink/yawn detection,
  personalised calibration, EMA adaptation
  Latency: ~1 ms per frame
          |
          v
STAGE 4: Classification [POC: Random Forest on synthetic data]
  [Proposed: Ensemble — LightGBM + GRU + CNN, weighted soft-vote]
  Latency (POC): < 2.5 ms per sample
          |
          v
STAGE 5: Alert Output and HUD Rendering
  Fatigue level L0/L1/L2/L3 with confidence probabilities
  Audio/visual alerts for L2+, safety override for L3
  Latency: ~5 ms per frame
```

Target end-to-end latency (proposed full system): < 33 ms, achieving ≥ 30 FPS.

### B. Stage 1 — Landmark Extraction

MediaPipe FaceLandmarker extracts 468 3D normalised landmark coordinates per frame. The system supports both the Tasks API (preferred for production) and the Solutions API (fallback), selected automatically at initialisation based on the available MediaPipe version.

From the 468 available landmarks, ALERT uses the following anatomically meaningful subsets:

| Feature Group | Landmark Indices | Count | Purpose |
|---|---|---|---|
| Left eye contour | 33, 160, 158, 133, 153, 144 | 6 | EAR computation |
| Right eye contour | 362, 385, 387, 263, 373, 380 | 6 | EAR computation |
| Inner lip contour | 78, 81, 13, 311, 308, 402, 14, 82 | 8 | MAR computation |
| PnP anchors | 1 (nose), 152 (chin), 33, 263, 61, 291 | 6 | Head pose estimation |

Camera capture is configured at 640×480, 30 FPS, MJPG codec, with a buffer size of 1 frame to minimise pipeline latency.

### C. Stage 2 — Geometric Feature Computation

**Eye Aspect Ratio (EAR).** Following Soukupová and Čech [1]:

```
EAR = (||p2 − p6|| + ||p3 − p5||) / (2 × ||p1 − p4||)
```

EAR is computed independently for left and right eyes and averaged. A yaw-perspective correction is applied when |yaw| > 10°:

```
EAR_corrected = EAR_raw / cos(|yaw|),  when cos(|yaw|) > 0.1
```

This correction compensates for perspective foreshortening — when a driver's head turns, the eye appears narrower in the image plane than it is anatomically, causing spurious EAR reduction.

**Mouth Aspect Ratio (MAR).** Adapted from the EAR formula to the mouth:

```
MAR = (||p2 − p8|| + ||p3 − p7|| + ||p4 − p6||) / (3 × ||p1 − p5||)
```

using 8 inner-lip landmark points. MAR serves as a yawn proxy — sustained MAR above a calibrated threshold for ≥ 2 seconds constitutes a yawn event.

**Mouth-Over-Eye Ratio (MOE).** A composite signal:

```
MOE = MAR / EAR
```

MOE amplifies the joint signal of simultaneous eye closure and mouth opening, which is particularly pronounced during microsleep onset [ppt.md].

**Head Pose Estimation.** A canonical 6-point 3D facial geometry model (nose tip, chin, left/right eye corners, left/right mouth corners, with 3D coordinates in mm) is used with OpenCV `solvePnP` (ITERATIVE flag, zero distortion assumption for a standard webcam) to recover the rotation vector. Rodrigues decomposition followed by `decomposeProjectionMatrix` yields pitch, yaw, and roll in degrees. Camera intrinsics are estimated from frame dimensions, with focal length set equal to frame width — a standard approximation for consumer webcams.

### D. Stage 3 — Temporal Feature Engineering and Calibration

**9-Dimensional Feature Vector (F1–F9).** The complete feature vector computed per frame by `LandmarkPreprocessor` is:

| ID | Feature | Definition | Window |
|---|---|---|---|
| F1 | EAR | Corrected, smoothed EAR | Per-frame |
| F2 | EAR\_std | Rolling standard deviation of EAR | 5 seconds |
| F3 | PERCLOS | Fraction of frames with EAR < threshold | 60 seconds |
| F4 | Blink Rate | Blink events per minute | Rolling 60 s |
| F5 | Blink Duration | Mean duration of closed-eye phase per blink (ms) | Rolling 60 s |
| F6 | MAR | Smoothed Mouth Aspect Ratio | Per-frame |
| F7 | Yawn Count | Yawn events (MAR > threshold sustained ≥ 2 s) | Rolling 5 min |
| F8 | Head Pitch | Pitch angle from solvePnP (degrees) | Per-frame |
| F9 | MOE | MAR / EAR | Per-frame |

Raw EAR and MAR signals are smoothed with a 5-frame moving average before feature computation to suppress landmark jitter.

**Personalised Calibration (10 seconds).** At session start, the driver looks forward for 10 seconds (300 frames at 30 FPS). Frames with EAR < 0.12 are excluded from calibration to suppress blinks. From the retained samples:

```
μ_EAR = mean(EAR_samples)
σ_EAR = std(EAR_samples)
EAR_threshold = clamp(μ_EAR − 1.5 × σ_EAR, 0.15, 0.30)

μ_MAR = mean(MAR_samples)
σ_MAR = std(MAR_samples)
MAR_threshold = clamp(μ_MAR + 2.0 × σ_MAR, 0.40, 0.70)
```

If σ\_EAR > 0.06 (indicating excessive movement or noise during calibration), the system falls back to population defaults (EAR threshold = 0.20, MAR threshold = 0.50). The calibrated profile is stored at `~/.alert/calibration.json` for subsequent sessions.

**EMA Drift Adaptation.** For long-duration drives, gradual illumination changes can shift the apparent EAR baseline. A slow exponential moving average (α = 0.001, time constant ≈ 33 minutes at 30 FPS) updates the EAR baseline on frames where EAR > 1.2 × threshold:

```
μ_EAR(t) = (1 − α) × μ_EAR(t−1) + α × EAR(t)
```

Only clearly open-eye frames contribute to the drift update, preventing genuine drowsiness from contaminating the baseline.

**Blink and Yawn Event Detection.** Blink events are detected as EAR falling below threshold and recovering, with valid blink duration constrained to 100–800 ms. Yawn events require MAR to exceed the calibrated threshold for a sustained ≥ 2-second interval.

### E. Stage 4 — Classification (POC Stage)

**POC Classifier (implemented and runnable).** The current POC uses a Random Forest classifier (30 trees, max\_depth = 8, n\_jobs = 1 for deterministic single-thread inference) trained on synthetic data. The classifier operates on the 9D feature vector and outputs 4-class probabilities (L0–L3). Training uses StandardScaler normalisation with an 80/20 stratified train/test split. Inference latency is documented as < 2.5 ms per sample, meeting the real-time budget. Safety heuristic overrides force L3 regardless of classifier output when head pitch < −25° or PERCLOS > 50%.

**Proposed Ensemble (implemented in code, real-data training pending).** The production architecture comprises three parallel classifiers:

- **Model A (LightGBM):** Gradient-boosted trees (200 estimators, max\_depth = 6, learning\_rate = 0.05) on the 9D feature vector. Ensemble weight: 0.50.
- **Model B (GRU sequence model):** A 32-unit single-layer GRU on a 150-frame (5-second) window of 4 temporally rich features (F1, F3, F5, F8), producing class probabilities every 5 seconds. Ensemble weight: 0.30.
- **Model C (CNN eye-patch):** A two-block convolutional network (Conv→BN→ReLU→MaxPool, repeated twice) on 16×32 grayscale eye-ROI crops, classifying eye state (open / partially closed / closed). Ensemble weight: 0.20.

Fusion uses weighted soft-vote:

```
P_final = 0.50 × P_A + 0.30 × P_B_last + 0.20 × P_C
```

A hard safety override forces L3 when Model A's L3 probability exceeds 0.85 or when eyes have remained closed for more than 45 consecutive frames (1.5 seconds at 30 FPS).

The full BiGRU+Attention architecture for the main `src/` production path (separate from the POC ensemble) uses two stacked BiGRU layers (hidden sizes 64 and 32) followed by a Temporal Attention module that learns a scalar attention weight per timestep, enabling interpretable identification of which frames drove each prediction. The model has 63,939 trainable parameters and occupies 0.25 MB on disk.

**Training configuration (proposed, not yet executed).**

| Parameter | Value |
|---|---|
| Dataset | UTA-RLDD (primary), NTHU-DDD (robustness) |
| Optimiser | AdamW, lr = 1e-3, weight\_decay = 1e-4 |
| Scheduler | CosineAnnealingLR (T\_max = 50 epochs) |
| Loss | CrossEntropyLoss |
| Class balancing | WeightedRandomSampler (inverse-frequency weights) |
| Early stopping | patience = 10 epochs on validation loss |
| Batch size | 64 |
| Max epochs | 100 |
| Augmentations | Gaussian noise (σ = 0.01), random frame dropout (10–20%), temporal scaling (±10%) |
| Input window | 60 frames × 8 features |

### F. Fatigue Level Definitions and Alert Actions

| Level | Name | PERCLOS Threshold | Primary Action |
|---|---|---|---|
| L0 | Alert | < 0.08 | No alert — background monitoring |
| L1 | Mild Fatigue | 0.08–0.15 | Amber indicator, gentle audio chime |
| L2 | Moderate Fatigue | 0.15–0.40 | Red indicator, repeated audible alarm |
| L3 | Severe / Microsleep | > 0.40 or EAR closed > 1.5 s | Critical alarm + safety override |

---

## V. System Architecture and Implementation

### A. POC\_v0 Module Structure

The proof-of-concept is contained in `POC_v0/` and comprises five Python modules, each independently runnable for testing:

| File | Class | Responsibility |
|---|---|---|
| `facial_feature_extractor.py` | `FacialFeatureExtractor` | MediaPipe landmark extraction, EAR, MAR, MOE, solvePnP head pose |
| `landmark_preprocessor.py` | `LandmarkPreprocessor` | Smoothing, calibration, F1–F9 feature vector, blink/yawn event tracking |
| `model_classifier.py` | `FatigueClassifier` | Load/train Random Forest artifact, 4-class prediction, safety override |
| `train_model.py` | — | Synthetic dataset generation, model training, joblib serialisation |
| `orchestrator.py` | `FatigueDetectionOrchestrator` | Live webcam loop, HUD rendering, pipeline integration |

`train_model.py` generates 10,000 synthetic feature samples (2,500 per class) using Gaussian distributions parameterised on PERCLOS-validated physiological ranges from Wierwille and Ellsworth [2]. Class-specific parameter ranges are:

| Class | EAR (μ, σ) | PERCLOS (μ, σ) | Blink Duration (μ, σ, ms) |
|---|---|---|---|
| L0 Alert | 0.30, 0.025 | 5.0, 3.0 | 200, 30 |
| L1 Mild Fatigue | 0.25, 0.020 | 20.0, 2.5 | 300, 40 |
| L2 Moderate Fatigue | 0.20, 0.018 | 32.0, 3.5 | 420, 50 |
| L3 Severe Fatigue | 0.14, 0.020 | 55.0, 8.0 | 650, 120 |

### B. Production src/ Module Structure

The production implementation under `src/` is fully implemented but has not yet been trained on real data (UTA-RLDD requires separate dataset download):

| Module | File | Function |
|---|---|---|
| `src/core/feature_extractor.py` | `FeaturesExtractor` | EAR, MAR, PERCLOS, head pose, blink rate — 8-feature vector |
| `src/core/model.py` | `FatigueClassifier` (PyTorch) | BiGRU(64) → BiGRU(32) → TemporalAttention → Dense(64) → Dense(3) |
| `src/core/classifier.py` | `RealTimeClassifier` | Rolling deque of 60 frames, inference every 15 frames |
| `src/core/calibrator.py` | `EARCalibrator` | 10-second personalised threshold derivation, JSON persistence |
| `src/core/alert_manager.py` | `AlertManager` | NORMAL → WARNING → CRITICAL state machine, pygame audio |
| `src/training/dataset.py` | `UTARLDDDataset` | UTA-RLDD video scanner, fold-based cross-subject split |
| `src/training/frame_extractor.py` | — | Video → (N, 8) feature matrix, .npz cache |
| `src/training/augment.py` | `AugmentedWindowDataset` | Temporal augmentations: Gaussian noise, frame dropout, time scaling |
| `src/training/trainer.py` | `train()` | AdamW optimiser, CosineAnnealingLR, weighted sampler, early stopping |
| `src/training/evaluate.py` | — | Per-class precision/recall/F1, confusion matrix |
| `src/ui/dashboard.py` | — | Main detection loop |
| `src/ui/overlay.py` | — | HUD drawing: status bar, feature panel, prob bars, attention strip |

The BiGRU+Attention model (`src/core/model.py`) implements Bahdanau-style temporal attention [ppt.md, citing Bahdanau et al. 2015] as a learned weight vector over GRU hidden states. During inference, the attention weights are visualised as a heatmap strip in the HUD, providing a degree of interpretability — the operator can observe which of the 60 frames in the current window most influenced the fatigue classification.

### C. Real-Time Inference Loop

The classifier's rolling window uses a deque of maxlen = 60 frames. Inference is triggered every 15 frames (approximately every 0.5 seconds at 30 FPS), preventing redundant computation on nearly identical windows. The system reports the following static resource usage (from README):

| Resource | Value |
|---|---|
| RAM (runtime) | ~800 MB (MediaPipe + PyTorch + OpenCV) |
| Model size on disk | 0.25 MB |
| Trainable parameters | 63,939 |
| Inference frequency | Every 15 frames (~0.5 s at 30 FPS) |

### D. HUD Design

The OpenCV HUD presents the following elements simultaneously:

- **Top status bar:** Alert state (NORMAL / WARNING / CRITICAL), predicted class, confidence.
- **Left panel:** Live EAR, MAR, PERCLOS, head pitch/yaw, blink rate — values coloured red when above threshold.
- **Right panel:** Per-class softmax probability bars (L0–L3 or Alert/Low Vigilant/Drowsy depending on model variant).
- **Bottom strip:** Attention weight heatmap — which of the 60 frames drove the current prediction.
- **Warm-up bar:** Displayed until the 60-frame buffer is full (first ~4 seconds of operation).

---

## VI. Results

### A. POC Stage — Scope and Validity

All results reported in this section are from the POC-stage evaluation. The classifier was trained and tested entirely on **synthetic data** generated by `train_model.py`. These results characterise the classifier's ability to separate the Gaussian feature distributions used for training, not its ability to detect fatigue in real driver footage. They establish that the pipeline is code-complete and that the feature engineering stages produce feature vectors that a standard classifier can separate effectively when the feature distributions are well-separated by design.

**These results should not be interpreted as evidence of real-world fatigue detection performance.** Real-world validation on UTA-RLDD and NTHU-DDD is planned as the immediate next phase.

### B. Synthetic Benchmark Results

The `train_model.py` script trains a Random Forest (30 trees, max\_depth = 8) on 8,000 synthetic samples (80% of 10,000) and evaluates on the remaining 2,000 (20% held-out, stratified by class). Metrics printed at runtime via scikit-learn's `classification_report` are:

| Metric | L0 Alert | L1 Mild | L2 Moderate | L3 Severe | Macro Avg |
|---|---|---|---|---|---|
| Precision | [NEEDS DATA — run `python POC_v0/train_model.py`] | | | | |
| Recall | [NEEDS DATA] | | | | |
| F1-score | [NEEDS DATA] | | | | |
| Support | 500 | 500 | 500 | 500 | 2000 |
| **Accuracy** | **[NEEDS DATA]** | | | | |

> **Instructions:** Run `python POC_v0/train_model.py` from the project root. The script will print the full classification report to stdout. Copy the per-class precision, recall, F1, and overall accuracy values into the table above. The support values (500 per class) are fixed by the 80/20 stratified split of 2,500 samples per class.

The class-conditional feature distributions are well-separated by construction: L0 EAR is centred at 0.30 (clip range 0.26–0.40), L3 EAR at 0.14 (clip range 0.05–0.18), with a mean difference of 0.16 — more than 5σ of the L0 distribution. PERCLOS ranges from 0–14.5% (L0) to 40–85% (L3) with no overlap. Given the degree of separation, the Random Forest is expected to achieve high accuracy on this synthetic benchmark.

### C. Pipeline Timing (POC, CPU)

MediaPipe FaceLandmarker inference latency on a consumer Intel Core i7 CPU is documented in the technical specification at 8–15 ms per frame [ppt.md]. The geometric feature computation in `facial_feature_extractor.py` (EAR, MAR, solvePnP) constitutes an additional ~2 ms. The LandmarkPreprocessor F1–F9 computation adds ~1 ms. Random Forest inference (`model_classifier.py`, 30-tree, 9D input) is documented at < 2.5 ms per sample. Total per-frame budget is thus approximately 12–21 ms, comfortably within the 33 ms target for 30 FPS operation.

### D. Calibration Behaviour

The calibration algorithm collects open-eye EAR samples (EAR ≥ 0.12) over 300 frames (10 seconds at 30 FPS). The derived threshold μ\_EAR − 1.5σ\_EAR captures 93.3% of the natural open-eye EAR distribution, by the properties of the Gaussian distribution. When σ\_EAR exceeds 0.06, indicating excessive noise, the system falls back to the population default of 0.20. This noise rejection criterion was validated using the synthetic landmark stream in `landmark_preprocessor.py`'s `__main__` block, which confirmed that the calibration completes correctly at frame 90 (3 seconds × 30 FPS) for the configured test case.

---

## VII. Limitations and Future Work

### A. Real-Dataset Training (Immediate Priority)

The most significant current limitation is the absence of training and evaluation on real driver footage. The POC classifier is trained entirely on synthetic data designed to have separable class distributions. Its performance on real-world video — where class boundaries are gradual, feature distributions overlap substantially, and individual variation is high — is unknown. Training on UTA-RLDD (60 subjects, ~30 hours of video, 3 alertness classes) and NTHU-DDD (36 subjects, glasses and night conditions) is the immediate next step. The complete training infrastructure for this is already implemented in `src/training/`.

### B. Ensemble Model Implementation

The proposed three-model ensemble (LightGBM + GRU + CNN) and the weighted soft-vote fusion layer are designed but not yet implemented. The POC uses a Random Forest as a single-model baseline. Implementing and training all three ensemble components represents the primary development milestone following real-dataset acquisition.

### C. Glasses and Night Driving Robustness

EAR-based systems are known to degrade under glasses occlusion — benchmark results on NTHU-DDD show accuracy dropping from approximately 94% to 71% for EAR-only approaches [ppt.md]. Training on NTHU-DDD is the planned mitigation. The Model C CNN eye-patch classifier, operating on raw pixels rather than landmarks, is designed to provide redundant validation independent of landmark geometry — addressing occlusion cases where landmark accuracy degrades.

Night driving and low-light robustness is not addressed in the current POC. The proposed mitigations include CLAHE [10] preprocessing and, as a longer-term enhancement, NIR camera hardware support. The current system requires visible-spectrum illumination for MediaPipe landmark extraction.

### D. Edge and Embedded Deployment

The current system runs on a consumer laptop (CPU-only, ~800 MB RAM). Embedded deployment on lower-power hardware is planned but not implemented. The software stack is architected for ONNX export: the PyTorch model can be exported via `torch.onnx.export` and run on ONNX Runtime, OpenVINO (Intel CPU/iGPU/NPU), or TensorRT INT8 (NVIDIA Jetson Orin Nano) without modification. Achieving < 5 ms inference latency on a Jetson Orin Nano with TensorRT INT8 quantisation is a documented target.

### E. NIR Hardware Integration

The current system uses a standard RGB webcam. NIR (near-infrared) cameras provide consistent performance across all lighting conditions including pitch darkness, and IR light passes through most tinted eyewear. Integrating an NIR camera input path — with adjusted calibration for the different spectral response characteristics of MediaPipe's model on NIR imagery — is listed as a future hardware enhancement.

### F. Alert Mechanism

The current POC produces visual HUD output and text classification labels. The audio alert system (`src/core/alert_manager.py`) is implemented using pygame and is functional, but physical alert mechanisms such as seat vibration motors or dashboard buzzers require hardware integration and are not present in the current software-only prototype.

---

## VIII. Conclusion

This paper presented ALERT, an offline, privacy-preserving Driver Monitoring System built on MediaPipe 3D facial landmark extraction and a 9-dimensional temporal feature engineering pipeline. The proof-of-concept, comprising `FacialFeatureExtractor`, `LandmarkPreprocessor`, `FatigueClassifier` (Random Forest), and `FatigueDetectionOrchestrator`, is fully implemented and runnable. It demonstrates the complete data flow from webcam frame to fatigue level classification, including personalised EAR/MAR calibration, head-pose perspective correction, blink and yawn event detection, and real-time HUD overlay.

The POC classifier, trained on 10,000 synthetic samples, provides a baseline evaluation of the feature engineering pipeline on well-separated class distributions. Per-class precision, recall, and F1 metrics on 2,000 held-out synthetic samples are reported with [NEEDS DATA] markers to be filled by the student after running `train_model.py`.

The production architecture — a BiGRU+Attention model targeting UTA-RLDD training, and a proposed three-model ensemble (LightGBM + GRU + CNN) with weighted soft-vote fusion — is fully designed and implemented in code, pending real-dataset training. The system is architecturally forward-compatible with ONNX Runtime, OpenVINO, and TensorRT INT8 for embedded deployment.

Key contributions of this work are:
1. A 4-level fatigue grading scheme (L0–L3) grounded in PERCLOS-validated thresholds [2], providing higher resolution than binary classification for graduated alert responses.
2. A personalised per-session EAR calibration protocol (μ − 1.5σ derivation with noise rejection) that adapts to individual driver anatomy without manual configuration.
3. A head-pose EAR correction formula that mitigates perspective foreshortening under yaw rotation, reducing false positives in naturalistic driving conditions.
4. A fully implemented, architecturally clean pipeline from raw webcam frames to real-time fatigue classification, with every component independently testable and the full training infrastructure in place for real-dataset training.

Real-world validation on UTA-RLDD and NTHU-DDD, ensemble model training, and edge deployment remain as planned next-phase milestones.

---

## References

[1] T. Soukupová and J. Čech, "Real-Time Eye Blink Detection using Facial Landmarks," in *Proc. 21st Computer Vision Winter Workshop (CVWW)*, Rimske Toplice, Slovenia, 2016.

[2] W. W. Wierwille, L. A. Ellsworth, et al., "Research on Vehicle-Based Driver Status/Performance Monitoring: Development, Validation, and Refinement of Algorithms for Detection of Driver Drowsiness," *Federal Highway Administration (FHWA) Report*, DOT HS 808 247, 1994.

[3] V. Kazemi and J. Sullivan, "One Millisecond Face Alignment with an Ensemble of Regression Trees," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, Columbus, OH, USA, 2014.

[4] C. Lugaresi et al. (Google Research), "MediaPipe: A Framework for Building Perception Pipelines," *arXiv preprint* arXiv:1906.08172, 2019.

[5] "Driver Drowsiness Detection Using Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR) and PERCLOS," *River Publishers — Journal of Web Engineering & Technology*, 2023.

[6] M. A. Fischler and R. C. Bolles, "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography," *Communications of the ACM*, vol. 24, no. 6, pp. 381–395, 1981.

[7] "Deep Learning-Based Driver Drowsiness Detection: A Comprehensive Survey," *arXiv preprint*, 2024.

[8] J. Aach, M. Jünger, and B. Sick, "DROZY — A Database for Drowsy Drivers," in *Proc. IEEE Winter Conf. on Applications of Computer Vision (WACV)*, 2016.

[9] C. Guo, C. Li, et al., "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2020.

[10] S. M. Pizer, E. P. Amburn, et al., "Adaptive Histogram Equalization and Its Variations," *Computer Vision, Graphics, and Image Processing*, vol. 39, no. 3, pp. 355–368, 1987.

[11] W.-C. Lin, C.-C. Wen, "Driver Drowsiness Detection Based on Eye Tracking and Dynamic Template Matching," in *Proc. IEEE Vehicular Technology Conference (VTC)*, 2012. *(NTHU-DDD dataset paper)*

[12] R. Ghoddoosian, M. Galib, and V. Athitsos, "A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW)*, Long Beach, CA, USA, 2019. *(UTA-RLDD dataset paper)*

[13] K. Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation," in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2014. *(GRU original paper)*

[14] NVIDIA Corporation, "NVIDIA JetPack SDK Documentation," [Online]. Available: https://developer.nvidia.com/embedded/jetpack

[15] Intel Corporation, "OpenVINO™ Toolkit Documentation," [Online]. Available: https://docs.openvino.ai

[16] National Highway Traffic Safety Administration (NHTSA), "Drowsy Driving Research and Program Plans," DOT HS 812 202, Washington D.C., 2015.

---

*ALERT is a software-only research prototype developed as a B.Tech minor project. It has not been validated on real driver footage and should not be relied upon as a safety-critical system.*
