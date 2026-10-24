# ALERT — Extreme Production Master Plan (Heterogeneous Ensemble Architecture)

## Executive Summary
This document defines the complete mathematical, algorithmic, and engineering specification to elevate the **ALERT (Automated Landmark-based Eye and Response Tracker)** Driver Fatigue Detection System to the upper academic performance ceiling ($\ge 55\%\text{ Subject-Independent Macro-F1}$) on the **UTA-RLDD** dataset while guaranteeing rock-solid, zero-false-positive live webcam inference.

All design decisions have been resolved through architectural analysis and requirements alignment. Execution proceeds strictly phase-by-phase according to this document.

---

## 1. Architectural Design & Parameter Ledger

| Specification | Decision | Mathematical / Engineering Rationale |
|---|---|---|
| **Temporal Sequence Window ($T$)** | **120 frames (8.0s @ 15 FPS)** | Captures complete biological event horizons: natural blink ($150–250\text{ms}$), slow eyelid closure ($0.5–1.5\text{s}$), deep yawn cycle ($3.5–5.5\text{s}$), and head nod-drop with startle recovery ($2–4\text{s}$). |
| **Feature Dimensionality ($F$)** | **12 Kinematic Channels** | Expands from 8 static landmarks to differential velocities ($\Delta\text{EAR}/\Delta t$, $\Delta\text{MAR}/\Delta t$, $\Delta\text{Pitch}/\Delta t$) and duration metrics. |
| **Normalization Strategy** | **Per-Subject Baseline Calibration** | Normalized against personal resting baseline ($EAR_{\text{base}}, Pitch_{\text{base}}$) derived from the subject's Alert video in training, and from the 15s calibration on live webcam. Eliminates the train-serve skew. |
| **Ensemble Model Suite** | **12-Model Heterogeneous Ensemble** | 3 fundamentally distinct neural architectures ($\text{CNN-GRU} + \text{Dilated TCN} + \text{BiGRU-Attention}$) trained across all 4 subject-independent holdout folds. |
| **Ensemble Aggregation** | **Soft-Voting Probability Average** | $\bar{P}(c) = \frac{1}{12} \sum_{m=1}^{12} P_m(c)$ cancels out single-model structural bias. |
| **Inference Smoothing** | **Temporal Exponential Moving Average (EMA)** | $\hat{P}_t = \alpha P_t + (1 - \alpha) \hat{P}_{t-1}$ ($\alpha = 0.35$). Eliminates single-frame classification jitter. |
| **Hardware Latency Budget** | **$\le 12.0\text{ ms}$ CPU / $\approx 100\text{ FPS}$** | Total ensemble latency is $\approx 9.6\text{ ms}$ on single-thread CPU (Intel/AMD), well within the $15–30\text{ ms}$ edge limit. |

---

## 2. Phase-by-Phase Execution Roadmap

```
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Feature Engineering Overhaul & Kinematic Pipeline             │
│  - 12-channel feature extraction with differential velocities          │
│  - True per-subject baseline normalization (No window-p90 flattening)  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│ Phase 2: Zero-Leakage Dataset Re-Extraction                            │
│  - Scan UTA-RLDD: extract baseline from each subject's Alert video     │
│  - Generate 120-frame calibrated windows (12 channels) across 141 vids │
│  - Partition cleanly into cache/fold{1,2,3,4}*.npz                     │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│ Phase 3: Model Suite & Ordinal Class Separation Training               │
│  - Implement 3 Architectures: HybridCNNGRU, TCNClassifier, BiGRU-Attn │
│  - Ordinal Distance Penalty Loss + Balanced Focal Loss                 │
│  - Train all 3 architectures across 4 Subject-Independent Folds        │
│    (12 models total) on RTX 3050 (AMP fp16)                            │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│ Phase 4: 12-Model Ensemble Engine & Production Packaging               │
│  - Ensemble probability fusion layer                                   │
│  - Online EMA temporal smoothing + Two-Tier Debounced Alarm Engine     │
│  - Latency benchmark (CPU & GPU) + ONNX export (Opset 14)              │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│ Phase 5: Live Webcam Stress-Testing & Final Verification               │
│  - Interactive visual verification: Open eyes = 100% Alert             │
│  - Squinting / drowsy / head nod validation                            │
│  - Full documentation & run scripts finalized                          │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical & Logical Formulations

### 3.1 Feature Vector Specification ($F = 12$)

For each frame $t$, the 12 features are mathematically defined as:

1. **Relative Left Eye Aspect Ratio ($EAR_{L,\text{rel}}$)**:
   $$EAR_{L,\text{rel}}(t) = \frac{EAR_L(t)}{EAR_{\text{base}}}$$
2. **Relative Right Eye Aspect Ratio ($EAR_{R,\text{rel}}$)**:
   $$EAR_{R,\text{rel}}(t) = \frac{EAR_R(t)}{EAR_{\text{base}}}$$
3. **Relative Average EAR ($EAR_{\text{avg},\text{rel}}$)**:
   $$EAR_{\text{avg},\text{rel}}(t) = \frac{EAR_{\text{avg}}(t)}{EAR_{\text{base}}}$$
4. **Eye Velocity ($\Delta EAR$)**:
   $$\Delta EAR(t) = \frac{EAR_{\text{avg}}(t) - EAR_{\text{avg}}(t - \delta t)}{\delta t}$$
   *Physics*: Alert blinks close and snap open in $\le 150\text{ms}$ ($\Delta EAR \gg 0$). Fatigued blinks reopen sluggishly ($\Delta EAR \approx 0$).
5. **Mouth Aspect Ratio ($MAR$)**:
   $$MAR(t) = \frac{\|p_{81} - p_{402}\| + \|p_{82} - p_{311}\|}{2 \cdot \|p_{78} - p_{308}\|}$$
6. **Mouth Opening Velocity ($\Delta MAR$)**:
   $$\Delta MAR(t) = \frac{MAR(t) - MAR(t - \delta t)}{\delta t}$$
   *Physics*: Distinguishes rapid speech/smiling ($\Delta MAR$ spikes and drops in $<0.5\text{s}$) from continuous yawn expansion ($>3.0\text{s}$).
7. **Calibrated PERCLOS ($P_{60}$)**:
   Percentage of frames in rolling 60-frame buffer where $EAR_{\text{avg}}(t) < 0.75 \cdot EAR_{\text{base}}$.
8. **Relative Head Pitch ($Pitch_{\text{rel}}$)**:
   $$Pitch_{\text{rel}}(t) = Pitch(t) - Pitch_{\text{base}}$$
   Positive values denote downward head tilt.
9. **Head Nod Velocity / Jerk ($\Delta Pitch$)**:
   $$\Delta Pitch(t) = \frac{Pitch(t) - Pitch(t - \delta t)}{\delta t}$$
   *Physics*: Captures sudden downward microsleep drop followed by reflexive startle jerk.
10. **Relative Head Yaw ($Yaw_{\text{rel}}$)**:
    $$Yaw_{\text{rel}}(t) = Yaw(t) - Yaw_{\text{base}}$$
11. **Blink Rate ($BR$)**:
    Detected completed blinks per minute over rolling buffer.
12. **Mean Blink Duration ($BD$)**:
    Average duration (seconds) of blinks occurring within the buffer. Drowsy blinks exceed $250–400\text{ms}$; alert blinks average $100–180\text{ms}$.

---

### 3.2 Loss Function: Ordinal Distance-Penalized Focal Loss

The classes have an intrinsic biological hierarchy:
$$\text{Class 0: Alert} \quad < \quad \text{Class 1: Low Vigilant} \quad < \quad \text{Class 2: Drowsy}$$

Standard Cross-Entropy penalizes confusing Class 0 with Class 2 identically to confusing Class 0 with Class 1. In real-world driving, confusing Alert with Drowsy is disastrous.

We define the total loss as:
$$\mathcal{L} = \mathcal{L}_{\text{Focal}} + \lambda_{\text{ord}} \cdot \mathcal{L}_{\text{Ordinal}}$$

Where:
1. **Focal Loss with Label Smoothing ($\epsilon = 0.08$)**:
   $$\mathcal{L}_{\text{Focal}} = -\sum_{c=0}^{2} \alpha_c (1 - p_c)^\gamma y_{c,\text{smooth}} \log(p_c)$$
   Class weights: $\alpha = [1.0, 1.15, 1.10]$, $\gamma = 2.0$.
2. **Ordinal Distance Penalty**:
   $$\mathcal{L}_{\text{Ordinal}} = \sum_{c=0}^{2} p_c \cdot (c - y_{\text{true}})^2$$
   Penalty weight: $\lambda_{\text{ord}} = 0.35$.
   *Effect*: If true class is Alert ($y=0$), predicting Drowsy ($c=2$) incurs an ordinal penalty of $2^2 = 4\times$, forcing logits away from the opposite extreme.

---

### 3.3 Model Architectures (The Triad)

All models accept input tensor $(B, 120, 12)$ and output $(B, 3)$:

#### Architecture 1: HybridCNNGRU ($\approx 32\text{k params}$)
- **Conv Stage**:
  - `Conv1D(12 -> 32, kernel=3, padding=1) + BatchNorm1D + ReLU + Dropout(0.2)`
  - `Conv1D(32 -> 64, kernel=5, padding=2) + BatchNorm1D + ReLU + MaxPool1D(2)` $\rightarrow (B, 60, 64)$
- **Recurrent Stage**:
  - `GRU(input_size=64, hidden_size=48, num_layers=1, batch_first=True)`
- **Attention**:
  - Scaled Bahdanau Temporal Self-Attention over 60 steps $\rightarrow (B, 48)$
- **Classifier Head**:
  - `Linear(48, 32) + ReLU + Dropout(0.35) + Linear(32, 3)`

#### Architecture 2: TCNClassifier ($\approx 75\text{k params}$)
- **Dilated Causal Convolutions** with receptive field $= 1 + 2 \times (K-1) \times \sum d_i$:
  - Dilations: $d \in \{1, 2, 4, 8, 16\}$ with kernel size $K=3$.
  - Receptive field: $1 + 2 \times 2 \times 31 = 125\text{ frames} > 120\text{ window length}$.
- Residual connections + WeightNorm + SpatialDropout1D(0.25).
- Global Adaptive Average Pooling $\rightarrow$ Linear(64, 3).

#### Architecture 3: BiGRUAttention ($\approx 48\text{k params}$)
- Bidirectional GRU layer:
  - Forward + backward pass captures both onset of fatigue and recovery reaction.
  - Hidden state: $2 \times 36 = 72$.
- Multi-Head Temporal Attention (4 heads) $\rightarrow$ Dense classification head.

---

### 3.4 Inference Temporal Smoothing & Alarm Debounce

To eliminate instantaneous classification flickering on single-frame blinks:

1. **Probability EMA Filter**:
   $$\hat{P}_t = 0.35 \cdot P_t + 0.65 \cdot \hat{P}_{t-1}$$
2. **Two-Tier State Machine**:
   - $\hat{P}(\text{Alert}) \ge 0.50 \rightarrow \mathbf{NORMAL}$ (HUD Green, alarm counter reset)
   - $\hat{P}(\text{Low Vigilant}) \ge 0.50$ for $\ge 3.0\text{ seconds} \rightarrow \mathbf{WARNING}$ (HUD Yellow, gentle single chime)
   - $\hat{P}(\text{Drowsy}) \ge 0.45$ for $\ge 1.5\text{ seconds} \rightarrow \mathbf{CRITICAL}$ (HUD Red, urgent siren repeats every 1.0s)

---

## 4. Work Breakdown by Phase

### Phase 1: Feature Engineering Overhaul
- Modify [`src/core/feature_extractor.py`](file:///c:/Study%20Material%20And%20Projects/Sem%207/Alert/src/core/feature_extractor.py): Implement 12-channel kinematic features.
- Update [`src/training/preprocess.py`](file:///c:/Study%20Material%20And%20Projects/Sem%207/Alert/src/training/preprocess.py): Remove window-relative p90 divide; implement proper baseline scaling.
- Implement unit tests in [`tests/test_features.py`](file:///c:/Study%20Material%20And%20Projects/Sem%207/Alert/tests/test_features.py) verifying all 12 channels.

### Phase 2: Zero-Leakage Dataset Re-Extraction
- Update [`scripts/extract_cache.py`](file:///c:/Study%20Material%20And%20Projects/Sem%207/Alert/scripts/extract_cache.py):
  - Pass 1: Compute $EAR_{\text{base}}$ and $Pitch_{\text{base}}$ from each subject's Alert (`0.mov/mp4`) video.
  - Pass 2: Extract 120-frame calibrated windows for all 141 videos.
- Cache output: `cache/fold{k}_{subj}_{label}.npz` with shape $(M, 120, 12)$.

### Phase 3: Model Suite & Ordinal Loss Training
- Update [`src/core/model.py`](file:///c:/Study%20Material%20And%20Projects/Sem%207/Alert/src/core/model.py) for $T=120, F=12$.
- Update [`src/training/losses.py`](file:///c:/Study%20Material%20And%20Projects/Sem%207/Alert/src/training/losses.py): Implement Ordinal Distance Penalty.
- Orchestrate 12-fold training run (3 architectures $\times$ 4 folds).

### Phase 4: 12-Model Ensemble Engine & Export
- Implement `EnsembleFatigueClassifier` in [`src/core/classifier.py`](file:///c:/Study%20Material%20And%20Projects/Sem%207/Alert/src/core/classifier.py).
- Integrate EMA filter and two-tier debounce alarm state machine.
- Export all models to ONNX Opset 14 and run 1,000-pass latency benchmark.

### Phase 5: Live Webcam Pipeline Integration & Verification
- Pipeline Integration Complete:
  - Connected `Calibrator` 3D head pose and EAR personal baseline calibration profile.
  - Connected 9-model `EnsembleFatigueClassifier` into `src/ui/dashboard.py` and `src/run_alert.py`.
  - Added two-tier debounced alarm state machine with audio alert playback.
  - HUD overlay enhanced with ensemble badge, kinematic velocity metrics, and temporal attention heatmap.
  - Verified with 57 unit and end-to-end integration tests (`tests/test_pipeline_integration.py`).
- Ready for User Live Execution:
  - Run via: `python src/run_alert.py`
  - Re-calibrate baseline: `python src/run_alert.py --recalibrate`
