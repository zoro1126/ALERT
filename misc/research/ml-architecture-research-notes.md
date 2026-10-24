# ALERT: ML Architecture Research Notes

> **Topic:** Sequential Models, Ensemble Architectures & Dataset Strategy  
> **Author:** Research Notes for ALERT (Automated Landmark based Eye and Response Tracker)  
> **Date:** August 2026  

---

## Q1. Why Not Use a Sequential Model (RNN / LSTM) for Temporal Facial Signals?

### The Core Theoretical Argument *For* Sequential Models

Driver fatigue manifests as a **temporal process**, not an instantaneous event. A single EAR value at frame $t$ tells you very little on its own — a person can blink and have the same EAR as someone microsleeping. What actually matters is the **pattern of change over time**:

- Blink frequency *trends* over 5–10 minutes (slow blink rate decrease → fatigue onset).
- PERCLOS is *inherently* a rolling temporal window metric.
- Head pitch drooping follows a slow **oscillatory gravity-response pattern** unique to drowsiness.
- The *variability* of EAR (EAR std dev) only makes sense as a time-series quantity.

This makes the problem a textbook use case for **sequence modelling**. Formally:

$$\hat{y}_t = f(x_t, x_{t-1}, x_{t-2}, \ldots, x_{t-T})$$

where $x_t = [F_1^t, F_2^t, \ldots, F_9^t]$ is the 9-dimensional feature vector at frame $t$, and $T$ is the lookback window. An LSTM, GRU, or Transformer encoder is purpose-built for this type of mapping.

---

### Why Classical Models Dominate in Practice (For Edge DMS)

Despite the theoretical elegance of RNNs, the **overwhelming majority of production-grade DMS systems** still use classical ML (Random Forests, XGBoost, SVMs) or lightweight CNNs. Here is why:

#### 1. **The Feature Vector Already Encodes History**

The feature engineering step in ALERT pre-computes temporal statistics *into* each feature vector:

| Feature | Encodes Temporal History |
|---|---|
| $F_2$ (EAR std dev) | Rolling 5-second variance |
| $F_3$ (PERCLOS) | Rolling 60-second eye closure ratio |
| $F_4$ (Blink Rate) | Events per rolling minute |
| $F_5$ (Blink Duration) | Per-blink historical average |
| $F_7$ (Yawn Count) | Events in 5-minute window |

When you hand-engineer these sliding window statistics into the feature vector, **the classifier's input already contains the relevant temporal context**. A Random Forest operating on this feature vector is implicitly doing temporal reasoning — just through pre-aggregated statistics rather than learned recurrent weights. This is why the RF still works surprisingly well despite being "memoryless."

#### 2. **Sequential Models Require Significant Data**

LSTMs and Transformers are **sample-hungry**. They need thousands of continuous time-series sequences (not just individual frame labels) to learn meaningful temporal patterns. Most fatigue datasets — including UTA-RLDD — provide **clip-level labels**, not **frame-level labels**, making it hard to construct clean, long sequence training samples.

A Random Forest trained on pre-computed feature vectors, on the other hand, converges well with a few thousand labelled samples.

#### 3. **Latency: The Sequential Model Problem**

For real-time edge inference at 30 FPS:

| Model | Single-Sample Inference | Runs at 30 FPS on CPU? |
|---|---|---|
| Random Forest (30 trees) | ~2.8 ms | YES |
| GRU (64 hidden units) | ~8–15 ms | Borderline |
| LSTM (128 hidden units) | ~15–25 ms | NO |
| Transformer (4 heads, 2 layers) | ~30–60 ms | NO |
| XGBoost (100 trees, n_jobs=-1) | ~78 ms (thread overhead) | NO |

The 30 FPS constraint leaves only **~33 ms** per frame. After MediaPipe landmark extraction (~18 ms), pre-processing (~3 ms), and rendering (~5 ms), only **~7 ms** is available for the classifier. This practically rules out full LSTM/Transformer models on CPU-only hardware.

> **However:** GRUs with small hidden dimensions (32–64 units) CAN fit within the 7 ms budget, especially when quantized to INT8 via ONNX Runtime or OpenVINO.

#### 4. **Temporal Context Window Already Captured by Feature Engineering**

The PERCLOS window is 60 seconds × 30 fps = **1800 frames**. An LSTM trying to maintain hidden state over 1800 timesteps would suffer severe gradient vanishing — even LSTMs typically have trouble with sequences longer than 200–300 steps without attention mechanisms. The rolling window statistics circumvent this problem entirely.

---

### Recommended Architecture Stance: Hybrid Two-Tier Approach

The right answer is not "either/or." The best approach for ALERT is a **two-tier temporal architecture**:

```
TIER 1: Frame-Level (Fast Path, ~5 ms)
  Random Forest / LightGBM on F1–F9 feature vector
  → Immediate per-frame alert (L0–L3)
         |
         | Feeds per-frame predictions + raw features
         v
TIER 2: Sequence-Level (Slow Path, every 5 seconds)
  GRU (32 hidden units) on last 150 frames of F1-F9
  → Smoothed trend classification, early L1 detection
```

- **Tier 1** runs every frame for real-time immediate response.
- **Tier 2** runs every 5 seconds to catch gradual fatigue trends that the instantaneous classifier misses (e.g., slow EAR decline over 2 minutes).
- The final output is the **higher severity** prediction from either tier.

> This is conceptually similar to the two-path architecture used in commercial DMS solutions by Seeing Machines (AU) and Smart Eye (SE).

---

## Q2. Ensemble of Models for Robustness and Generalisation

Using a single model is always a vulnerability. An ensemble averages out each model's individual failure modes. For a fatigue detection DMS, the ideal ensemble combines models that are **independently strong on different signal channels**.

### Proposed Ensemble Architecture for ALERT

```
MODEL A                MODEL B                MODEL C
LightGBM /         GRU Sequence            CNN Eye-Patch
Random Forest       Classifier              Classifier

Input:              Input:                  Input:
F1–F9 feature       150-frame               32x16px cropped
vector              time-series of          eye-region ROI
                    F1, F3, F5, F8          (grayscale)
     |                   |                      |
     +-------------------+----------------------+
                         |
                  FUSION LAYER
                  Weighted Soft Vote
                  (w_A=0.5, w_B=0.3, w_C=0.2)
                         |
                  Final Fatigue Level
                  L0 / L1 / L2 / L3
```

### Description of Each Model in the Ensemble

#### Model A: LightGBM / Random Forest (Primary Geometric Classifier)
- **Input:** Pre-computed 9D feature vector [F1 ... F9] at each frame.
- **Why:** Fast (~2–3 ms), interpretable, robust to noisy individual features via bagging. Handles EAR/MAR/PERCLOS feature space extremely well.
- **Strength:** Frame-by-frame responsiveness. Excellent at catching **acute** events (sudden eye closure, yawn).
- **Weakness:** No memory of historical drift; relies entirely on feature engineering to approximate temporal context.
- **Weight in ensemble:** 0.5 (primary signal).

#### Model B: GRU Sequence Classifier (Temporal Trend Detector)
- **Input:** Sliding window of **150 consecutive frames** (5 seconds at 30 FPS) of features [F1, F3, F5, F8] — EAR, PERCLOS, blink duration, and head pitch (the most temporally meaningful signals).
- **Architecture:** 1-layer GRU with 32 hidden units → FC(16) → Softmax(4).
- **Why:** GRU is faster than LSTM (no cell state) and generalises better on short sequences. A 32-unit GRU inference takes ~6–9 ms on CPU, fitting our latency budget when run at 6 FPS (every 5th frame) rather than 30 FPS.
- **Strength:** Catches **gradual drowsiness onset** — slow EAR decline over 2–3 minutes that appears "normal" at any single frame but is clearly trending downward over the sequence.
- **Weakness:** Needs more labelled sequence data; requires stateful inference (maintains hidden state between windows).
- **Weight in ensemble:** 0.3 (secondary temporal signal).

#### Model C: Lightweight CNN Eye-Patch Classifier (Vision-Level Redundancy)
- **Input:** Cropped and normalised **eye-region ROI** (~32×16px grayscale) extracted from the raw frame using MediaPipe landmark bounding box.
- **Architecture:** MobileNetV3-Small fine-tuned for open/closed/partial eye states, or a custom tiny CNN:
  ```
  Conv2D(16, 3x3) → ReLU → MaxPool
  Conv2D(32, 3x3) → ReLU → MaxPool
  Flatten → FC(32) → Softmax(3)  [open / partially closed / closed]
  ```
- **Why:** Operates directly on pixel data, completely independent of landmark accuracy. Provides **redundancy against landmark jitter and occlusion**, where Model A's EAR would spike erroneously.
- **Strength:** Detects actual eye state from texture/appearance, not geometry. Works even when landmark tracking degrades (e.g., sunglasses, head turn >25°).
- **Weakness:** Needs large annotated eye-image dataset; adds ~5–8 ms latency. Eye state (open/closed) is binary, not a full fatigue score — must be converted into a contribution score.
- **Weight in ensemble:** 0.2 (vision-level validation).

### Fusion Strategy

**Weighted Soft-Vote** on class probability vectors is preferred over hard voting:

```
P_final(k) = w_A * P_A(k) + w_B * P_B(k) + w_C * P_C(k)
```

where k ∈ {L0, L1, L2, L3} and sum(w_i) = 1.

Weights are calibrated empirically on a held-out validation set. An alternative is **stacking** — training a small logistic regression meta-learner on the three models' probability outputs, which automatically learns optimal weights per condition.

### Additional Ensemble Members to Consider (Future)

| Model | Signal Channel | Use Case |
|---|---|---|
| Head Pose CNN | 3-DOF head orientation | Detect nodding / head drop without relying on landmark geometry |
| Action Unit (AU) Detector | Facial muscle micro-expressions | Detect subtle AU45 (blink) and AU26 (jaw drop) via FACS |
| Physiological Proxy (HR) | Head-motion micro-tremor for rPPG | Non-contact heart rate estimation from skin colour fluctuation |
| Gaze Estimator | Eye gaze vector | Detect unfocused / fixed stare patterns |

---

## Q3. Is UTA-RLDD Enough? Dataset Strategy

### What UTA-RLDD Provides

UTA-RLDD (University of Texas at Arlington — Real-Life Drowsiness Dataset) is one of the most valuable open datasets for our use case:

- **60 participants** (diverse age, gender, ethnicity)
- **Naturalistic webcam conditions** — uncontrolled lighting, varied backgrounds, natural head movement
- **Three states:** Alert, Low Vigilance, Drowsy (maps naturally to L0/L1/L2)
- **~30-minute recordings** per participant (~108,000+ frames)
- **Captured in real home/office environments**, not a simulator lab

For the POC, UTA-RLDD is a solid starting point. **But it is not sufficient** for a production-grade, robust DMS for the following reasons:

---

### Identified Gaps in UTA-RLDD

| Dimension | UTA-RLDD | Gap / Risk |
|---|---|---|
| Lighting conditions | Mostly daylight indoor | No night driving, tunnel, flickering light, sunlight glare |
| Glasses/sunglasses | Rarely present | EAR-based models fail with most glasses; no occlusion training data |
| Severe fatigue (L3 / microsleep) | Not well represented | Hard to capture naturalistically — ethical constraints |
| In-vehicle mounting angle | Webcam from front-facing desk | Real DMS cameras are mounted at dash level, angled upward — different perspective |
| Ethnicity / eye shape diversity | Moderate | East Asian eye shapes (narrower EAR baseline) need explicit coverage |
| Dynamic head movement | Low | Drivers move heads far more — mirror checks, lane changes, etc. |
| Infrared / low-light imaging | None | NIR camera data critical for night-time capability |

---

### Recommended Multi-Dataset Strategy

To build a robust, generalising model, ALERT should train on a **union of multiple complementary datasets**:

#### Tier 1: Primary Datasets (Use Together)

| Dataset | Participants | Conditions | What It Adds |
|---|---|---|---|
| UTA-RLDD | 60 | Naturalistic webcam, 3 labels | Real-life base; good EAR/PERCLOS dynamics |
| NTHU-DDD | ~36 | Day/Night + Glasses/Sunglasses | CRITICAL: adds glasses occlusion & night lighting |
| YawDD | 107 | In-vehicle dashcam, 2 cameras | CRITICAL: real driving perspective/mounting angle |
| MRL Eye Dataset | 37+ | 8 scales, varied light/pose | Large-scale eye-state labels for CNN eye-patch model |

Combined, these four datasets give ~300+ subjects across diverse conditions, lighting, occlusion, and camera mounting angles.

#### Tier 2: Supplementary Datasets (Add for Robustness)

| Dataset | What It Adds |
|---|---|
| DROZY | Includes EEG ground truth (Karolinska Sleepiness Scale); best drowsiness severity labelling |
| CEW (Closed Eyes in the Wild) | Large-scale in-the-wild closed eye images; perfect for CNN eye-patch model pretraining |
| 300-W / 300-VW | Landmark accuracy validation under extreme pose and lighting |
| LFW (Labeled Faces in the Wild) | Diversity augmentation for face detector robustness |

---

### Dataset Harmonisation Challenge

A critical practical issue: **different datasets use different label schemas**.

| Dataset | Labels |
|---|---|
| UTA-RLDD | Alert / Low Vigilance / Drowsy |
| NTHU-DDD | Drowsy / Non-Drowsy (binary per frame) |
| YawDD | Normal / Talking / Yawning |
| DROZY | KSS 1–9 (9-point Karolinska scale) |

You will need to define a **common label mapping** before training:

```
L0 (Alert)       <- UTA: Alert         | NTHU: Non-Drowsy | DROZY: KSS 1-4
L1 (Mild)        <- UTA: Low Vigilance | NTHU: Non-Drowsy | DROZY: KSS 5-6
L2 (Moderate)    <- UTA: Drowsy        | NTHU: Drowsy     | DROZY: KSS 7-8
L3 (Severe)      <- (from DROZY KSS 9 + NTHU extreme sequences only)
```

> WARNING: L3 (severe/microsleep) data is extremely rare in all open datasets due to safety and ethical recording constraints. Consider **class-weighted loss functions** or **SMOTE-style oversampling** to handle severe class imbalance, or use synthetic sequence augmentation for L3.

---

### Final Dataset Recommendation Summary

```
Phase 1 (Current POC):
    [DONE] UTA-RLDD only (sufficient for proof of concept)

Phase 2 (Development — before first real model release):
    [TODO] UTA-RLDD + NTHU-DDD + YawDD + MRL Eye
    Target: ~500+ subjects, day/night, glasses, in-vehicle angle

Phase 3 (Production):
    [FUTURE] All above + DROZY + CEW + domain-specific data collection
    In-vehicle custom recordings under NIR illumination at target
    camera mounting position (dashcam angle, ~70cm from face)
```

> TIP: Even with excellent datasets, **fine-tuning on a small set of in-domain data** (real DMS camera angle, your specific hardware, representative ethnicity distribution for your target market) will deliver the largest single accuracy improvement. Dataset diversity + domain-specific fine-tuning is the production-grade path.

---

## Summary Table

| Question | Short Answer |
|---|---|
| Why not LSTM/RNN? | Feature engineering already encodes temporal context; RF is 10–28x faster; use a hybrid two-tier approach — RF for real-time + GRU for trend detection |
| Ensemble composition? | Model A (LightGBM) + Model B (GRU 32-unit) + Model C (tiny CNN eye-patch), fused via weighted soft vote |
| Is UTA-RLDD enough? | No for production. Add NTHU-DDD (glasses + night), YawDD (in-vehicle angle), MRL Eye (eye-patch CNN). Use DROZY for severity calibration. |

---

*Research notes compiled for ALERT — Pre-Development Stage, August 2026.*
