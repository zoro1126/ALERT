# Monthly Project Progress Report
## Project A.L.E.R.T. (Adaptive Landmark Eye-state & Reaction-Time)

* **Reporting Period:** August 08, 2026 – September 19, 2026 (6 Weeks)
* **Academic Program:** B.Tech. Computer Engineering (Artificial Intelligence & Machine Learning)
* **Institution:** Silver Oak College of Engineering & Technology, Silver Oak University
* **Project ID:** CE_AIML_31
* **Students:** Prem Deshani (2301030600043), Kavya Kothari (2301030600045), Mehulsinh Rathod (2301030600049)
* **Project Guide:** Ms. Vishwa Patel

---

### 1. Executive Summary

During the monthly period from **August 08, 2026 to September 19, 2026**, the ALERT development team completed the end-to-end design, preprocessing, and real-time proof-of-concept (POC) implementation for a privacy-preserving, edge-deployable Driver Fatigue Monitoring System.

Key accomplishments achieved during this timeframe:
1. **Literature Survey & Formulation:** Synthesized 10 core peer-reviewed research papers covering facial landmark analysis, temporal sequence modeling, adaptive thresholding, and automotive edge computing.
2. **Feature Extraction Pipeline:** Developed a 30+ FPS landmark analysis engine using Google MediaPipe Tasks API (468 3D landmarks via XNNPACK CPU delegate) combined with OpenCV `solvePnP` 3D head pose estimation.
3. **Adaptive Baseline Calibration:** Implemented and validated a personalized 10-second calibration routine that eliminates false positives caused by natural anatomical eye-shape variation.
4. **Complete Dataset Processing:** Ingested all 141 naturalistic multi-minute driving videos from the UTA-RLDD dataset (60 subjects, 4 folds) and generated an optimized in-memory feature cache containing **84,942 temporal window sequences** (29 MB compressed).
5. **Deep Learning Sequence Architecture:** Implemented a lightweight 63,939-parameter BiGRU network with Temporal Self-Attention and multi-threaded data augmentation.
6. **Operational Review-2 Milestone:** Successfully deployed an interactive real-time camera dashboard (`run_alert.py --demo`) with live HUD telemetry and procedural tiered audio warnings, while initiating multi-epoch deep learning model training.

---

### 2. Project Motivation & Scope Definition

Driver drowsiness and momentary micro-sleeps account for over 20% of fatal commercial and highway vehicular accidents worldwide. While existing commercial Driver Monitoring Systems (DMS) rely on high-end GPUs or cloud IoT streaming, they face severe latency jitter (150–500 ms), high recurring cellular bandwidth costs ($15–$30/month/vehicle), and biometric privacy liabilities under GDPR.

**Project Scope:**
* **100% On-Device CPU Inference:** Execute full real-time fatigue detection on standard consumer/edge x86/ARM processors without requiring discrete GPUs.
* **Zero Cloud Dependency:** Guarantee absolute driver privacy and continuous safety in tunnels, remote highways, and cellular dead zones.
* **Low Computational Footprint:** Sustain <2 GB RAM utilization, <100 MB model size, and sub-40 ms end-to-end latency per frame.

---

### 3. Chronological Sprint Summary (Course of 6 Weeks)

* **Week 1 (08/08/2026 – 14/08/2026) — Problem Formulation & Literature Survey:**
  * Formalized system requirements, project architecture, and coding standards.
  * Conducted in-depth literature review across 10 benchmark papers on EAR, MAR, PERCLOS, and temporal models.
  * Established the central configuration module (`Config`) defining all global thresholds and hyper-parameters.

* **Week 2 (15/08/2026 – 21/08/2026) — Landmark & Geometric Feature Engineering:**
  * Integrated Google MediaPipe Tasks `FaceLandmarker` with XNNPACK CPU delegate for 468 3D facial landmarks.
  * Built mathematical modules for Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), rolling PERCLOS, and blink rate.
  * Implemented `HeadPoseEstimator` via `cv2.solvePnP` to track Head Pitch, Yaw, and Roll from 6 canonical 3D facial anchors.

* **Week 3 (22/08/2026 – 28/08/2026) — Dataset Ingestion & Video Feature Extractor:**
  * Downloaded and validated the UTA-RLDD dataset structure across 4 cross-validation folds.
  * Built video dataset scanner and downsampling frame extractor (`TARGET_FPS = 15`).
  * Created batch cache builder (`scripts/extract_cache.py`) converting long-duration video streams into compressed `.npz` sequences.

* **Week 4 (29/08/2026 – 04/09/2026) — Deep Learning Model & Augmentation:**
  * Implemented `FatigueClassifier` consisting of a 2-stage BiGRU (64 $\rightarrow$ 32 units) and a Temporal Self-Attention layer.
  * Created unit test suites verifying gradient propagation, attention weight normalization, and save/load roundtrips.
  * Built temporal sequence augmentations (Gaussian noise, DTW time-warp, channel dropout, temporal cropping).

* **Week 5 (05/09/2026 – 11/09/2026) — Calibration, State Machine & Real-Time POC:**
  * Developed the adaptive 10-second open-eye calibrator (`calibrator.py`) to derive personalized driver thresholds.
  * Created a 3-tier safety state machine (`alert_manager.py`) with debouncing and procedural audio tone alerts.
  * Built the OpenCV HUD interface (`overlay.py`) displaying live EAR, MAR, head pitch, and fatigue alerts at 30+ FPS.

* **Week 6 (12/09/2026 – 19/09/2026) — Full Dataset Caching & Empirical Benchmarking:**
  * Completed feature extraction across all 141 videos in UTA-RLDD, generating 84,942 feature windows (29 MB cache).
  * Debugged array interpolation in `time_warp` augmentation.
  * Executed empirical CPU training benchmarks (68.5s per epoch, 1,510 samples/sec throughput on Intel Core i7-7820HQ).
  * Prepared Review-2 slide deck and technical documentation.

---

### 4. Technical Architecture & Modules Implemented

```
+--------------------------------------------------------------------------+
|                       Real-Time Camera Stream (30 FPS)                   |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
| MediaPipe Tasks FaceLandmarker (468 3D Landmarks via XNNPACK CPU, ~15ms) |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
| Geometric Feature Extractor (EAR_L, EAR_R, EAR_avg, MAR, solvePnP Pose)  |
+--------------------------------------------------------------------------+
          |                                                 |
          v                                                 v
+-----------------------+                         +------------------------+
| 10s Adaptive Baseline |                         | 60-Frame Rolling Queue |
| Threshold Calibrator  |                         | (4.0s Window @ 15 FPS) |
+-----------------------+                         +------------------------+
          |                                                 |
          v                                                 v
+--------------------------------------------------------------------------+
| Alert State Machine (NORMAL / WARNING / CRITICAL) + BiGRU Attention DL   |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
| Multi-Modal Output: Procedural Pygame Audio Tones + OpenCV Visual HUD   |
+--------------------------------------------------------------------------+
```

1. **`core.feature_extractor`:** Computes 6-point Euclidean EAR per eye, 8-point mouth contour MAR, and solves perspective-n-point geometry against a 3D canonical face model for pitch/yaw angles.
2. **`core.calibrator`:** Records 10 seconds (150 frames) of attentive driving to compute user-specific baseline EAR ($\mu - 2.5\sigma$ or $\mu \times 0.75$), serializing parameters to local JSON.
3. **`core.model`:** 2-layer Bidirectional GRU with learned attention weights allocating importance to micro-sleep frames. Total parameters: **63,939** (0.25 MB on disk).
4. **`core.alert_manager`:** State machine transitioning between Normal (green), Warning (amber chime, 1.5s sustained low vigilance), and Critical (continuous red buzzer, 2.5s sustained eye closure/head drop).
5. **`ui.dashboard` & `ui.overlay`:** Real-time OpenCV rendering engine drawing telemetry graphs, numerical readouts, and bounding anchors at <2 ms HUD rendering overhead.

---

### 5. Dataset & Preprocessing Metrics

* **Target Dataset:** University of Texas at Arlington Real-Life Drowsiness Dataset (UTA-RLDD).
* **Composition:** 60 participants, 141 long-duration video streams captured under ambient indoor and vehicle lighting.
* **Extracted Cache Volume:**
  * **Fold 1:** 36 videos $\rightarrow$ 21,780 windows (7.4 MB)
  * **Fold 2:** 36 videos $\rightarrow$ 21,540 windows (7.3 MB)
  * **Fold 3:** 33 videos $\rightarrow$ 20,939 windows (7.1 MB)
  * **Fold 4:** 36 videos $\rightarrow$ 20,683 windows (7.0 MB)
  * **Total:** **141 videos $\rightarrow$ 84,942 sequence windows (28.8 MB total storage)**.
* **Extraction Throughput:** Fully automated batch extraction using stateless MediaPipe Tasks in downsampled image mode (15 FPS).

---

### 6. Empirical Performance Benchmarks

Benchmarked on test machine (Intel Core i7-7820HQ CPU @ 2.90 GHz, 4 Cores / 8 Threads, 32 GB RAM, Integrated Graphics):

| Performance Dimension | Empirical Measurement | Target Specification | Status |
| :--- | :--- | :--- | :---: |
| **MediaPipe Landmark Latency** | 22.4 – 26.1 ms / frame | <35 ms | **Exceeded** |
| **solvePnP Head Pose Latency** | 0.8 – 1.2 ms / frame | <5 ms | **Exceeded** |
| **BiGRU-Attention Inference** | 2.1 – 2.8 ms / stride | <10 ms | **Exceeded** |
| **Full Pipeline Live Latency** | **28.5 – 33.2 ms (>30 FPS)** | **<50 ms (20 FPS)** | **Exceeded** |
| **Peak Operational Memory** | 185 MB RAM | <2,048 MB | **Exceeded** |
| **Model Checkpoint Size** | 0.25 MB (.pt file) | <100 MB | **Exceeded** |
| **Training DataLoader Speed** | 1,510 samples / second | >500 samples/s | **Exceeded** |
| **CPU Training Epoch Time** | **68.5 seconds / epoch** | <180 seconds | **Exceeded** |

---

### 7. Current Project Status (Review-2 Milestone)

* **Functional Deliverable:** The **Adaptive Baseline & Geometric Threshold Proof-of-Concept (POC)** is 100% operational, fully tested, and demonstrated live via webcam.
* **Data Preparation:** 100% complete across all 4 folds of the UTA-RLDD dataset (84,942 windows cached and verified).
* **Deep Learning Training:** The PyTorch BiGRU-Attention training pipeline, loss functions, learning rate schedulers, and augmentations are fully coded and benchmarked; full multi-epoch model convergence and validation runs are actively underway.

---

### 8. Key Challenges Faced & Resolved

1. **MediaPipe Tasks API Compatibility:** MediaPipe v0.10+ deprecated the legacy `mp.solutions.face_mesh` API. Designed a lightweight `_LandmarkAdapter` shim to map Tasks API `NormalizedLandmark` outputs to canonical dot-access geometry without incurring memory copies.
2. **Ethnic and Individual Anatomical Bias:** Drivers with naturally smaller eye apertures triggered constant false-positive warnings under static EAR thresholds. Resolved via the 10-second personal baseline calibrator.
3. **Data Augmentation Dimension Mismatch:** Fixed an interpolation bug in `time_warp` where non-uniform knot steps were mismatched against continuous sequence grids, restoring 1,510 samples/sec DataLoader throughput.
4. **Linux Camera Device Enumeration:** Overcame virtual video loopback issues by adding comprehensive camera index error handling and multi-device discovery.

---

### 9. Roadmap for Next Reporting Period

1. **Complete Multi-Epoch Model Training:** Train the BiGRU-Attention classifier for 25–50 epochs with Cosine Annealing learning rate scheduling to maximize validation macro-F1 score across fold 4.
2. **Comprehensive Evaluation & Confusion Matrix:** Generate automated evaluation reports calculating per-class precision, recall, and ROC curves for Alert, Low Vigilance, and Drowsy states.
3. **Embedded Inference Optimization:** Export PyTorch weights to ONNX format and evaluate INT8 quantization under OpenVINO for low-power edge CPU/NPU hardware.
4. **Near-Infrared (NIR) Hardware Exploration:** Evaluate camera sensor behavior under active 850 nm / 940 nm IR illumination for nighttime cabin conditions.

---

**Report Prepared By:** Prem Deshani, Kavya Kothari, Mehulsinh Rathod  
**Reviewed & Approved By:** Ms. Vishwa Patel (Project Guide)  
**Date:** September 19, 2026
