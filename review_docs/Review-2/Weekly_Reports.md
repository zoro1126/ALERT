# ALERT: Driver Fatigue Detection System
## B.Tech CE Minor Project — Weekly Progress Reports (6 Weeks)
### Reporting Period: 08/08/2026 to 19/09/2026

> **Handwriting-Friendly Format:**  
> All responses below are kept concise, direct, and short (1–2 lines per field) based on the official template [`review_docs/Review-2/weekly_report.pdf`](file:///home/prem/Desktop/projects/ALERT/review_docs/Review-2/weekly_report.pdf) so they can be easily rewritten by hand.

---

# Week 1: 08/08/2026 – 14/08/2026

### Section A: Details of Student
* **1. Name(s) of Student(s):** Prem Deshani, Kavya Kothari, Mehulsinh Rathod
* **2. Enrollment No.:** 2301030600043, 2301030600045, 2301030600049
* **3. Department:** Computer Engineering (AI & ML)
* **4. Group/Individual:** Group (Project ID: CE_AIML_31)
* **5. Project Phase:** Minor Project Phase I
* **6. Project Title:** A.L.E.R.T. (Adaptive Landmark Eye-state & Reaction-Time)
* **7. Name of Guide:** Ms. Vishwa Patel
* **8. Name of Co-Guide:** N/A
* **9. Week No. & Dates:** Week 1: 08/08/2026 – 14/08/2026
* **10. Date of previous meetings:** 08/08/2026 (Project Allotment & Kickoff)

### Section B: Objectives Planned for the Week
* **9. Previous Review progress:** Project formulation stage (0%).
* **10. Planned at start of week:** Define problem statement, survey foundational papers, and set up project repository.

### Work Completed During the Week
* **11. Actual progress made:** Finalized ALERT scope (offline, CPU-only DMS); set up Git repository structure; created centralized `Config` class for hyper-parameters.

### Literature Review Progress
* **12. New Papers Studied:** YES
* **13. Total papers reviewed:** 3 papers
* **14. Key learning / gap identified:** Static EAR thresholds fail across diverse eye shapes; cloud-based systems introduce latency jitter and privacy issues.

### Scope & Problem Definition Status
* **15. Status:**
  * Problem statement: Finalized (Real-time CPU fatigue detection).
  * Scope: Defined (Offline, privacy-preserving, consumer hardware).
  * Objectives: Refined (Focus on <50ms latency without GPU).
  * Changes suggested: Avoid cloud APIs; use 100% on-device inference.

### Section C: Implementation / Experimentation Status
* **16. Module implemented:** Repository scaffolding and `src/utils/config.py`.
* **17. Tools / Technologies used:** Python 3.11, Git, VS Code.
* **18. Results obtained:** Clean project architecture and global configuration established.
* **19. Pending work:** Landmark detection pipeline and geometric feature calculations.

### Industry / Expert Interaction
* **20. Meeting conducted:** No
* **21. Expert Name:** N/A
* **22. Mode:** N/A
* **23. Purpose & outcome:** N/A

### Issues / Challenges Faced
* **24. Challenges:** Choosing between dlib (slow on CPU) and MediaPipe Tasks API.

### Section D: Guide's Suggestions & Corrections
* **25. Guide suggestions:** Benchmark MediaPipe on CPU; design multi-feature fusion beyond eye blinks.
* **26. Plan for next week:** Integrate MediaPipe FaceLandmarker and implement EAR, MAR, and PERCLOS.

### Overall Progress Status
* **27. Completion Percentage:** 15%
* **28. Progress Rating:** Very Good

---

# Week 2: 15/08/2026 – 21/08/2026

### Section A: Details of Student
* **1. Name(s) of Student(s):** Prem Deshani, Kavya Kothari, Mehulsinh Rathod
* **2. Enrollment No.:** 2301030600043, 2301030600045, 2301030600049
* **3. Department:** Computer Engineering (AI & ML)
* **4. Group/Individual:** Group (Project ID: CE_AIML_31)
* **5. Project Phase:** Minor Project Phase I
* **6. Project Title:** A.L.E.R.T. (Adaptive Landmark Eye-state & Reaction-Time)
* **7. Name of Guide:** Ms. Vishwa Patel
* **8. Name of Co-Guide:** N/A
* **9. Week No. & Dates:** Week 2: 15/08/2026 – 21/08/2026
* **10. Date of previous meetings:** 14/08/2026

### Section B: Objectives Planned for the Week
* **9. Previous Review progress:** 15% (Project configuration and literature initialized).
* **10. Planned at start of week:** Extract 468 3D landmarks and compute EAR, MAR, PERCLOS, and head pose.

### Work Completed During the Week
* **11. Actual progress made:** Implemented `feature_extractor.py` with EAR, MAR, rolling PERCLOS, and blink counter; integrated `solvePnP` for head pitch/yaw estimation.

### Literature Review Progress
* **12. New Papers Studied:** YES
* **13. Total papers reviewed:** 5 papers (Added MediaPipe + PERCLOS and UTA-RLDD baseline).
* **14. Key learning / gap identified:** Head rotation distorts 2D eye geometry; 3D head pitch is necessary to detect nodding fatigue.

### Scope & Problem Definition Status
* **15. Status:**
  * Problem statement: Finalized.
  * Scope: Maintained.
  * Objectives: Refined (Added head pose compensation).
  * Changes suggested: None.

### Section C: Implementation / Experimentation Status
* **16. Module implemented:** `FeatureExtractor` and `HeadPoseEstimator` in `src/core/feature_extractor.py`.
* **17. Tools / Technologies used:** OpenCV (solvePnP), MediaPipe Tasks, NumPy.
* **18. Results obtained:** 468 landmarks extracted at ~25 ms/frame on CPU; accurate 3D pitch/yaw angles.
* **19. Pending work:** Dataset ingestion and video feature extraction pipeline.

### Industry / Expert Interaction
* **20. Meeting conducted:** No
* **21. Expert Name:** N/A
* **22. Mode:** N/A
* **23. Purpose & outcome:** N/A

### Issues / Challenges Faced
* **24. Challenges:** Adapting MediaPipe Tasks API format to match geometric formula inputs.

### Section D: Guide's Suggestions & Corrections
* **25. Guide suggestions:** Add unit tests for EAR and head pose calculations under extreme angles.
* **26. Plan for next week:** Ingest UTA-RLDD dataset and build batch feature extraction script.

### Overall Progress Status
* **27. Completion Percentage:** 30%
* **28. Progress Rating:** Very Good

---

# Week 3: 22/08/2026 – 28/08/2026

### Section A: Details of Student
* **1. Name(s) of Student(s):** Prem Deshani, Kavya Kothari, Mehulsinh Rathod
* **2. Enrollment No.:** 2301030600043, 2301030600045, 2301030600049
* **3. Department:** Computer Engineering (AI & ML)
* **4. Group/Individual:** Group (Project ID: CE_AIML_31)
* **5. Project Phase:** Minor Project Phase I
* **6. Project Title:** A.L.E.R.T. (Adaptive Landmark Eye-state & Reaction-Time)
* **7. Name of Guide:** Ms. Vishwa Patel
* **8. Name of Co-Guide:** N/A
* **9. Week No. & Dates:** Week 3: 22/08/2026 – 28/08/2026
* **10. Date of previous meetings:** 21/08/2026

### Section B: Objectives Planned for the Week
* **9. Previous Review progress:** 30% (Geometric feature extraction and head pose complete).
* **10. Planned at start of week:** Ingest UTA-RLDD video dataset and build batch feature extraction script.

### Work Completed During the Week
* **11. Actual progress made:** Structured UTA-RLDD (141 videos across 4 folds); built `dataset.py` scanner and `frame_extractor.py`; built `extract_cache.py` sliding window builder.

### Literature Review Progress
* **12. New Papers Studied:** YES
* **13. Total papers reviewed:** 7 papers (Added Adaptive EAR and VigilEye CPU papers).
* **14. Key learning / gap identified:** Downsampling videos to 15 FPS preserves blink dynamics while cutting computation by 50%.

### Scope & Problem Definition Status
* **15. Status:**
  * Problem statement: Finalized.
  * Scope: Maintained.
  * Objectives: Maintained.
  * Changes suggested: Implement subject-independent 4-fold cross-validation.

### Section C: Implementation / Experimentation Status
* **16. Module implemented:** `dataset.py`, `frame_extractor.py`, and `scripts/extract_cache.py`.
* **17. Tools / Technologies used:** OpenCV VideoCapture, MediaPipe Tasks, Python Pathlib.
* **18. Results obtained:** Sample videos processed into 60-frame sliding windows with stride 15.
* **19. Pending work:** Full dataset batch caching and deep learning sequence model.

### Industry / Expert Interaction
* **20. Meeting conducted:** No
* **21. Expert Name:** N/A
* **22. Mode:** N/A
* **23. Purpose & outcome:** N/A

### Issues / Challenges Faced
* **24. Challenges:** Video format inconsistencies and handling multi-part video filenames in Fold 3.

### Section D: Guide's Suggestions & Corrections
* **25. Guide suggestions:** Ensure strict separation of subjects between training and validation folds.
* **26. Plan for next week:** Design lightweight BiGRU neural network with temporal attention.

### Overall Progress Status
* **27. Completion Percentage:** 50%
* **28. Progress Rating:** Excellent

---

# Week 4: 29/08/2026 – 04/09/2026

### Section A: Details of Student
* **1. Name(s) of Student(s):** Prem Deshani, Kavya Kothari, Mehulsinh Rathod
* **2. Enrollment No.:** 2301030600043, 2301030600045, 2301030600049
* **3. Department:** Computer Engineering (AI & ML)
* **4. Group/Individual:** Group (Project ID: CE_AIML_31)
* **5. Project Phase:** Minor Project Phase I
* **6. Project Title:** A.L.E.R.T. (Adaptive Landmark Eye-state & Reaction-Time)
* **7. Name of Guide:** Ms. Vishwa Patel
* **8. Name of Co-Guide:** N/A
* **9. Week No. & Dates:** Week 4: 29/08/2026 – 04/09/2026
* **10. Date of previous meetings:** 28/08/2026

### Section B: Objectives Planned for the Week
* **9. Previous Review progress:** 50% (Dataset pipeline and feature extraction complete).
* **10. Planned at start of week:** Design PyTorch temporal model and implement sequence data augmentations.

### Work Completed During the Week
* **11. Actual progress made:** Implemented `FatigueClassifier` (2-stage BiGRU + Temporal Attention, 63.9k params); developed `augment.py` (noise, time-warp, channel mask, temporal crop).

### Literature Review Progress
* **12. New Papers Studied:** YES
* **13. Total papers reviewed:** 9 papers (Added BiLSTM keypoints and SAFE-DRIVE-AI attention).
* **14. Key learning / gap identified:** Temporal self-attention assigns high learned weights to micro-sleep frames within normal sequences.

### Scope & Problem Definition Status
* **15. Status:**
  * Problem statement: Finalized.
  * Scope: Maintained.
  * Objectives: Maintained.
  * Changes suggested: Keep model footprint <100 KB for fast CPU inference.

### Section C: Implementation / Experimentation Status
* **16. Module implemented:** `model.py`, `augment.py`, and `CachedWindowDataset` in `trainer.py`.
* **17. Tools / Technologies used:** PyTorch 2.x, NumPy, SciPy.
* **18. Results obtained:** Model passes all gradient and shape tests; parameter count is 63,939 (0.25 MB).
* **19. Pending work:** Adaptive threshold calibration, alert state machine, and live camera UI.

### Industry / Expert Interaction
* **20. Meeting conducted:** No
* **21. Expert Name:** N/A
* **22. Mode:** N/A
* **23. Purpose & outcome:** N/A

### Issues / Challenges Faced
* **24. Challenges:** Class imbalance in sequential windows between Alert and Drowsy states.

### Section D: Guide's Suggestions & Corrections
* **25. Guide suggestions:** Use WeightedRandomSampler to prevent majority-class bias during training.
* **26. Plan for next week:** Build baseline calibrator, tiered alert machine, and OpenCV live dashboard.

### Overall Progress Status
* **27. Completion Percentage:** 68%
* **28. Progress Rating:** Excellent

---

# Week 5: 05/09/2026 – 11/09/2026

### Section A: Details of Student
* **1. Name(s) of Student(s):** Prem Deshani, Kavya Kothari, Mehulsinh Rathod
* **2. Enrollment No.:** 2301030600043, 2301030600045, 2301030600049
* **3. Department:** Computer Engineering (AI & ML)
* **4. Group/Individual:** Group (Project ID: CE_AIML_31)
* **5. Project Phase:** Minor Project Phase I
* **6. Project Title:** A.L.E.R.T. (Adaptive Landmark Eye-state & Reaction-Time)
* **7. Name of Guide:** Ms. Vishwa Patel
* **8. Name of Co-Guide:** N/A
* **9. Week No. & Dates:** Week 5: 05/09/2026 – 11/09/2026
* **10. Date of previous meetings:** 04/09/2026

### Section B: Objectives Planned for the Week
* **9. Previous Review progress:** 68% (BiGRU architecture and augmentations complete).
* **10. Planned at start of week:** Build adaptive calibrator, tiered alert state machine, and live HUD overlay.

### Work Completed During the Week
* **11. Actual progress made:** Implemented `calibrator.py` (10s open-eye calibration); developed `alert_manager.py` with procedural audio tones; built `dashboard.py` and `overlay.py` with `--demo` mode.

### Literature Review Progress
* **12. New Papers Studied:** YES
* **13. Total papers reviewed:** 10 papers (Completed with Head pose + landmark fusion paper).
* **14. Key learning / gap identified:** Synthesizing procedural sine-wave audio tones via Pygame removes dependency on external WAV files.

### Scope & Problem Definition Status
* **15. Status:**
  * Problem statement: Finalized.
  * Scope: Maintained.
  * Objectives: Refined (Added standalone threshold POC demo mode).
  * Changes suggested: Allow running system with threshold-only logic before model training finishes.

### Section C: Implementation / Experimentation Status
* **16. Module implemented:** `calibrator.py`, `alert_manager.py`, `overlay.py`, and `run_alert.py`.
* **17. Tools / Technologies used:** OpenCV (HUD), Pygame (procedural audio), JSON.
* **18. Results obtained:** Live webcam demo runs at >30 FPS with responsive visual HUD and debounced audio alarms.
* **19. Pending work:** Full dataset cache extraction across all folds and training benchmarking.

### Industry / Expert Interaction
* **20. Meeting conducted:** No
* **21. Expert Name:** N/A
* **22. Mode:** N/A
* **23. Purpose & outcome:** N/A

### Issues / Challenges Faced
* **24. Challenges:** Linux webcam device index assignment (`/dev/video0` virtual vs `/dev/video1` hardware).

### Section D: Guide's Suggestions & Corrections
* **25. Guide suggestions:** Include available video device diagnostics in camera error messages; prepare Review-2 PPT.
* **26. Plan for next week:** Run full dataset cache extraction, benchmark CPU training, and finalize Review-2 presentation.

### Overall Progress Status
* **27. Completion Percentage:** 82%
* **28. Progress Rating:** Excellent

---

# Week 6: 12/09/2026 – 19/09/2026

### Section A: Details of Student
* **1. Name(s) of Student(s):** Prem Deshani, Kavya Kothari, Mehulsinh Rathod
* **2. Enrollment No.:** 2301030600043, 2301030600045, 2301030600049
* **3. Department:** Computer Engineering (AI & ML)
* **4. Group/Individual:** Group (Project ID: CE_AIML_31)
* **5. Project Phase:** Minor Project Phase I
* **6. Project Title:** A.L.E.R.T. (Adaptive Landmark Eye-state & Reaction-Time)
* **7. Name of Guide:** Ms. Vishwa Patel
* **8. Name of Co-Guide:** N/A
* **9. Week No. & Dates:** Week 6: 12/09/2026 – 19/09/2026
* **10. Date of previous meetings:** 11/09/2026

### Section B: Objectives Planned for the Week
* **9. Previous Review progress:** 82% (Operational real-time POC and HUD complete).
* **10. Planned at start of week:** Run full dataset feature extraction, benchmark training, and prepare Review-2 deliverables.

### Work Completed During the Week
* **11. Actual progress made:** Processed all 141 UTA-RLDD videos into 84,942 feature windows (29 MB cache); fixed `time_warp` interpolation bug; benchmarked CPU training (68.5s/epoch, 1,510 samples/s); finalized Review-2 presentation.

### Literature Review Progress
* **12. New Papers Studied:** NO
* **13. Total papers reviewed:** 10 papers (Literature review finalized).
* **14. Key learning / gap identified:** Batch size of 128 maximizes Intel MKL AVX2 vectorization on CPU, reducing training time to ~22 minutes.

### Scope & Problem Definition Status
* **15. Status:**
  * Problem statement: Finalized.
  * Scope: Review-2 Milestone achieved (Baseline/threshold POC operational; Model training in progress).
  * Objectives: Maintained.
  * Changes suggested: None.

### Section C: Implementation / Experimentation Status
* **16. Module implemented:** Full batch extraction run, `time_warp` bugfix in `augment.py`, and `trainer.py` benchmark.
* **17. Tools / Technologies used:** PyTorch CPU, Intel MKL, MediaPipe Tasks, PowerPoint / Markdown.
* **18. Results obtained:** 84,942 windows cached; 68.5s/epoch training speed; Review-2 presentation and monthly report complete.
* **19. Pending work:** Completing multi-epoch model convergence and validation evaluation.

### Industry / Expert Interaction
* **20. Meeting conducted:** No
* **21. Expert Name:** N/A
* **22. Mode:** N/A
* **23. Purpose & outcome:** N/A

### Issues / Challenges Faced
* **24. Challenges:** Managing long execution time (~3.5 hrs) during multi-video feature extraction on CPU.

### Section D: Guide's Suggestions & Corrections
* **25. Guide suggestions:** Proceed with full multi-epoch model training and evaluate confusion matrix metrics.
* **26. Plan for next week:** Complete 25-epoch BiGRU training run on Fold 4 validation and export to ONNX runtime.

### Overall Progress Status
* **27. Completion Percentage:** 90%
* **28. Progress Rating:** Excellent
