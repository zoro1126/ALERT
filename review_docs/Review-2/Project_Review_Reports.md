# Silver Oak College of Engineering and Technology
## B.Tech CE Minor Project (Phase I) — Periodic Review Reports
### Project A.L.E.R.T. (Adaptive Landmark Eye-state & Reaction-Time)

* **Month 1: 08/08/2026 to 19/09/2026 – Scan Copy Link:** `[Scan Copy Link / Drive URL]`
* **Week 1: 20/09/2026 to 26/09/2026 – Scan Copy Link:** `[Scan Copy Link / Drive URL]`

> **Handwriting-Friendly Format:**  
> All responses below are structured according to the official review template [`review_docs/Review-2/weekly_report.pdf`](file:///home/prem/Desktop/projects/ALERT/review_docs/Review-2/weekly_report.pdf). Entries are kept short, factual, and concise (1–2 lines per field) for easy handwritten transcription into physical booklets.

---

# Month 1: 08/08/2026 to 19/09/2026 – Scan Copy Link

### Section A: Details of Student
* **1. Name(s) of Student(s):** Prem Deshani, Kavya Kothari, Mehulsinh Rathod
* **2. Enrollment No.:** 2301030600043, 2301030600045, 2301030600049
* **3. Department:** Computer Engineering (AI & ML)
* **4. Group/Individual:** Group (Project ID: CE_AIML_31)
* **5. Minor Project Phase:** Minor Project Phase I
* **6. Project Title:** A.L.E.R.T. (Adaptive Landmark Eye-state & Reaction-Time)
* **7. Name of the Guide:** Ms. Vishwa Patel
* **8. Name of the Co-Guide (if any):** N/A
* **9. Month No. & Dates:** Month 1: 08/08/2026 to 19/09/2026
* **10. Date(s) of previous meetings of Guide:** 08/08/2026, 22/08/2026, 05/09/2026

### Section B: Objectives Planned for the Month
* **9. Previous Review date and progress achieved:** Project initiation stage (0% progress as of 08/08/2026).
* **10. What was planned at the start of the month:** Finalize problem definition, develop landmark feature extraction pipeline, design deep learning sequence model, and build working real-time baseline POC.

### Work Completed During the Month
* **11. Actual progress Made:** 
  * Implemented 468 3D landmark extraction via MediaPipe Tasks API on CPU.
  * Formulated 8D geometric feature vector (EAR, MAR, rolling PERCLOS, solvePnP head pitch/yaw).
  * Built 63,939-parameter BiGRU network with Temporal Self-Attention and data augmentations.
  * Developed 10-second open-eye adaptive baseline calibrator to eliminate eye-shape bias.
  * Created real-time camera dashboard (`run_alert.py --demo`) with live HUD and procedural audio alerts.

### Literature Review Progress
* **12. New Papers Studied this Period:** YES
* **13. Total papers reviewed till date:** 10 core papers (Target: min. 4–5 core papers met).
* **14. Key learning / gap identified:** Static thresholding causes severe ethnic/facial bias; 10s adaptive calibration and multi-modal feature fusion are necessary for reliable edge DMS.

### Scope & Problem Definition Status
* **15. Status:**
  * **Problem statement finalized:** Yes (Real-time CPU-only driver drowsiness detection).
  * **Scope defined:** Yes (100% offline, privacy-preserving, consumer hardware).
  * **Objectives refined:** Yes (Achieve >25 FPS without dedicated GPU).
  * **Changes suggested this month:** Focus on CPU-optimized landmark features rather than heavy 3D-CNNs.

### Section C: Implementation / Experimentation Status
* **16. Module / Algorithm / Model implemented this month:** 
  * `core/feature_extractor.py` (EAR, MAR, solvePnP head pose).
  * `core/calibrator.py` (10s adaptive baseline calibration).
  * `core/model.py` (2-stage BiGRU + Temporal Self-Attention).
  * `core/alert_manager.py` (State machine + procedural Pygame audio tones).
  * `ui/dashboard.py` and `ui/overlay.py` (Live OpenCV telemetry HUD).
* **17. Tools / Technologies used:** Python 3.11, MediaPipe Tasks, OpenCV, Pygame, PyTorch 2.x, NumPy.
* **18. Results obtained:** Operational real-time POC running at >30 FPS (~25 ms latency) on webcam with live calibration and tiered audio alarms.
* **19. Pending implementation work:** Full dataset feature extraction across all 141 videos and multi-epoch model training.

### Industry / Expert Interaction (if applicable)
* **20. Visit / Meeting conducted:** No
* **21. Expert / Industry Name:** N/A
* **22. Mode:** N/A
* **23. Purpose & outcome:** N/A

### Issues / Challenges Faced
* **24. Issues / Challenges Faced:** MediaPipe Tasks API integration without legacy solutions; Linux camera index assignment (`/dev/video0` loopback vs `/dev/video1` hardware webcam).

### Section D: Guide's Suggestions & Corrections (To be filled during review)
* **25. Guide's Suggestions & Corrections:** Ensure subject-independent data partitioning and develop a standalone demo mode for Review-2.
* **26. Plan for Next Period (Clear, achievable tasks):** Extract full feature cache from UTA-RLDD dataset, benchmark CPU training speed, and prepare Review-2 presentation.

### Overall Progress Status
* **27. Completion Percentage (approx.):** 80%
* **28. Progress Rating:** Excellent

| Role | Name | Signature | Date |
| :--- | :--- | :--- | :--- |
| **Students** | Prem Deshani, Kavya Kothari, Mehulsinh Rathod | | 19/09/2026 |
| **Guide** | Ms. Vishwa Patel | | 19/09/2026 |

---

# Week 1: 20/09/2026 to 26/09/2026 – Scan Copy Link

### Section A: Details of Student
* **1. Name(s) of Student(s):** Prem Deshani, Kavya Kothari, Mehulsinh Rathod
* **2. Enrollment No.:** 2301030600043, 2301030600045, 2301030600049
* **3. Department:** Computer Engineering (AI & ML)
* **4. Group/Individual:** Group (Project ID: CE_AIML_31)
* **5. Minor Project Phase:** Minor Project Phase I
* **6. Project Title:** A.L.E.R.T. (Adaptive Landmark Eye-state & Reaction-Time)
* **7. Name of the Guide:** Ms. Vishwa Patel
* **8. Name of the Co-Guide (if any):** N/A
* **9. Week No. & Dates:** Week 1: 20/09/2026 to 26/09/2026
* **10. Date(s) of previous meetings of Guide:** 19/09/2026 (Month 1 Review)

### Section B: Objectives Planned for the Week
* **9. Previous Review date and progress achieved:** 19/09/2026, 80% progress (Baseline/threshold POC operational).
* **10. What was planned at the start of the week:** Execute full feature caching on all 141 UTA-RLDD videos, fix data augmentation bugs, benchmark CPU training throughput, and prepare Review-2 slides.

### Work Completed During the Week
* **11. Actual progress Made:** 
  * Extracted and cached all 141 UTA-RLDD videos into 84,942 sliding window sequences (29 MB compressed cache across 4 folds).
  * Debugged and resolved interpolation array mismatch in `time_warp` augmentation in `src/training/augment.py`.
  * Verified DataLoader throughput (1,510 samples/sec) and executed 1-epoch CPU training benchmark (68.5s/epoch on Intel Core i7-7820HQ).
  * Prepared Review-2 slide deck, presentation notes, and literature review matrix based on university format.

### Literature Review Progress
* **12. New Papers Studied this Week:** YES (Synthesized and mapped into Review-2 presentation)
* **13. Total papers reviewed till date:** 10 papers
* **14. Key learning / gap identified:** Batch size of 128 maximizes Intel MKL AVX2 vectorization on CPU, reducing full training convergence to ~22 minutes.

### Scope & Problem Definition Status
* **15. Status:**
  * **Problem statement finalized:** Yes
  * **Scope defined:** Yes (Review-2 Milestone: Threshold POC operational; Deep learning model training in progress).
  * **Objectives refined:** Yes
  * **Changes suggested this week:** Track compressed cache in version control for zero-setup replication.

### Section C: Implementation / Experimentation Status
* **16. Module / Algorithm / Model implemented this week:** 
  * `scripts/extract_cache.py` (Full dataset batch extraction run).
  * `src/training/augment.py` (Bugfix in `time_warp` interpolation).
  * `src/training/trainer.py` (CPU training loop benchmark & dataset loader).
  * `review_docs/Review-2/Presentation_Content.md` (Slide deck content).
* **17. Tools / Technologies used:** PyTorch 2.x CPU (Intel MKL / OpenMP), MediaPipe Tasks, NumPy, Git.
* **18. Results obtained:** 84,942 feature windows verified; training benchmark at 68.5s/epoch; Review-2 presentation ready.
* **19. Pending implementation work:** Multi-epoch model training to convergence, confusion matrix evaluation, and ONNX runtime export.

### Industry / Expert Interaction (if applicable)
* **20. Visit / Meeting conducted this week:** No
* **21. Expert / Industry Name:** N/A
* **22. Mode:** N/A
* **23. Purpose & outcome:** N/A

### Issues / Challenges Faced
* **24. Issues / Challenges Faced:** Long batch extraction duration (~3.5 hours on CPU); terminal appeared silent due to post-video logging (resolved).

### Section D: Guide's Suggestions & Corrections (To be filled during review)
* **25. Guide's Suggestions & Corrections:** Highlight operational baseline POC during Review-2 presentation; proceed with multi-epoch BiGRU training.
* **26. Plan for Next Week (Clear, achievable tasks):** Complete 25-epoch BiGRU training on Fold 4 validation, compute per-class F1 scores, and export model to ONNX.

### Overall Progress Status
* **27. Completion Percentage (approx.):** 90%
* **28. Progress Rating:** Excellent

| Role | Name | Signature | Date |
| :--- | :--- | :--- | :--- |
| **Students** | Prem Deshani, Kavya Kothari, Mehulsinh Rathod | | 26/09/2026 |
| **Guide** | Ms. Vishwa Patel | | 26/09/2026 |
