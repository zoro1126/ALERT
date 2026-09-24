<div align="center">

# Abstract

</div>

Driver fatigue and drowsiness contribute to over 20% of fatal highway accidents worldwide. Conventional Driver Monitoring Systems relying on cloud-based architectures suffer from network latency, connectivity dead zones, and biometric privacy liabilities, making them unsuitable for reliable real-world deployment.

To address this, we present **ALERT** (*Automated Landmark based Eye and Response Tracker*), a fully offline, privacy-preserving Driver Monitoring System designed for real-time fatigue detection on consumer and embedded edge hardware without any cloud dependency.

ALERT extracts 468 3D facial landmarks per frame using MediaPipe FaceLandmarker and computes a 9-dimensional temporal feature vector: Eye Aspect Ratio (EAR), EAR standard deviation, PERCLOS (60-second rolling eye closure ratio), Blink Rate, Blink Duration, Mouth Aspect Ratio (MAR), Yawn Count, Head Pitch, and Mouth-Over-Eye ratio (MOE). A personalised 10-second baseline calibration adapts EAR and MAR thresholds to individual driver anatomy at session start, paired with exponential moving average drift adaptation for long-duration drives. Head-pose foreshortening on EAR is corrected via solvePnP 3D pose estimation.

Fatigue classification is performed by a three-model ensemble: a LightGBM gradient boosted classifier on the 9D feature vector for per-frame geometric detection, a 32-unit GRU sequence model on 5-second temporal windows for gradual onset trend detection, and a lightweight CNN operating on raw 32x16 pixel eye-patch crops for occlusion-robust pixel-level validation. The three outputs are fused via weighted soft-vote to classify driver state into four severity levels — L0 Alert, L1 Mild Fatigue, L2 Moderate Fatigue, and L3 Severe/Microsleep — targeting greater than 97% recall on L2 and L3 classes. The full pipeline achieves under 33ms end-to-end latency at 30 FPS on Intel Core i5/i7 consumer hardware, with an architecture forward-compatible with ONNX Runtime, OpenVINO, and TensorRT INT8 for embedded deployment on NVIDIA Jetson Orin Nano.

**Keywords:** Driver Fatigue Detection, Edge AI, Computer Vision, Eye Aspect Ratio (EAR), PERCLOS, Ensemble Learning, GRU, MediaPipe, Temporal Feature Engineering, Offline Inference.
