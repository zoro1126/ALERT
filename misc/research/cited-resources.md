# Cited Resources & References

> All references used in the Driver Fatigue Detection research document, organized by category.

---

## Seminal Papers

### [R1] EAR — Eye Aspect Ratio

- **Title:** Real-Time Eye Blink Detection using Facial Landmarks
- **Authors:** Tereza Soukupová, Jan Čech
- **Venue:** 21st Computer Vision Winter Workshop (CVWW), Rimske Toplice, Slovenia
- **Year:** 2016
- **Key Contribution:** Introduced the Eye Aspect Ratio (EAR) as a scalar quantity to characterize eye openness using six facial landmark points per eye. Demonstrated real-time blink detection using EAR features with an SVM classifier.
- **URL:** <https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf>

### [R2] PERCLOS Standard

- **Title:** Development of a Drowsiness Warning System Based on the PERCLOS Drowsiness Measure
- **Authors:** Wierwille, W.W., Ellsworth, L.A., et al.
- **Venue:** FHWA (Federal Highway Administration) Report
- **Year:** 1994
- **Key Contribution:** Established PERCLOS (Percentage of Eye Closure over time) as a validated drowsiness metric, defined as the proportion of time the eyes are ≥80% closed within a specified time window.

### [R3] Facial Landmark Detection — dlib

- **Title:** One Millisecond Face Alignment with an Ensemble of Regression Trees
- **Authors:** Vahid Kazemi, Josephine Sullivan
- **Venue:** IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- **Year:** 2014
- **Key Contribution:** Introduced the ensemble of regression trees (ERT) method for rapid facial landmark detection (68 landmarks), forming the basis of dlib's shape predictor.

### [R4] MediaPipe Face Mesh

- **Title:** MediaPipe: A Framework for Building Perception Pipelines
- **Authors:** Google Research (Camillo Lugaresi et al.)
- **Venue:** arXiv preprint (arXiv:1906.08172)
- **Year:** 2019
- **Key Contribution:** Introduced the MediaPipe framework including Face Mesh with 468 facial landmark points, optimized for real-time mobile/edge inference.

---

## Driver Monitoring & Drowsiness Detection

### [R5] Multi-Feature Drowsiness Detection

- **Title:** Driver Drowsiness Detection Using Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR) and PERCLOS
- **Venue:** River Publishers — Journal of Web Engineering & Technology
- **Year:** 2023
- **Key Contribution:** Combined EAR, MAR, and PERCLOS metrics for multi-signal fatigue detection with adaptive thresholds.
- **URL:** <https://www.riverpublishers.com>

### [R6] Head Pose Estimation via PnP

- **Title:** Perspective-n-Point Problem — Head Pose Estimation from 2D-3D Correspondences
- **Authors:** Various (foundational work by Fischler & Bolles, 1981; modern adaptations)
- **Key Contribution:** The solvePnP algorithm maps 2D facial landmarks to a 3D face model to extract rotation (Euler angles: yaw, pitch, roll) and translation vectors, enabling head pose estimation.

### [R7] Hybrid CNN + Geometric Feature Models

- **Title:** Deep Learning-Based Driver Drowsiness Detection: A Comprehensive Survey
- **Venue:** arXiv preprint
- **Year:** 2024
- **Key Contribution:** Surveyed the integration of classical geometric ratios (EAR/MAR) with CNNs and Vision Transformers (ViTs) for improved drowsiness detection under challenging conditions.
- **URL:** <https://arxiv.org>

---

## Lighting & Environmental Robustness

### [R8] Zero-DCE — Low-Light Enhancement

- **Title:** Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement
- **Authors:** Chunle Guo, Chongyi Li, et al.
- **Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- **Year:** 2020
- **Key Contribution:** Proposed a zero-reference deep curve estimation method (Zero-DCE) for enhancing low-light images without requiring paired training data. Applicable as a preprocessing step for NIR/low-light face detection.

### [R9] CLAHE — Contrast Limited Adaptive Histogram Equalization

- **Title:** Adaptive Histogram Equalization and Its Variations
- **Authors:** Pizer, S.M., Amburn, E.P., et al.
- **Venue:** Computer Vision, Graphics, and Image Processing
- **Year:** 1987
- **Key Contribution:** Introduced CLAHE as an enhancement over standard histogram equalization, limiting noise amplification by operating on localized tiles — critical for preprocessing low-light face images.

### [R10] Multi-Spectral Face Detection

- **Title:** Deep Multi-Spectral Learning for Thermal Infrared and Visible Face Detection
- **Venue:** IEEE Transactions on Image Processing
- **Year:** 2022
- **Key Contribution:** Demonstrated that models leveraging both visible and thermal/NIR imagery achieve significantly higher face detection accuracy across all lighting conditions compared to single-spectrum approaches.

---

## Benchmark Datasets

### [R11] NTHU-DDD (National Tsing Hua University — Driver Drowsiness Detection)

- **Description:** Multi-condition drowsiness detection dataset featuring varied lighting (day/night), glasses/sunglasses occlusion, and multiple subjects.
- **Recent Benchmark:** Transformer-based models achieved 99.71% accuracy (2025); hybrid VGG16+LeNet achieved 98.69%.
- **URL:** <http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/>

### [R12] UTA-RLDD (University of Texas at Arlington — Real-Life Drowsiness Dataset)

- **Description:** Real-life drowsiness dataset with 60 participants in three states: alert, low vigilance, and drowsy. Captured via webcam in naturalistic conditions.
- **Recent Benchmark:** YOLOv5/v8 achieved 100% precision/recall (mAP@0.5: 99.5%); KNN classifier achieved 98.89%.
- **URL:** <https://sites.google.com/view/utaboredanddrowsydataset/>

### [R13] Blink Detection Datasets

- **ZJU Eye Blink Dataset** — Standard benchmark for eye blink detection algorithms.
- **Eyeblink8** — Eight-subject dataset for evaluating blink frequency detection.
- **Silesian Facial Dataset** — Used for evaluating landmark-based blink detection under varied conditions.

---

## Edge Computing & Hardware

### [R14] NVIDIA Jetson Platform

- **Title:** NVIDIA JetPack SDK Documentation
- **Publisher:** NVIDIA Corporation
- **Key Contribution:** Provides CUDA-accelerated inference on edge devices (Jetson Nano, Orin Nano, AGX Orin) with TensorRT optimization for real-time DMS applications.
- **URL:** <https://developer.nvidia.com/embedded/jetpack>

### [R15] TensorRT — Model Optimization

- **Title:** NVIDIA TensorRT Documentation
- **Publisher:** NVIDIA Corporation
- **Key Contribution:** INT8 quantization, layer fusion, and kernel auto-tuning for maximizing inference throughput on NVIDIA GPUs.
- **URL:** <https://developer.nvidia.com/tensorrt>

### [R16] OpenVINO Toolkit

- **Title:** OpenVINO™ Toolkit Documentation
- **Publisher:** Intel Corporation
- **Key Contribution:** Hardware-agnostic model optimization for Intel CPUs, iGPUs, and NPUs. Supports ONNX model import and INT8 quantization.
- **URL:** <https://docs.openvino.ai>

### [R17] Google Coral & LiteRT (TFLite)

- **Title:** Google Coral Edge TPU / LiteRT Documentation
- **Publisher:** Google
- **Key Contribution:** Edge TPU accelerator for TFLite models; LiteRT (formerly TFLite) for lightweight model deployment on mobile and embedded platforms.
- **URL:** <https://coral.ai/docs/> | <https://ai.google.dev/edge/litert>

### [R18] ESP32-CAM Module

- **Title:** ESP32-CAM Technical Reference
- **Publisher:** Espressif Systems
- **Key Contribution:** Ultra-low-cost ($5–15) Wi-Fi/BLE-enabled microcontroller with OV2640 camera module. Suitable for image capture and lightweight IoT gateway tasks.
- **URL:** <https://www.espressif.com/en/products/devkits>

---

## Software Libraries & Frameworks

### [R19] dlib Library

- **URL:** <http://dlib.net/>
- **Key Use:** 68-point facial landmark detection using shape_predictor_68_face_landmarks.dat

### [R20] Google MediaPipe

- **URL:** <https://mediapipe.dev/>
- **Key Use:** 468-point Face Mesh for detailed facial landmark tracking, optimized for real-time CPU inference.

### [R21] OpenCV

- **URL:** <https://opencv.org/>
- **Key Use:** Computer vision primitives — image I/O, color space conversion, CLAHE, solvePnP for head pose estimation.

### [R22] ONNX Runtime

- **URL:** <https://onnxruntime.ai/>
- **Key Use:** Cross-platform inference engine supporting multiple hardware backends (CPU, GPU, TensorRT, OpenVINO).

---

## Cloud & IoT Architecture

### [R23] AWS IoT Core / Greengrass

- **URL:** <https://aws.amazon.com/iot/>
- **Key Use:** Fleet management, OTA firmware updates, telemetry aggregation, and cloud-based model retraining for DMS.

### [R24] MQTT Protocol

- **URL:** <https://mqtt.org/>
- **Key Use:** Lightweight pub/sub messaging protocol for transmitting fatigue event metadata from vehicle edge devices to cloud backends.

### [R25] Firebase / Google Cloud IoT

- **URL:** <https://firebase.google.com/>
- **Key Use:** Real-time database, cloud functions, and device management for IoT-based DMS architectures.

---

*Last updated: July 2026*
