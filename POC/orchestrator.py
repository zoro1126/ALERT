import cv2
import time
import numpy as np
from typing import Dict, Any, Optional

from facial_feature_extractor import FacialFeatureExtractor, open_webcam
from landmark_preprocessor import LandmarkPreprocessor
from model_classifier import FatigueClassifier


class FatigueDetectionOrchestrator:
    """
    Main Orchestrator for Driver Fatigue Detection System.
    Integrates:
      - FacialFeatureExtractor (MediaPipe 3D Landmark & Head Pose PnP)
      - LandmarkPreprocessor (Temporal smoothing, EAR/MAR calibration, F1-F9 features)
      - FatigueClassifier (Trained Random Forest / XGBoost ML Model)
      - Real-Time HUD Dashboard & Alert Engine
    """

    def __init__(
        self,
        target_fps: float = 30.0,
        calibration_duration_sec: float = 10.0
    ):
        self.target_fps = target_fps
        self.calibration_duration_sec = calibration_duration_sec

        self.extractor = FacialFeatureExtractor()
        self.preprocessor = LandmarkPreprocessor(
            fps=target_fps,
            calibration_duration_sec=calibration_duration_sec
        )
        self.classifier = FatigueClassifier()

        # FPS calculation state
        self.prev_frame_time = time.time()
        self.current_fps = target_fps

        # Calibration timer state
        self.calibration_start_time: Optional[float] = None
        self.calibration_elapsed_sec: float = 0.0

    def draw_hud(
        self,
        frame: np.ndarray,
        metrics: Optional[Dict[str, Any]],
        processed: Optional[Dict[str, Any]],
        prediction: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Renders a sleek, high-performance Driver Monitoring HUD onto the video frame.
        Displays ML Model predictions, confidence probabilities, calibration progress,
        facial landmarks, EAR/MAR signals, PERCLOS, and Head Pose.
        """
        h, w, _ = frame.shape

        # Semi-transparent ROI blending (top & bottom bars only)
        top_roi = frame[0:145, 0:w]
        top_bg = np.full_like(top_roi, 20, dtype=np.uint8)
        frame[0:145, 0:w] = cv2.addWeighted(top_roi, 0.35, top_bg, 0.65, 0)

        bot_roi = frame[h - 55:h, 0:w]
        bot_bg = np.full_like(bot_roi, 15, dtype=np.uint8)
        frame[h - 55:h, 0:w] = cv2.addWeighted(bot_roi, 0.35, bot_bg, 0.65, 0)

        # Calculate live FPS
        curr_time = time.time()
        time_diff = curr_time - self.prev_frame_time
        if time_diff > 0:
            self.current_fps = 0.85 * self.current_fps + 0.15 * (1.0 / time_diff)
        self.prev_frame_time = curr_time

        # FPS & System Status Header
        status_color = (0, 255, 0) if metrics else (0, 0, 255)
        status_text = "TRACKING: ACTIVE" if metrics else "SEARCHING FOR DRIVER..."
        cv2.putText(frame, f"ALERT DMS | FPS: {self.current_fps:.1f}", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (w - 230, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # -------------------------------------------------------------
        # CALIBRATION TIMER & PROGRESS BAR
        # -------------------------------------------------------------
        if processed and not processed["is_calibrated"]:
            if self.calibration_start_time is None:
                self.calibration_start_time = time.time()

            self.calibration_elapsed_sec = min(
                self.calibration_duration_sec,
                time.time() - self.calibration_start_time
            )

            progress_ratio = self.calibration_elapsed_sec / self.calibration_duration_sec
            time_remaining = max(0.0, self.calibration_duration_sec - self.calibration_elapsed_sec)

            # Draw Calibration Alert Banner
            calib_text = f"CALIBRATING BASELINE: {time_remaining:.1f}s (LOOK FORWARD)"
            cv2.putText(frame, calib_text, (15, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

            # Calibration Progress Bar
            bar_x, bar_y, bar_w, bar_h = 15, 62, 380, 10
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
            fill_w = int(bar_w * progress_ratio)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 215, 255), -1)

        elif processed and processed["is_calibrated"]:
            calib_complete_text = (
                f"CALIBRATED BASELINE -> EAR THRESH: {processed['ear_threshold']:.3f} | "
                f"MAR THRESH: {processed['mar_threshold']:.3f}"
            )
            cv2.putText(frame, calib_complete_text, (15, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 128), 2)

        # -------------------------------------------------------------
        # METRICS & STATS DISPLAY
        # -------------------------------------------------------------
        if metrics and processed and prediction:
            feats = processed["features"]
            landmarks = metrics["landmarks_2d"]

            # Draw Facial Landmarks Mesh Overlays (Eyes: Green, Mouth: Orange)
            for idx in FacialFeatureExtractor.LEFT_EYE_LANDMARKS + FacialFeatureExtractor.RIGHT_EYE_LANDMARKS:
                cv2.circle(frame, tuple(landmarks[idx]), 2, (0, 255, 0), -1)

            for idx in FacialFeatureExtractor.MOUTH_LANDMARKS:
                cv2.circle(frame, tuple(landmarks[idx]), 2, (0, 165, 255), -1)

            # Stats Column 1: Eye & Mouth Signals
            ear_color = (0, 255, 0) if feats["F1_EAR"] >= processed["ear_threshold"] else (0, 0, 255)
            mar_color = (0, 255, 0) if feats["F6_MAR"] <= processed["mar_threshold"] else (0, 165, 255)

            cv2.putText(frame, f"EAR: {feats['F1_EAR']:.2f} (Corr: {processed['corrected_ear']:.2f})",
                        (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.52, ear_color, 2)
            cv2.putText(frame, f"MAR: {feats['F6_MAR']:.2f} | MOE: {feats['F9_MOE']:.2f}",
                        (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.52, mar_color, 2)
            cv2.putText(frame, f"EAR_std: {feats['F2_EAR_std']:.3f}",
                        (15, 137), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

            # Stats Column 2: Temporal Analytics (PERCLOS, Blinks, Yawns)
            perclos_color = (0, 255, 0)
            if feats["F3_PERCLOS"] > 25.0:
                perclos_color = (0, 0, 255)
            elif feats["F3_PERCLOS"] > 15.0:
                perclos_color = (0, 215, 255)

            cv2.putText(frame, f"PERCLOS (60s): {feats['F3_PERCLOS']:.1f}%",
                        (260, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.52, perclos_color, 2)
            cv2.putText(frame, f"Blinks: {feats['F4_Blink_Rate']:.0f}/min ({feats['F5_Blink_Duration_ms']:.0f}ms)",
                        (260, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

            # Stats Column 3: Pose & Yawn Count
            cv2.putText(frame, f"Pitch: {feats['F8_Head_Pitch']:.1f}° | Roll: {metrics['head_pose']['roll']:.1f}°",
                        (w - 240, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
            cv2.putText(frame, f"Yawns (5m): {feats['F7_Yawn_Count']}",
                        (w - 240, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

            # -------------------------------------------------------------
            # ML MODEL CLASSIFIER OUTPUT & PROBABILITIES
            # -------------------------------------------------------------
            class_id = prediction["class_id"]
            label = prediction["label"]
            conf = prediction["confidence"]
            probs = prediction["probabilities"]
            is_override = prediction["is_override"]

            # Set Banner Colors by Predicted Class
            if class_id == 0:  # L0 Alert
                banner_color = (0, 255, 0)
                display_status = "L0 ALERT - DRIVER ALERT"
            elif class_id == 1:  # L1 Mild
                banner_color = (0, 215, 255)
                display_status = "L1 MILD FATIGUE - EARLY DROWSINESS"
            elif class_id == 2:  # L2 Moderate
                banner_color = (0, 140, 255)
                display_status = "L2 MODERATE FATIGUE - WARNING ALARM"
            else:  # L3 Severe
                banner_color = (0, 0, 255)
                display_status = "L3 SEVERE FATIGUE - CRITICAL EMERGENCY!"

            if is_override:
                display_status += " (SAFETY OVERRIDE)"

            # Draw Bottom Alert Text & Confidence
            cv2.putText(frame, f"ML CLASS: {display_status}", (15, h - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, banner_color, 2)

            conf_text = f"CONF: {conf*100:.1f}% | PROBS [L0:{probs[0]:.2f} L1:{probs[1]:.2f} L2:{probs[2]:.2f} L3:{probs[3]:.2f}]"
            cv2.putText(frame, conf_text, (15, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

            cv2.putText(frame, "PRESS 'Q' TO QUIT", (w - 170, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        return frame

    def run(self):
        """Runs the live webcam orchestration loop."""
        print("[INFO] Starting Driver Fatigue Detection Orchestration Pipeline...")
        cv2.setNumThreads(4)

        cap, cam_idx = open_webcam()

        if cap is None or not cap.isOpened():
            print("[WARNING] No active camera stream found.")
            print("[INFO] Testing pipeline with synthetic frame...")
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            metrics = self.extractor.process_frame(test_frame)
            processed = self.preprocessor.process(metrics) if metrics else None
            pred = self.classifier.predict(processed["feature_vector"], processed["features"]) if processed else None
            self.draw_hud(test_frame, metrics, processed, pred)
            self.extractor.close()
            print("[INFO] Test complete.")
            return

        try:
            cv2.namedWindow("ALERT: Automated Landmark based Eye and Response Tracker", cv2.WINDOW_NORMAL)
            print(f"[INFO] Camera stream active (Device Index {cam_idx}). Displaying ALERT HUD at 30 FPS...")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[WARNING] Empty frame received from camera stream.")
                    break

                # 1. Feature Extraction (MediaPipe 3D Landmarks + EAR/MAR/Pose)
                metrics = self.extractor.process_frame(frame)

                # 2. Landmark Preprocessing (Smoothing, Calibration & F1-F9 Feature Vector)
                processed = None
                prediction = None
                if metrics:
                    processed = self.preprocessor.process(metrics)

                    # 3. Classical ML Classifier Prediction (L0, L1, L2, L3 Multiclass Probabilities)
                    if processed:
                        prediction = self.classifier.predict(
                            processed["feature_vector"],
                            raw_features=processed["features"]
                        )

                # 4. Render High-Performance HUD Dashboard
                hud_frame = self.draw_hud(frame, metrics, processed, prediction)

                # 5. Display Live Stream Window
                cv2.imshow("Driver Fatigue Detection System (ML Powered)", hud_frame)

                # Exit key: 'q' or ESC
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("[INFO] User requested exit.")
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.extractor.close()
            print("[INFO] Orchestrator shutdown complete.")


if __name__ == "__main__":
    orchestrator = FatigueDetectionOrchestrator(
        target_fps=30.0,
        calibration_duration_sec=10.0
    )
    orchestrator.run()
