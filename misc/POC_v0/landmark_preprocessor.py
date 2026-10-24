import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

class LandmarkPreprocessor:
    """
    Landmark Preprocessor for Driver Fatigue Feature Extraction.
    
    Responsibilities:
    1. Temporal Smoothing (Moving Average / Median Filter) to eliminate jitter & sensor noise.
    2. Personalized Baseline Calibration (10–30s silent onboarding) to compute driver-specific EAR/MAR thresholds.
    3. Head-Pose EAR Correction (EAR_corrected = EAR / cos(|yaw|)) to prevent false alarms from perspective distortion.
    4. Temporal Sliding Window Aggregator to compute PERCLOS, Blink Rate, Blink Duration, Yawn Count, and EAR_std.
    """

    def __init__(
        self,
        fps: float = 30.0,
        calibration_duration_sec: float = 10.0,
        perclos_window_sec: float = 60.0,
        yawn_window_sec: float = 300.0,
        ear_smoothing_window: int = 5,
        default_ear_threshold: float = 0.20,
        default_mar_threshold: float = 0.50
    ):
        """
        Args:
            fps: Video frame rate (default 30.0 FPS).
            calibration_duration_sec: Duration in seconds to collect baseline stats (default 10s).
            perclos_window_sec: Sliding window for PERCLOS & Blink analytics (default 60s).
            yawn_window_sec: Sliding window for Yawn Count analytics (default 300s / 5 min).
            ear_smoothing_window: Frame window size for moving average smoothing (default 5 frames).
            default_ear_threshold: Fallback population EAR threshold.
            default_mar_threshold: Fallback population MAR threshold.
        """
        self.fps = max(1.0, fps)
        self.calibration_target_frames = int(calibration_duration_sec * self.fps)
        self.perclos_window_size = int(perclos_window_sec * self.fps)
        self.yawn_window_size = int(yawn_window_sec * self.fps)

        # 1. Temporal Smoothing Buffers
        self.ear_raw_buffer = deque(maxlen=ear_smoothing_window)
        self.mar_raw_buffer = deque(maxlen=ear_smoothing_window)

        # 2. Calibration State
        self.is_calibrated = False
        self.calibration_ear_samples: List[float] = []
        self.calibration_mar_samples: List[float] = []
        
        self.mu_ear: float = 0.0
        self.sigma_ear: float = 0.0
        self.ear_threshold: float = default_ear_threshold
        
        self.mu_mar: float = 0.0
        self.sigma_mar: float = 0.0
        self.mar_threshold: float = default_mar_threshold

        # 3. Temporal Window Buffers for Classical ML Features
        self.ear_history = deque(maxlen=self.perclos_window_size)
        self.mar_history = deque(maxlen=self.yawn_window_size)
        self.pitch_history = deque(maxlen=self.perclos_window_size)
        
        # Blink tracking state
        self.blink_active = False
        self.blink_start_frame = 0
        self.blink_durations_ms: deque = deque(maxlen=100) # last 100 blinks
        self.blink_timestamps: deque = deque(maxlen=100)   # frame indices of blinks
        
        # Yawn tracking state
        self.yawn_active = False
        self.yawn_start_frame = 0
        self.yawn_timestamps: deque = deque(maxlen=50)

        self.current_frame_idx = 0

    def smooth_signal(self, new_val: float, buffer: deque, method: str = "mean") -> float:
        """Applies moving average or median filter over a rolling frame buffer."""
        buffer.append(new_val)
        if method == "median":
            return float(np.median(buffer))
        return float(np.mean(buffer))

    def apply_head_pose_correction(self, raw_ear: float, yaw_deg: float, pitch_deg: float) -> float:
        """
        Corrects EAR for perspective foreshortening when head turns.
        Formula: EAR_corrected = EAR_raw / cos(|yaw|) when |yaw| > 10°
        """
        abs_yaw = abs(yaw_deg)
        if abs_yaw > 10.0:
            yaw_rad = np.radians(abs_yaw)
            cos_val = np.cos(yaw_rad)
            if cos_val > 0.1: # prevent division by zero / extreme blowing up
                return min(0.5, raw_ear / cos_val)
        return raw_ear

    def update_calibration(self, ear: float, mar: float) -> bool:
        """
        Collects non-blink open-eye samples during onboarding to compute driver-specific baseline thresholds.
        
        Returns:
            True when calibration completes on current frame, else False.
        """
        if self.is_calibrated:
            return True

        # Exclude obvious blinks during calibration collection (EAR < 0.12)
        if ear >= 0.12:
            self.calibration_ear_samples.append(ear)
            self.calibration_mar_samples.append(mar)

        if len(self.calibration_ear_samples) >= self.calibration_target_frames:
            # Phase 2: Compute Baseline Stats
            self.mu_ear = float(np.mean(self.calibration_ear_samples))
            self.sigma_ear = float(np.std(self.calibration_ear_samples))
            
            self.mu_mar = float(np.mean(self.calibration_mar_samples))
            self.sigma_mar = float(np.std(self.calibration_mar_samples))

            # Sigma-offset threshold derivation (Method A: μ - 1.5σ)
            derived_ear_thresh = self.mu_ear - (1.5 * self.sigma_ear)
            
            # Phase 3: Boundary Guard Rails (0.15 <= EAR_thresh <= 0.30)
            self.ear_threshold = max(0.15, min(0.30, derived_ear_thresh))
            
            # MAR Threshold (μ + 2.0σ, guarded between 0.40 and 0.70)
            derived_mar_thresh = self.mu_mar + (2.0 * self.sigma_mar)
            self.mar_threshold = max(0.40, min(0.70, derived_mar_thresh))

            # Reject calibration if noise is too high (σ_EAR > 0.06)
            if self.sigma_ear > 0.06:
                self.ear_threshold = 0.20  # Fallback to default
                self.mar_threshold = 0.50

            self.is_calibrated = True
            print(f"[CALIBRATION COMPLETE] EAR Threshold: {self.ear_threshold:.3f} (μ={self.mu_ear:.3f}, σ={self.sigma_ear:.3f}) | MAR Threshold: {self.mar_threshold:.3f}")
            return True

        return False

    def _update_online_drift(self, ear: float):
        """Slow Exponential Moving Average (EMA) baseline update for long drives."""
        if not self.is_calibrated:
            return
        # Only update baseline from clearly open eye frames
        if ear > self.ear_threshold * 1.2:
            alpha = 0.001
            self.mu_ear = (1.0 - alpha) * self.mu_ear + alpha * ear
            derived = self.mu_ear - (1.5 * self.sigma_ear)
            self.ear_threshold = max(0.15, min(0.30, derived))

    def _process_blink_and_yawn_events(self, ear: float, mar: float):
        """Tracks individual blink and yawn durations & frequencies."""
        frame_time_ms = 1000.0 / self.fps

        # Blink Detection Heuristic: EAR < threshold
        if ear < self.ear_threshold:
            if not self.blink_active:
                self.blink_active = True
                self.blink_start_frame = self.current_frame_idx
        else:
            if self.blink_active:
                self.blink_active = False
                blink_frames = self.current_frame_idx - self.blink_start_frame
                duration_ms = blink_frames * frame_time_ms
                
                # Valid blink duration (100ms to 800ms)
                if 100.0 <= duration_ms <= 800.0:
                    self.blink_durations_ms.append(duration_ms)
                    self.blink_timestamps.append(self.current_frame_idx)

        # Yawn Detection Heuristic: MAR > threshold sustained >= 2.0 seconds
        if mar > self.mar_threshold:
            if not self.yawn_active:
                self.yawn_active = True
                self.yawn_start_frame = self.current_frame_idx
        else:
            if self.yawn_active:
                self.yawn_active = False
                yawn_frames = self.current_frame_idx - self.yawn_start_frame
                yawn_duration_sec = yawn_frames / self.fps
                if yawn_duration_sec >= 2.0:
                    self.yawn_timestamps.append(self.current_frame_idx)

    def process(self, raw_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocesses raw frame metrics from FacialFeatureExtractor and returns computed features.

        Args:
            raw_metrics: Dictionary containing 'avg_ear', 'mar', 'head_pose' from extractor.

        Returns:
            Dictionary containing smoothed signals, baseline thresholds, and temporal ML feature vectors (F1-F9).
        """
        self.current_frame_idx += 1

        raw_ear = raw_metrics.get("avg_ear", 0.0)
        raw_mar = raw_metrics.get("mar", 0.0)
        pose = raw_metrics.get("head_pose", {"pitch": 0.0, "yaw": 0.0, "roll": 0.0})
        yaw = pose.get("yaw", 0.0)
        pitch = pose.get("pitch", 0.0)

        # 1. Temporal Smoothing
        smoothed_ear = self.smooth_signal(raw_ear, self.ear_raw_buffer, method="mean")
        smoothed_mar = self.smooth_signal(raw_mar, self.mar_raw_buffer, method="mean")

        # 2. Head-Pose Perspective Correction
        corrected_ear = self.apply_head_pose_correction(smoothed_ear, yaw, pitch)

        # 3. Baseline Calibration Update
        self.update_calibration(corrected_ear, smoothed_mar)
        self._update_online_drift(corrected_ear)

        # 4. Update Window Buffers
        self.ear_history.append(corrected_ear)
        self.mar_history.append(smoothed_mar)
        self.pitch_history.append(pitch)

        # 5. Process Discrete Events (Blinks & Yawns)
        self._process_blink_and_yawn_events(corrected_ear, smoothed_mar)

        # --- COMPUTE FEATURE UNIVERSE (F1 - F9) ---
        # F1: Instantaneous Corrected EAR
        f1_ear = corrected_ear

        # F2: EAR_std (rolling 5s window)
        ear_5s_count = int(min(len(self.ear_history), 5.0 * self.fps))
        f2_ear_std = float(np.std(list(self.ear_history)[-ear_5s_count:])) if ear_5s_count > 1 else 0.0

        # F3: PERCLOS (% of frames in 60s window where EAR < EAR_threshold)
        if len(self.ear_history) > 0:
            closed_frames = sum(1 for e in self.ear_history if e < self.ear_threshold)
            f3_perclos = (closed_frames / len(self.ear_history)) * 100.0
        else:
            f3_perclos = 0.0

        # F4: Blink Rate (blinks per minute in rolling 60s window)
        cutoff_60s_frame = self.current_frame_idx - self.perclos_window_size
        recent_blinks = sum(1 for t in self.blink_timestamps if t >= cutoff_60s_frame)
        f4_blink_rate = float(recent_blinks)  # already over 60s window

        # F5: Avg Blink Duration (ms) in 60s window
        if len(self.blink_durations_ms) > 0:
            f5_avg_blink_duration = float(np.mean(self.blink_durations_ms))
        else:
            f5_avg_blink_duration = 200.0  # default normal blink duration in ms

        # F6: Instantaneous MAR
        f6_mar = smoothed_mar

        # F7: Yawn Count in rolling 5 min window
        cutoff_300s_frame = self.current_frame_idx - self.yawn_window_size
        f7_yawn_count = sum(1 for t in self.yawn_timestamps if t >= cutoff_300s_frame)

        # F8: Instantaneous Head Pitch Angle (°)
        f8_head_pitch = pitch

        # F9: Mouth-over-Eye Ratio (MOE = MAR / EAR)
        f9_moe = f6_mar / f1_ear if f1_ear > 0 else 0.0

        # Output Feature Vector for Classical ML Classifiers (SVM / Random Forest / XGBoost)
        feature_vector = np.array([
            f1_ear, f2_ear_std, f3_perclos, f4_blink_rate,
            f5_avg_blink_duration, f6_mar, f7_yawn_count,
            f8_head_pitch, f9_moe
        ], dtype=np.float32)

        return {
            "is_calibrated": self.is_calibrated,
            "ear_threshold": self.ear_threshold,
            "mar_threshold": self.mar_threshold,
            "smoothed_ear": smoothed_ear,
            "corrected_ear": corrected_ear,
            "smoothed_mar": smoothed_mar,
            "features": {
                "F1_EAR": f1_ear,
                "F2_EAR_std": f2_ear_std,
                "F3_PERCLOS": f3_perclos,
                "F4_Blink_Rate": f4_blink_rate,
                "F5_Blink_Duration_ms": f5_avg_blink_duration,
                "F6_MAR": f6_mar,
                "F7_Yawn_Count": f7_yawn_count,
                "F8_Head_Pitch": f8_head_pitch,
                "F9_MOE": f9_moe
            },
            "feature_vector": feature_vector
        }


if __name__ == "__main__":
    print("[INFO] Initializing LandmarkPreprocessor...")
    preprocessor = LandmarkPreprocessor(fps=30.0, calibration_duration_sec=3.0)

    # Test with synthetic landmark metrics stream
    print("[INFO] Processing 100 synthetic frame metrics...")
    for f in range(100):
        # Simulate normal driver for 60 frames, then drowsy driver (EAR drops, MAR rises)
        if f < 60:
            sim_ear = 0.28 + np.random.normal(0, 0.01)
            sim_mar = 0.20 + np.random.normal(0, 0.01)
            sim_yaw = np.random.normal(0, 2)
        else:
            sim_ear = 0.16 + np.random.normal(0, 0.01)  # Drowsy low EAR
            sim_mar = 0.55 + np.random.normal(0, 0.02)  # Yawning high MAR
            sim_yaw = 15.0  # Head turned slightly

        raw_metrics = {
            "avg_ear": float(sim_ear),
            "mar": float(sim_mar),
            "head_pose": {"pitch": -5.0, "yaw": float(sim_yaw), "roll": 0.0}
        }

        output = preprocessor.process(raw_metrics)

        if f == 80:
            print(f"\n[FRAME {f} SAMPLE PREPROCESSED OUTPUT]")
            print(f"Is Calibrated: {output['is_calibrated']}")
            print(f"Calibrated EAR Threshold: {output['ear_threshold']:.3f}")
            print(f"Calibrated MAR Threshold: {output['mar_threshold']:.3f}")
            print(f"Corrected EAR: {output['corrected_ear']:.3f}")
            print("Feature Vector (F1-F9):", np.round(output['feature_vector'], 3))
            print("Feature Details:", output['features'])

    print("\n[INFO] LandmarkPreprocessor test completed successfully.")
