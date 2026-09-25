"""
dashboard.py
============
Real-time driver fatigue detection loop.

Wires together:
  Camera → MediaPipe FaceLandmarker → FeatureExtractor → RealTimeClassifier
  → AlertManager → Overlay → Display

Run via:  python src/run_alert.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import mediapipe as mp

from core.alert_manager import AlertManager, AlertState
from core.calibrator import Calibrator
from core.classifier import RealTimeClassifier
from core.feature_extractor import FeatureExtractor
from training.frame_extractor import _LandmarkAdapter, _find_task_model
from ui import overlay as hud
from utils.config import Config


# ---------------------------------------------------------------------------
# MediaPipe setup helpers
# ---------------------------------------------------------------------------

def _build_landmarker():
    """Create a FaceLandmarker in LIVE_STREAM mode for real-time use."""
    VisionRunningMode  = mp.tasks.vision.RunningMode
    FaceLandmarker     = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOpts = mp.tasks.vision.FaceLandmarkerOptions
    BaseOptions        = mp.tasks.BaseOptions

    opts = FaceLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=str(_find_task_model())),
        running_mode=VisionRunningMode.IMAGE,   # stateless per-frame
        num_faces=Config.MEDIAPIPE_MAX_FACES,
        min_face_detection_confidence=Config.MEDIAPIPE_DETECTION_CONF,
        min_face_presence_confidence=Config.MEDIAPIPE_DETECTION_CONF,
        min_tracking_confidence=Config.MEDIAPIPE_TRACKING_CONF,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(opts)


# ---------------------------------------------------------------------------
# Calibration phase
# ---------------------------------------------------------------------------

def _run_calibration(
    cap: cv2.VideoCapture,
    landmarker,
    extractor: FeatureExtractor,
) -> float:
    """
    Show a calibration countdown, collect open-eye EAR frames, return threshold.
    Skips if a saved calibration already exists.
    """
    saved = Calibrator.load()
    if saved is not None:
        print(f"[Calibrator] Loaded saved threshold: {saved:.4f}")
        extractor.update_threshold(saved)
        return saved

    cal = Calibrator()
    print("[Calibrator] Keep eyes open and look at the camera...")

    while not cal.is_ready():
        ret, bgr = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        h, w = bgr.shape[:2]

        if result.face_landmarks:
            adapted = _LandmarkAdapter(result.face_landmarks[0])
            features = extractor.extract(adapted, w, h)
            cal.add_frame(features.ear_avg)

        # Draw calibration UI
        progress = cal.progress()
        bgr_disp = bgr.copy()
        hud.draw_warmup_bar(bgr_disp, progress)
        cv2.putText(bgr_disp, "CALIBRATING — Keep eyes open",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        cv2.imshow("ALERT", bgr_disp)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    threshold = cal.finalize()
    extractor.update_threshold(threshold)
    print(f"[Calibrator] Threshold set to {threshold:.4f}")
    return threshold


# ---------------------------------------------------------------------------
# Main detection loop
# ---------------------------------------------------------------------------

def run(
    model_path: str | Path = Config.MODEL_PATH,
    camera_idx: int        = Config.CAMERA_INDEX,
    show_fps:   bool       = True,
) -> None:
    """
    Start the real-time fatigue detection dashboard.

    Parameters
    ----------
    model_path : path to trained .pt checkpoint
    camera_idx : OpenCV camera index (0 = default webcam)
    show_fps   : draw FPS counter on HUD
    """
    # --- Camera ---
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_idx}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Components ---
    extractor  = FeatureExtractor(fps=float(Config.TARGET_FPS))
    classifier = RealTimeClassifier(model_path=model_path)
    alert_mgr  = AlertManager()

    with _build_landmarker() as landmarker:

        # --- Calibration ---
        ear_threshold = _run_calibration(cap, landmarker, extractor)

        # --- Detection loop ---
        prev_time = time.perf_counter()
        probs  = None
        attn   = None

        print("[ALERT] Starting detection — press Q to quit.")

        while True:
            ret, bgr = cap.read()
            if not ret:
                print("[ALERT] Camera lost — exiting.")
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if not result.face_landmarks:
                hud.draw_no_face(bgr)
            else:
                adapted  = _LandmarkAdapter(result.face_landmarks[0])
                features = extractor.extract(adapted, frame_w, frame_h)

                clf_result = classifier.update(features.to_numpy())
                if clf_result is not None:
                    probs = clf_result.probabilities
                    attn  = clf_result.attention
                    alert_mgr.update(clf_result.class_index)

                last = classifier.last_result
                if last is not None and probs is not None:
                    hud.draw_status_bar(
                        bgr, alert_mgr.state,
                        last.label_name,
                        float(probs[last.class_index]),
                    )
                    hud.draw_prob_bars(bgr, probs)
                    if attn is not None:
                        hud.draw_attention_strip(bgr, attn)
                else:
                    hud.draw_warmup_bar(bgr, classifier.buffer_fill)

                hud.draw_feature_panel(
                    bgr,
                    features.ear_left, features.ear_right, features.ear_avg,
                    features.mar, features.perclos,
                    features.head_pitch, features.head_yaw,
                    features.blink_rate,
                    ear_threshold=ear_threshold,
                )

            # FPS counter
            if show_fps:
                now = time.perf_counter()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                cv2.putText(bgr, f"FPS {fps:.0f}",
                            (frame_w - 80, frame_h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            cv2.imshow("ALERT", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[ALERT] Session ended.")


# ---------------------------------------------------------------------------
# Demo mode — no model required, threshold-based rule engine only
# ---------------------------------------------------------------------------

def run_demo(
    camera_idx: int  = Config.CAMERA_INDEX,
    show_fps:   bool = True,
) -> None:
    """
    Run the full pipeline WITHOUT a trained model.

    Alert state is determined by simple thresholds:
      EAR_avg < threshold for > WARNING_DURATION  →  WARNING
      PERCLOS  > PERCLOS_CRITICAL                 →  CRITICAL
      |head_pitch| > HEAD_PITCH_THRESHOLD         →  WARNING

    All MediaPipe, feature extraction, and HUD code runs normally.
    Use this to verify the camera / MediaPipe / HUD pipeline works
    before training a model.

    Press Q to quit.
    """
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        import glob
        available = sorted(glob.glob("/dev/video*"))
        hint = (f"  Available: {', '.join(available)} → try --camera 1"
                if available else "  No /dev/video* devices found.")
        raise RuntimeError(f"Cannot open camera index {camera_idx}\n{hint}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    extractor = FeatureExtractor(fps=float(Config.TARGET_FPS))
    alert_mgr = AlertManager()

    # Rule-based classifier state (replaces BiGRU)
    import time as _time
    low_ear_since: float | None = None

    print("[DEMO] Running in demo mode — no model required. Press Q to quit.")

    with _build_landmarker() as landmarker:
        ear_threshold = _run_calibration(cap, landmarker, extractor)

        prev_time = time.perf_counter()

        while True:
            ret, bgr = cap.read()
            if not ret:
                print("[DEMO] Camera lost — exiting.")
                break

            rgb      = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = landmarker.detect(mp_image)

            if not result.face_landmarks:
                hud.draw_no_face(bgr)
                low_ear_since = None
            else:
                adapted  = _LandmarkAdapter(result.face_landmarks[0])
                features = extractor.extract(adapted, frame_w, frame_h)

                # --- Simple rule-based state ---
                now = _time.monotonic()
                if features.ear_avg < ear_threshold:
                    if low_ear_since is None:
                        low_ear_since = now
                    elapsed = now - low_ear_since
                else:
                    low_ear_since = None
                    elapsed = 0.0

                if features.perclos > Config.PERCLOS_CRITICAL:
                    class_idx = 2   # Drowsy
                elif elapsed > Config.WARNING_DURATION or features.perclos > Config.PERCLOS_WARNING:
                    class_idx = 1   # Low Vigilant
                else:
                    class_idx = 0   # Alert

                alert_mgr.update(class_idx)
                label = Config.CLASS_NAMES[class_idx]

                # Mock confidence: 1 - perclos for alert, perclos for drowsy
                confidence = (1.0 - features.perclos) if class_idx == 0 else features.perclos
                confidence = max(0.4, min(confidence, 0.99))

                # Mock prob bars (rule-based, not softmax)
                probs = np.zeros(3, dtype=np.float32)
                probs[class_idx] = confidence
                probs[(class_idx + 1) % 3] = (1.0 - confidence) * 0.6
                probs[(class_idx + 2) % 3] = (1.0 - confidence) * 0.4
                probs /= probs.sum()

                hud.draw_status_bar(bgr, alert_mgr.state, label, float(probs[class_idx]))
                hud.draw_prob_bars(bgr, probs)
                hud.draw_feature_panel(
                    bgr,
                    features.ear_left, features.ear_right, features.ear_avg,
                    features.mar, features.perclos,
                    features.head_pitch, features.head_yaw,
                    features.blink_rate,
                    ear_threshold=ear_threshold,
                )

                # Demo mode label
                cv2.putText(bgr, "DEMO MODE — no model",
                            (frame_w // 2 - 110, frame_h - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 255), 1)

            if show_fps:
                now_t = time.perf_counter()
                fps = 1.0 / max(now_t - prev_time, 1e-6)
                prev_time = now_t
                cv2.putText(bgr, f"FPS {fps:.0f}",
                            (frame_w - 80, frame_h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            cv2.imshow("ALERT — Demo", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[DEMO] Session ended.")
