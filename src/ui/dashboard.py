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
