"""
dashboard.py
============
Real-time driver fatigue detection loop.

Wires together:
  Camera -> MediaPipe FaceLandmarker -> FeatureExtractor -> EnsembleFatigueClassifier
  -> AlertManager -> Overlay -> Display

Run via:  python src/run_alert.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import mediapipe as mp

from core.alert_manager import AlertManager, AlertState
from core.calibrator import Calibrator
from core.classifier import RealTimeClassifier, EnsembleFatigueClassifier
from core.feature_extractor import FeatureExtractor, compute_ear, _LEFT_EYE, _RIGHT_EYE
from training.frame_extractor import _LandmarkAdapter, _find_task_model
from ui import overlay as hud
from utils.config import Config


# ---------------------------------------------------------------------------
# MediaPipe setup helpers
# ---------------------------------------------------------------------------

def _build_landmarker():
    """Create a FaceLandmarker in IMAGE mode for real-time stateless frame use."""
    VisionRunningMode  = mp.tasks.vision.RunningMode
    FaceLandmarker     = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOpts = mp.tasks.vision.FaceLandmarkerOptions
    BaseOptions        = mp.tasks.BaseOptions

    opts = FaceLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=str(_find_task_model())),
        running_mode=VisionRunningMode.IMAGE,
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
    force_recalibrate: bool = False,
) -> dict:
    """
    Show a calibration countdown, collect open-eye EAR and head posture frames.
    Skips if a saved calibration profile exists (unless force_recalibrate=True).
    """
    if not force_recalibrate:
        saved_profile = Calibrator.load_profile()
        if saved_profile is not None and "pitch_base" in saved_profile:
            print(
                f"[Calibrator] Loaded saved baseline profile:\n"
                f"  EAR_base   : {saved_profile['ear_base']:.4f}\n"
                f"  Threshold  : {saved_profile['ear_threshold']:.4f}\n"
                f"  Pitch_base : {saved_profile['pitch_base']:+.1f} deg\n"
                f"  Yaw_base   : {saved_profile['yaw_base']:+.1f} deg"
            )
            extractor.update_baseline(
                ear_base=saved_profile["ear_base"],
                pitch_base=saved_profile["pitch_base"],
                yaw_base=saved_profile["yaw_base"],
            )
            return saved_profile

    cal = Calibrator()
    print("[Calibrator] Starting 15s personal baseline calibration.")
    print("  Keep eyes naturally open and look straight ahead at the camera...")

    while not cal.is_ready():
        ret, bgr = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        h, w = bgr.shape[:2]
        live_ear = 0.0
        live_pitch = 0.0

        if result.face_landmarks:
            adapted = _LandmarkAdapter(result.face_landmarks[0])
            ear_l = compute_ear(adapted.landmark, _LEFT_EYE)
            ear_r = compute_ear(adapted.landmark, _RIGHT_EYE)
            live_ear = (ear_l + ear_r) / 2.0
            live_pitch, live_yaw = extractor._head_pose.estimate(adapted, w, h)
            cal.add_frame(live_ear, live_pitch, live_yaw)

        # Draw calibration UI
        progress = cal.progress()
        bgr_disp = bgr.copy()
        hud.draw_warmup_bar(bgr_disp, progress)

        remaining_sec = max(0, int((cal.n_frames - len(cal._samples)) / Config.TARGET_FPS))
        cv2.putText(bgr_disp, f"CALIBRATING BASELINE ({remaining_sec}s remaining)",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        cv2.putText(bgr_disp, "Look straight ahead with natural alert eyes",
                    (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        if live_ear > 0.0:
            cv2.putText(bgr_disp, f"Live EAR: {live_ear:.3f} | Pitch: {live_pitch:+.1f} deg",
                        (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1)

        cv2.imshow("ALERT", bgr_disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    threshold = cal.finalize()
    profile = Calibrator.load_profile() or {
        "ear_threshold": threshold,
        "ear_base": 0.30,
        "pitch_base": 0.0,
        "yaw_base": 0.0,
    }
    extractor.update_baseline(
        ear_base=profile["ear_base"],
        pitch_base=profile["pitch_base"],
        yaw_base=profile["yaw_base"],
    )
    extractor.reset()
    print(
        f"[Calibrator] Calibration complete!\n"
        f"  EAR_base   : {profile['ear_base']:.4f}\n"
        f"  Pitch_base : {profile['pitch_base']:+.1f} deg\n"
        f"  Saved to   : {Config.CALIBRATION_PATH}"
    )
    return profile


# ---------------------------------------------------------------------------
# Camera initialization helper
# ---------------------------------------------------------------------------

def _open_camera(camera_idx: int) -> cv2.VideoCapture:
    """
    Open cv2.VideoCapture for the given camera_idx.
    Falls back across standard indices (1, 0) if the requested index fails.
    """
    candidates = [camera_idx] + [idx for idx in (1, 0) if idx != camera_idx]
    for idx in candidates:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            if idx != camera_idx:
                print(f"[ALERT] Camera index {camera_idx} unavailable; fell back to camera index {idx}.")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
            return cap
        cap.release()

    import glob
    available = sorted(glob.glob("/dev/video*"))
    hint = (f"  Available: {', '.join(available)}"
            if available else "  No video devices found.")
    raise RuntimeError(f"Cannot open camera index {camera_idx} (tried {candidates})\n{hint}")


# ---------------------------------------------------------------------------
# Main detection loop
# ---------------------------------------------------------------------------

def run(
    model_path: str | Path | None = None,
    models_dir: str | Path = Config.MODEL_DIR,
    camera_idx: int = Config.CAMERA_INDEX,
    show_fps: bool = True,
    force_recalibrate: bool = False,
    device: str = "auto",
) -> None:
    """
    Start the real-time fatigue detection dashboard with the Heterogeneous Ensemble.

    Parameters
    ----------
    model_path        : optional specific .pt checkpoint (runs single-model mode)
    models_dir        : directory containing ensemble checkpoints (default: models/)
    camera_idx        : OpenCV camera index (0 = default webcam)
    show_fps          : draw FPS counter on HUD
    force_recalibrate : ignore saved calibration profile and re-run baseline setup
    device            : 'auto', 'cuda', or 'cpu'
    """
    # Device selection
    if device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = device

    # Open Camera
    cap = _open_camera(camera_idx)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Core components
    extractor = FeatureExtractor(fps=float(Config.TARGET_FPS))
    alert_mgr = AlertManager()

    # Model / Ensemble setup
    if model_path is not None and Path(model_path).is_file():
        print(f"[ALERT] Loading single checkpoint: {model_path} on {dev.upper()}")
        classifier = RealTimeClassifier(model_path=model_path, device=dev)
        is_ensemble = False
        model_badge = f"{Path(model_path).stem} ({dev.upper()})"
    else:
        print(f"[ALERT] Initializing Heterogeneous Ensemble from '{models_dir}' on {dev.upper()}...")
        classifier = EnsembleFatigueClassifier(models_dir=models_dir, device=dev)
        is_ensemble = True
        n_models = len(classifier._models)
        model_badge = f"{n_models}-Model Ensemble ({dev.upper()})"
        print(f"[ALERT] {n_models} models loaded into ensemble.")

    with _build_landmarker() as landmarker:
        # Calibration phase
        profile = _run_calibration(
            cap=cap,
            landmarker=landmarker,
            extractor=extractor,
            force_recalibrate=force_recalibrate,
        )

        prev_time = time.perf_counter()
        probs = None
        attn = None

        print("\n" + "=" * 60)
        print("  ALERT DRIVER FATIGUE DETECTION SYSTEM ACTIVE")
        print(f"  Mode     : {model_badge}")
        print(f"  Baseline : EAR={profile['ear_base']:.3f}, Pitch={profile['pitch_base']:+.1f} deg")
        print("  Press 'Q' in the camera window to exit.")
        print("=" * 60 + "\n")

        while True:
            ret, bgr = cap.read()
            if not ret:
                print("[ALERT] Camera stream disconnected — exiting.")
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if not result.face_landmarks:
                hud.draw_no_face(bgr)
            else:
                adapted = _LandmarkAdapter(result.face_landmarks[0])
                features = extractor.extract(adapted, frame_w, frame_h)
                feature_vec = features.to_numpy()

                clf_result = classifier.update(feature_vec)

                if is_ensemble and isinstance(classifier, EnsembleFatigueClassifier):
                    if clf_result is not None:
                        probs = clf_result.smoothed_probs
                        attn = clf_result.attention
                        alert_state = alert_mgr.update_from_alarm_level(clf_result.alarm_level)
                        label_name = clf_result.state_name
                        confidence = float(np.max(probs))
                    else:
                        alert_state = AlertState.NORMAL
                        label_name = "Warming up..."
                        confidence = 0.0
                else:
                    if clf_result is not None:
                        probs = clf_result.probabilities
                        attn = clf_result.attention
                        alert_state = alert_mgr.update(clf_result.class_index)
                        label_name = clf_result.label_name
                        confidence = float(probs[clf_result.class_index])
                    else:
                        alert_state = AlertState.NORMAL
                        label_name = "Warming up..."
                        confidence = 0.0

                if clf_result is not None and probs is not None:
                    hud.draw_status_bar(
                        bgr,
                        alert_state,
                        label_name,
                        confidence,
                        model_tag=model_badge,
                    )
                    hud.draw_prob_bars(bgr, probs)
                    if attn is not None:
                        hud.draw_attention_strip(bgr, attn)
                else:
                    fill = classifier.buffer_fill
                    hud.draw_warmup_bar(bgr, fill)
                    cv2.rectangle(bgr, (0, 0), (frame_w, 44), (0, 0, 0), -1)
                    cv2.putText(
                        bgr,
                        f"[WARMING UP] Window Buffer: {fill:.0%} ({int(fill * classifier.window_size)}/{classifier.window_size} frames)",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 220, 255),
                        2,
                    )

                hud.draw_feature_panel(
                    bgr,
                    ear_l=features.ear_left_rel * extractor.ear_base,
                    ear_r=features.ear_right_rel * extractor.ear_base,
                    ear_avg=features.ear_avg_rel * extractor.ear_base,
                    mar=features.mar,
                    perclos=features.perclos_calib,
                    pitch=features.head_pitch_rel,
                    yaw=features.head_yaw_rel,
                    blink_rate=features.blink_rate,
                    ear_threshold=extractor.ear_threshold,
                    d_ear=features.d_ear_dt,
                    d_pitch=features.d_pitch_dt,
                    blink_dur=features.blink_duration,
                )

            # FPS counter
            if show_fps:
                now = time.perf_counter()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                cv2.putText(
                    bgr,
                    f"FPS {fps:.0f}",
                    (frame_w - 85, frame_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (180, 180, 180),
                    1,
                )

            cv2.imshow("ALERT — Driver Fatigue Detection", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[ALERT] Session ended successfully.")


# ---------------------------------------------------------------------------
# Demo mode — no model required, threshold-based rule engine only
# ---------------------------------------------------------------------------

def run_demo(
    camera_idx: int = Config.CAMERA_INDEX,
    show_fps: bool = True,
    force_recalibrate: bool = False,
) -> None:
    """
    Run the full pipeline WITHOUT a trained model using threshold heuristics.
    """
    cap = _open_camera(camera_idx)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    extractor = FeatureExtractor(fps=float(Config.TARGET_FPS))
    alert_mgr = AlertManager()

    import time as _time
    low_ear_since: float | None = None

    print("[DEMO] Running in demo mode (rule-based). Press Q to quit.")

    with _build_landmarker() as landmarker:
        profile = _run_calibration(
            cap, landmarker, extractor, force_recalibrate=force_recalibrate
        )
        ear_threshold = profile["ear_threshold"]

        prev_time = time.perf_counter()

        while True:
            ret, bgr = cap.read()
            if not ret:
                print("[DEMO] Camera lost — exiting.")
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if not result.face_landmarks:
                hud.draw_no_face(bgr)
                low_ear_since = None
            else:
                adapted = _LandmarkAdapter(result.face_landmarks[0])
                features = extractor.extract(adapted, frame_w, frame_h)

                now = _time.monotonic()
                raw_ear = features.ear_avg_rel * extractor.ear_base
                if raw_ear < ear_threshold:
                    if low_ear_since is None:
                        low_ear_since = now
                    elapsed = now - low_ear_since
                else:
                    low_ear_since = None
                    elapsed = 0.0

                if features.perclos_calib > Config.PERCLOS_CRITICAL:
                    class_idx = 2
                elif elapsed > Config.WARNING_DURATION or features.perclos_calib > Config.PERCLOS_WARNING:
                    class_idx = 1
                else:
                    class_idx = 0

                alert_mgr.update(class_idx)
                label = Config.CLASS_NAMES[class_idx]

                confidence = (1.0 - features.perclos_calib) if class_idx == 0 else features.perclos_calib
                confidence = max(0.4, min(confidence, 0.99))

                probs = np.zeros(3, dtype=np.float32)
                probs[class_idx] = confidence
                probs[(class_idx + 1) % 3] = (1.0 - confidence) * 0.6
                probs[(class_idx + 2) % 3] = (1.0 - confidence) * 0.4
                probs /= probs.sum()

                hud.draw_status_bar(bgr, alert_mgr.state, label, float(probs[class_idx]), model_tag="Rule Heuristic")
                hud.draw_prob_bars(bgr, probs)
                hud.draw_feature_panel(
                    bgr,
                    ear_l=features.ear_left_rel * extractor.ear_base,
                    ear_r=features.ear_right_rel * extractor.ear_base,
                    ear_avg=raw_ear,
                    mar=features.mar,
                    perclos=features.perclos_calib,
                    pitch=features.head_pitch_rel,
                    yaw=features.head_yaw_rel,
                    blink_rate=features.blink_rate,
                    ear_threshold=ear_threshold,
                    d_ear=features.d_ear_dt,
                    d_pitch=features.d_pitch_dt,
                    blink_dur=features.blink_duration,
                )

                cv2.putText(bgr, "DEMO MODE — Heuristic Engine",
                            (frame_w // 2 - 120, frame_h - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 255), 1)

            if show_fps:
                now_t = time.perf_counter()
                fps = 1.0 / max(now_t - prev_time, 1e-6)
                prev_time = now_t
                cv2.putText(bgr, f"FPS {fps:.0f}",
                            (frame_w - 85, frame_h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            cv2.imshow("ALERT — Demo", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[DEMO] Session ended.")
