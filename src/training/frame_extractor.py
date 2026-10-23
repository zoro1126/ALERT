"""
frame_extractor.py
==================
Extracts per-frame 8-feature vectors from a single UTA-RLDD video file.

Uses the MediaPipe Tasks API (mediapipe >= 0.10) with FaceLandmarker.

Pipeline (per video)
--------------------
1. Open video with cv2.VideoCapture
2. Compute a downsampling stride so we read ~TARGET_FPS frames per second
3. For each sampled frame:
   a. Run FaceLandmarker in IMAGE mode (one shot per frame, no tracking state)
   b. Call FeatureExtractor.extract(landmarks, frame_w, frame_h)
   c. Append the 8-element float32 vector to a list
4. Return a (N, 8) float32 numpy array

Notes
-----
- FeatureExtractor is reset at the start of every video so PERCLOS / blink
  buffers from one video don't bleed into the next.
- Frames where FaceLandmarker finds no face are skipped silently.
- FaceLandmarker is initialised once per call and closed afterwards.
- The .task model file path is read from Config.FACE_LANDMARKER_TASK if
  present, otherwise falls back to a default relative path.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from core.feature_extractor import (
    FeatureExtractor,
    HeadPoseEstimator,
    compute_ear,
    _LEFT_EYE,
    _RIGHT_EYE,
)
from utils.config import Config

# ---------------------------------------------------------------------------
# Resolve the .task model file
# ---------------------------------------------------------------------------

def _find_task_model() -> Path:
    """Locate face_landmarker.task in known fallback locations."""
    # 1. Config override (if added later)
    if hasattr(Config, "FACE_LANDMARKER_TASK"):
        p = Path(Config.FACE_LANDMARKER_TASK)
        if p.exists():
            return p

    # 2. Project root / misc/  (where the POC stored it)
    candidates = [
        Path(Config.PROJECT_ROOT) / "misc" / "face_landmarker.task",
        Path(Config.PROJECT_ROOT) / "face_landmarker.task",
        Path(Config.PROJECT_ROOT) / "POC_v0" / "face_landmarker.task",
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        "face_landmarker.task not found. Download it from:\n"
        "  https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task\n"
        "and place it at: <project_root>/misc/face_landmarker.task"
    )


# ---------------------------------------------------------------------------
# Landmark adapter: Tasks API → legacy-compatible object
# ---------------------------------------------------------------------------

class _LandmarkAdapter:
    """
    Wraps a mediapipe.tasks NormalizedLandmark list so it is compatible with
    the existing FeatureExtractor / HeadPoseEstimator code that accesses
    landmarks via  face_landmarks.landmark[idx].x / .y / .z
    """

    class _LM:
        __slots__ = ("x", "y", "z")
        def __init__(self, x: float, y: float, z: float) -> None:
            self.x, self.y, self.z = x, y, z

    def __init__(self, normalized_landmarks) -> None:
        # normalized_landmarks is a list of NormalizedLandmark objects
        self.landmark = [
            self._LM(lm.x, lm.y, lm.z) for lm in normalized_landmarks
        ]


# ---------------------------------------------------------------------------
# Baseline extraction (from Alert video)
# ---------------------------------------------------------------------------

def extract_subject_baseline(
    video_path: Path | str,
    target_fps: float = Config.TARGET_FPS,
    max_frames: int = 450,
) -> tuple[float, float, float]:
    """
    Extract personal resting baseline (EAR_base, Pitch_base, Yaw_base)
    from a subject's Alert (0.mov / 0.mp4) video.

    Parameters
    ----------
    video_path : Path | str
        Path to the subject's Alert video.
    target_fps : float
        Target downsampled sampling rate (default 15 FPS).
    max_frames : int
        Maximum sampled frames to inspect (~30s at 15 FPS).

    Returns
    -------
    (ear_base, pitch_base, yaw_base) : tuple[float, float, float]
    """
    video_path = Path(video_path)
    task_model = _find_task_model()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        warnings.warn(f"extract_subject_baseline: cannot open {video_path}", stacklevel=2)
        return (0.30, 0.0, 0.0)

    native_fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w:    int   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h:    int   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_every: int = max(1, round(native_fps / target_fps))

    VisionRunningMode = mp.tasks.vision.RunningMode
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    BaseOptions = mp.tasks.BaseOptions

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(task_model)),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=Config.MEDIAPIPE_MAX_FACES,
        min_face_detection_confidence=Config.MEDIAPIPE_DETECTION_CONF,
        min_face_presence_confidence=Config.MEDIAPIPE_DETECTION_CONF,
        min_tracking_confidence=Config.MEDIAPIPE_TRACKING_CONF,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    ear_samples: list[float] = []
    pitch_samples: list[float] = []
    yaw_samples: list[float] = []
    head_pose = HeadPoseEstimator()

    with FaceLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while len(ear_samples) < max_frames:
            # Fast frame skipping without decoding unselected frames
            if frame_idx % sample_every != 0:
                if not cap.grab():
                    break
                frame_idx += 1
                continue

            ret, bgr_frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )
            result = landmarker.detect(mp_image)

            if not result.face_landmarks:
                continue

            adapted = _LandmarkAdapter(result.face_landmarks[0])
            ear_l = compute_ear(adapted.landmark, _LEFT_EYE)
            ear_r = compute_ear(adapted.landmark, _RIGHT_EYE)
            ear_avg = (ear_l + ear_r) / 2.0
            pitch, yaw = head_pose.estimate(adapted, frame_w, frame_h)

            ear_samples.append(ear_avg)
            pitch_samples.append(pitch)
            yaw_samples.append(yaw)

    cap.release()

    if not ear_samples:
        warnings.warn(
            f"extract_subject_baseline: no face detected in {video_path.name}",
            stacklevel=2,
        )
        return (0.30, 0.0, 0.0)

    # 75th percentile of EAR reflects resting open eyes without blink dips
    ear_base = float(np.percentile(ear_samples, 75))
    ear_base = max(min(ear_base, 0.45), 0.18)

    pitch_base = float(np.median(pitch_samples))
    yaw_base = float(np.median(yaw_samples))

    return (ear_base, pitch_base, yaw_base)


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_features(
    video_path: Path | str,
    target_fps: float = Config.TARGET_FPS,
    model_complexity: int = Config.MEDIAPIPE_COMPLEXITY,
    ear_base: float = 0.30,
    pitch_base: float = 0.0,
    yaw_base: float = 0.0,
) -> np.ndarray:
    """
    Extract an (N, 12) float32 feature matrix from a single video file,
    calibrated against the subject's personal resting baseline.

    Parameters
    ----------
    video_path : Path or str
        Path to the .mov / .mp4 video.
    target_fps : float
        Target sampling rate. Frames are subsampled from native FPS.
        Defaults to ``Config.TARGET_FPS`` (15).
    model_complexity : int
        0 = lite, 1 = full. Defaults to ``Config.MEDIAPIPE_COMPLEXITY``.
    ear_base : float
        Subject resting EAR baseline (from Alert video).
    pitch_base : float
        Subject resting pitch baseline (degrees).
    yaw_base : float
        Subject resting yaw baseline (degrees).

    Returns
    -------
    np.ndarray of shape (N, 12), dtype float32
        One row per sampled frame where a face was detected.
        Returns an empty (0, 12) array on failure.
    """
    video_path = Path(video_path)
    task_model = _find_task_model()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        warnings.warn(f"extract_features: cannot open {video_path}", stacklevel=2)
        return np.empty((0, Config.N_FEATURES), dtype=np.float32)

    native_fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w:    int   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h:    int   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_every: int = max(1, round(native_fps / target_fps))

    VisionRunningMode = mp.tasks.vision.RunningMode
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    BaseOptions = mp.tasks.BaseOptions

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(task_model)),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=Config.MEDIAPIPE_MAX_FACES,
        min_face_detection_confidence=Config.MEDIAPIPE_DETECTION_CONF,
        min_face_presence_confidence=Config.MEDIAPIPE_DETECTION_CONF,
        min_tracking_confidence=Config.MEDIAPIPE_TRACKING_CONF,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    feature_rows: list[np.ndarray] = []
    extractor = FeatureExtractor(
        fps=target_fps,
        ear_base=ear_base,
        pitch_base=pitch_base,
        yaw_base=yaw_base,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            # Fast frame skipping without decoding unselected frames
            if frame_idx % sample_every != 0:
                if not cap.grab():
                    break
                frame_idx += 1
                continue

            ret, bgr_frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )
            result = landmarker.detect(mp_image)

            if not result.face_landmarks:
                continue  # no face — skip frame

            adapted = _LandmarkAdapter(result.face_landmarks[0])
            features = extractor.extract(adapted, frame_w, frame_h)
            feature_rows.append(features.to_numpy())

    cap.release()

    if not feature_rows:
        warnings.warn(
            f"extract_features: no face detected in {video_path.name}",
            stacklevel=2,
        )
        return np.empty((0, Config.N_FEATURES), dtype=np.float32)

    return np.stack(feature_rows, axis=0)   # shape (N, 12)
