"""
Central configuration for the ALERT system.

All tunable parameters live here. Import Config wherever needed —
never hardcode paths, thresholds, or hyperparameters in other modules.
"""

from pathlib import Path

# Resolve project root relative to this file's location:
# src/utils/config.py → src/utils → src → project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Config:
    # ------------------------------------------------------------------ #
    # Paths                                                                #
    # ------------------------------------------------------------------ #
    PROJECT_ROOT: Path = _PROJECT_ROOT

    DATASET_ROOT: Path = _PROJECT_ROOT / "uta-reallife-drowsiness-dataset"
    CACHE_DIR: Path = _PROJECT_ROOT / "cache"
    MODEL_DIR: Path = _PROJECT_ROOT / "models"
    LOG_DIR: Path = _PROJECT_ROOT / "logs"

    MODEL_PATH: Path = MODEL_DIR / "alert_model_best.pt"
    TRAINING_LOG: Path = LOG_DIR / "training_metrics.json"
    ALERT_LOG: Path = LOG_DIR / "alerts.json"

    # Per-user calibration stored outside the repo
    CALIBRATION_PATH: Path = Path.home() / ".alert" / "calibration.json"

    # ------------------------------------------------------------------ #
    # Dataset label mapping (derived from video filename stem)             #
    # ------------------------------------------------------------------ #
    # UTA-RLDD: filename stem → class index
    LABEL_MAP: dict[str, int] = {
        "0": 0,   # Alert
        "5": 1,   # Low Vigilant
        "10": 2,  # Drowsy
    }
    CLASS_NAMES: list[str] = ["Alert", "Low Vigilant", "Drowsy"]
    N_CLASSES: int = 3

    # ------------------------------------------------------------------ #
    # Feature extraction                                                   #
    # ------------------------------------------------------------------ #
    TARGET_FPS: int = 15        # Downsample videos to this rate for extraction
    N_FEATURES: int = 8         # [EAR_L, EAR_R, EAR_avg, MAR, PERCLOS, pitch, yaw, blink_rate]

    # MediaPipe FaceMesh
    MEDIAPIPE_MAX_FACES: int = 1
    MEDIAPIPE_COMPLEXITY: int = 0           # 0 = lightest/fastest
    MEDIAPIPE_DETECTION_CONF: float = 0.5
    MEDIAPIPE_TRACKING_CONF: float = 0.5

    # ------------------------------------------------------------------ #
    # Sliding window                                                       #
    # ------------------------------------------------------------------ #
    WINDOW_SIZE: int = 60       # frames per sequence fed to model (~2s at 30fps)
    STRIDE: int = 15            # sliding window stride during feature extraction

    # ------------------------------------------------------------------ #
    # Fatigue thresholds (fallbacks — overridden by adaptive calibration)  #
    # ------------------------------------------------------------------ #
    EAR_THRESHOLD: float = 0.25         # below → eye considered closed
    MAR_THRESHOLD: float = 0.60         # above → yawn detected
    PERCLOS_WARNING: float = 0.15       # 15 % closure in window → Warning
    PERCLOS_CRITICAL: float = 0.30      # 30 % closure in window → Critical
    HEAD_PITCH_THRESHOLD: float = 20.0  # degrees downward → Head Drop alert

    # Adaptive calibration multipliers
    EAR_CALIB_MULTIPLIER: float = 0.75  # threshold = baseline * this
    MAR_CALIB_MULTIPLIER: float = 2.00  # yawn threshold = baseline * this
    CALIBRATION_SECONDS: int = 10       # duration of baseline capture

    # ------------------------------------------------------------------ #
    # Alert timing                                                         #
    # ------------------------------------------------------------------ #
    WARNING_DURATION: float = 1.5   # seconds of low-vigilant state → Warning
    CRITICAL_DURATION: float = 2.5  # seconds of drowsy state → Critical
    ALERT_DEBOUNCE_WARNING: float = 3.0   # min seconds between Warning beeps
    ALERT_DEBOUNCE_CRITICAL: float = 1.0  # min seconds between Critical alarms

    # ------------------------------------------------------------------ #
    # Model architecture                                                   #
    # ------------------------------------------------------------------ #
    HIDDEN_SIZE: int = 64       # BiGRU first layer hidden units
    HIDDEN_SIZE_2: int = 32     # BiGRU second layer hidden units
    FC_SIZE: int = 64           # Fully-connected layer size
    DROPOUT_RNN: float = 0.3
    DROPOUT_FC: float = 0.2

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #
    BATCH_SIZE: int = 64
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    MAX_EPOCHS: int = 100
    PATIENCE: int = 10          # early stopping on val macro-F1
    COSINE_T_MAX: int = 50      # CosineAnnealingLR period

    # ------------------------------------------------------------------ #
    # Camera / real-time                                                   #
    # ------------------------------------------------------------------ #
    CAMERA_INDEX: int = 0
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    TARGET_FRAME_MS: int = 50   # max ms per frame (~20 FPS floor)

    # ------------------------------------------------------------------ #
    # Misc                                                                 #
    # ------------------------------------------------------------------ #
    VERSION: str = "0.1.0"
    RANDOM_SEED: int = 42
