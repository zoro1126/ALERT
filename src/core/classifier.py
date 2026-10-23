"""
classifier.py
=============
Real-time fatigue classifier: maintains a rolling feature window and runs
the BiGRU model whenever the buffer is full.

Responsibilities
----------------
1. Accept one 8-feature vector per frame from FeatureExtractor.
2. Maintain a sliding deque of WINDOW_SIZE (60) frames.
3. When the deque is full (and every STRIDE frames thereafter), stack it
   into a (1, 60, 8) tensor and run FatigueClassifier.forward().
4. Return (class_index, probabilities, attention_weights) or None if the
   buffer hasn't warmed up yet.

Usage
-----
clf = RealTimeClassifier(model_path="models/alert_model_best.pt")
# per frame:
result = clf.update(feature_vector)   # np.ndarray (8,) float32
if result is not None:
    class_idx, probs, attn = result
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from core.model import FatigueClassifier, build_model
from training.normalizer import FeatureNormalizer
from training.preprocess import preprocess_window
from utils.config import Config


class ClassifierResult(NamedTuple):
    """Output of a single model inference."""
    class_index: int        # 0=Alert, 1=LowVigilant, 2=Drowsy
    probabilities: np.ndarray  # shape (3,) float32, sums to 1
    attention: np.ndarray   # shape (60,) float32 — per-frame attention weights
    label_name: str         # human-readable class name


class RealTimeClassifier:
    """
    Rolling-window fatigue classifier for live video streams.

    Parameters
    ----------
    model_path : Path or str
        Path to the .pt checkpoint saved by trainer.py.
    device : str
        'cpu' or 'cuda'. Defaults to 'cpu' (required for <2 GB RAM target).
    window_size : int
        Rolling buffer length (frames). Must match training config (60).
    stride : int
        Run inference every `stride` new frames (15 = 1 second at 15 FPS).
    """

    def __init__(
        self,
        model_path:  Path | str = Config.MODEL_PATH,
        device:      str  = "cpu",
        window_size: int  = Config.WINDOW_SIZE,
        stride:      int  = Config.STRIDE,
    ) -> None:
        self.window_size = window_size
        self.stride      = stride
        self.device      = torch.device(device)

        self._buffer: deque[np.ndarray] = deque(maxlen=window_size)
        self._frames_since_inference: int = 0
        self._last_result: ClassifierResult | None = None

        self._model      = self._load_model(Path(model_path))
        self._normalizer = self._load_normalizer(Path(model_path))

    # ------------------------------------------------------------------
    def _load_model(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {path}\n"
                "Train the model first:  python -m training.trainer"
            )
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        cfg  = ckpt.get("config", {})
        arch = cfg.get("arch", "bigru")
        model_kwargs = {
            k: v for k, v in cfg.items()
            if k not in ("arch", "parameters", "n_features", "n_classes")
        }

        # Legacy checkpoint support: remap fc1 -> head.0, fc_out -> head.3 if needed
        state_dict = dict(ckpt.get("model_state", {}))
        if "fc1.weight" in state_dict and "head.0.weight" not in state_dict:
            state_dict["head.0.weight"] = state_dict.pop("fc1.weight")
            state_dict["head.0.bias"] = state_dict.pop("fc1.bias")
        if "fc_out.weight" in state_dict and "head.3.weight" not in state_dict:
            state_dict["head.3.weight"] = state_dict.pop("fc_out.weight")
            state_dict["head.3.bias"] = state_dict.pop("fc_out.bias")

        try:
            model = build_model(
                arch=arch,
                n_features=cfg.get("n_features", Config.N_FEATURES),
                n_classes=cfg.get("n_classes", Config.N_CLASSES),
                **model_kwargs,
            ).to(self.device)
            model.load_state_dict(state_dict)
        except Exception:
            # Fallback for legacy checkpoints
            model = FatigueClassifier(
                n_features=cfg.get("n_features", Config.N_FEATURES),
                hidden1=cfg.get("hidden1", Config.HIDDEN_SIZE),
                hidden2=cfg.get("hidden2", Config.HIDDEN_SIZE_2),
                fc_size=cfg.get("fc_size", Config.FC_SIZE),
                n_classes=cfg.get("n_classes", Config.N_CLASSES),
            ).to(self.device)
            model.load_state_dict(state_dict)

        model.eval()
        return model

    # ------------------------------------------------------------------
    def _load_normalizer(self, model_path: Path) -> FeatureNormalizer | None:
        """
        Load normalizer stats from checkpoint metadata or feature_stats.npz.
        """
        # First check checkpoint metadata
        try:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            if "normalizer_stats" in ckpt and isinstance(ckpt["normalizer_stats"], dict):
                norm = FeatureNormalizer.from_dict(ckpt["normalizer_stats"])
                print("[RealTimeClassifier] Normalizer loaded from checkpoint metadata.")
                return norm
        except Exception:
            pass

        # Fallback to feature_stats.npz
        stats_path = model_path.parent / "feature_stats.npz"
        if stats_path.exists():
            norm = FeatureNormalizer.load(stats_path)
            print(f"[RealTimeClassifier] Normalizer loaded from {stats_path}")
            return norm

        print(
            f"[RealTimeClassifier] WARNING: {stats_path} not found.\n"
            "  Features will NOT be normalized — predictions may be unreliable.\n"
            "  Re-train the model to generate feature_stats.npz."
        )
        return None

    # ------------------------------------------------------------------
    def update(self, feature_vec: np.ndarray) -> ClassifierResult | None:
        """
        Feed one (8,) feature vector. Returns a ClassifierResult when the
        buffer is full AND it is a stride boundary, else returns None.

        Always returns the last valid result via `.last_result` property.
        """
        self._buffer.append(feature_vec.astype(np.float32))
        self._frames_since_inference += 1

        # Not enough frames yet
        if len(self._buffer) < self.window_size:
            return None

        # Only run inference at stride boundaries
        if self._frames_since_inference < self.stride:
            return None

        self._frames_since_inference = 0
        result = self._infer()
        self._last_result = result
        return result

    # ------------------------------------------------------------------
    def _infer(self) -> ClassifierResult:
        """Stack buffer → clip → normalize → tensor → model forward → result."""
        window = np.stack(list(self._buffer), axis=0)  # (60, 8) float32

        # Apply the exact preprocessing pipeline used during training:
        #   1. Outlier clipping + window-relative median pose centering
        #   2. Z-score normalize using training-set statistics
        window = preprocess_window(window)
        if self._normalizer is not None:
            window = self._normalizer.transform_window(window)

        x = torch.from_numpy(window).unsqueeze(0).to(self.device)  # (1, 60, 8)

        with torch.no_grad():
            logits, attn_w = self._model(x)
            probs = F.softmax(logits, dim=-1)

        class_idx = int(probs.argmax(dim=-1).item())
        probs_np  = probs.squeeze(0).cpu().numpy()
        attn_np   = attn_w.squeeze(0).cpu().numpy()

        return ClassifierResult(
            class_index=class_idx,
            probabilities=probs_np,
            attention=attn_np,
            label_name=Config.CLASS_NAMES[class_idx],
        )

    # ------------------------------------------------------------------
    @property
    def last_result(self) -> ClassifierResult | None:
        """Most recent inference result (or None if buffer not warm yet)."""
        return self._last_result

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the rolling buffer (call at session start or camera change)."""
        self._buffer.clear()
        self._frames_since_inference = 0
        self._last_result = None

    # ------------------------------------------------------------------
    @property
    def buffer_fill(self) -> float:
        """Fraction of window filled (0.0 - 1.0). Used for warm-up HUD."""
        return len(self._buffer) / self.window_size


# ---------------------------------------------------------------------------
# Phase 4: 12-Model Ensemble Engine & Debounced State Machine
# ---------------------------------------------------------------------------

class EnsembleResult(NamedTuple):
    """Output of an ensemble inference pass with two-tier alarm states."""
    state: int                  # 0=Alert, 1=LowVigilant, 2=Drowsy
    state_name: str             # "NORMAL (Alert)", "WARNING (Low Vigilant)", "CRITICAL (Drowsy)"
    smoothed_probs: np.ndarray  # shape (3,) float32 after EMA
    raw_probs: np.ndarray       # shape (3,) float32 soft-voted ensemble average
    alarm_active: bool          # True if warning or critical alarm triggered
    alarm_level: int            # 0=None, 1=Yellow Visual + Soft Chime, 2=Red Visual + Loud Siren
    attention: np.ndarray       # shape (WINDOW_SIZE,) pooled attention weights
    buffer_fill: float          # 0.0 to 1.0


class EnsembleFatigueClassifier:
    """
    12-Model Heterogeneous Ensemble Inference Engine with Two-Tier Alarm State Machine.

    Features:
      - 3 Architectures: HybridCNNGRU, TCNClassifier, BiGRUAttention
      - 4 Holdout Folds: soft-voting probability fusion across all available models
      - EMA Temporal Smoothing: P_ema = 0.35 * P_raw + 0.65 * P_prev
      - Debounced Alarm Engine:
          * Alert >= 0.50 -> NORMAL (resets warning/critical counters)
          * Low Vigilant >= 0.50 for >= 3.0s (45 frames) -> WARNING (level 1)
          * Drowsy >= 0.45 for >= 1.5s (22 frames) -> CRITICAL (level 2)
    """

    def __init__(
        self,
        model_paths: list[Path | str] | None = None,
        models_dir: Path | str = "models",
        device: str = "cpu",
        window_size: int = Config.WINDOW_SIZE,
        stride: int = Config.STRIDE,
        fps: float = Config.TARGET_FPS,
        ema_alpha: float = 0.35,
    ) -> None:
        self.window_size = window_size
        self.stride = stride
        self.fps = fps
        self.ema_alpha = ema_alpha
        self.device = torch.device(device)

        self._buffer: deque[np.ndarray] = deque(maxlen=window_size)
        self._frames_since_inference: int = 0
        self._smoothed_probs: np.ndarray | None = None
        self._last_result: EnsembleResult | None = None

        # Two-tier debounce counters
        self._consecutive_low_vig_frames: int = 0
        self._consecutive_drowsy_frames: int = 0
        self._low_vig_thresh_frames = int(3.0 * fps)
        self._drowsy_thresh_frames = int(1.5 * fps)

        # Load models and companion normalizers
        self._models: list[nn.Module] = []
        self._normalizers: list[FeatureNormalizer | None] = []

        if model_paths is None:
            models_path_obj = Path(models_dir)
            model_paths = sorted(models_path_obj.glob("ensemble_*.pt"))
            if not model_paths:
                model_paths = sorted(models_path_obj.glob("*.pt"))

        for mp in model_paths:
            m, norm = self._load_model_and_normalizer(Path(mp))
            if m is not None:
                self._models.append(m)
                self._normalizers.append(norm)

    def _load_model_and_normalizer(self, path: Path):
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            cfg = ckpt.get("config", {})
            arch = ckpt.get("arch", cfg.get("arch", "bigru"))

            model = build_model(
                arch=arch,
                n_features=cfg.get("n_features", Config.N_FEATURES),
                n_classes=cfg.get("n_classes", Config.N_CLASSES),
            ).to(self.device)

            state_dict = ckpt.get("model_state_dict", ckpt.get("model_state", {}))
            model.load_state_dict(state_dict)
            model.eval()

            norm = FeatureNormalizer()
            if "normalizer_mean" in ckpt and "normalizer_std" in ckpt:
                norm.mean_ = np.array(ckpt["normalizer_mean"], dtype=np.float32)
                norm.std_ = np.array(ckpt["normalizer_std"], dtype=np.float32)
            else:
                stats_p = path.with_name(path.stem + "_stats.npz")
                if stats_p.exists():
                    norm = FeatureNormalizer.load(stats_p)
                else:
                    norm = None

            return model, norm
        except Exception as e:
            warnings.warn(f"Failed to load ensemble checkpoint {path.name}: {e}", stacklevel=2)
            return None, None

    def update(self, feature_vec: np.ndarray) -> EnsembleResult | None:
        self._buffer.append(feature_vec.astype(np.float32))
        self._frames_since_inference += 1

        if len(self._buffer) < self.window_size:
            return None

        if self._frames_since_inference < self.stride:
            return self._last_result

        self._frames_since_inference = 0
        result = self._infer()
        self._last_result = result
        return result

    def _infer(self) -> EnsembleResult:
        if not self._models and self._smoothed_probs is None:
            raw = np.array([0.9, 0.05, 0.05], dtype=np.float32)
            return EnsembleResult(0, "NORMAL (Alert)", raw, raw, False, 0, np.zeros(self.window_size, dtype=np.float32), 1.0)

        window = np.stack(list(self._buffer), axis=0)  # (120, 12)
        window_pre = preprocess_window(window)

        all_probs = []
        all_attns = []

        if self._models:
            with torch.no_grad():
                for model, norm in zip(self._models, self._normalizers):
                    w = norm.transform_window(window_pre) if norm is not None else window_pre
                    x_t = torch.from_numpy(w).unsqueeze(0).to(self.device)
                    logits, attn = model(x_t)
                    p = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                    all_probs.append(p)
                    all_attns.append(attn.squeeze(0).cpu().numpy())

            raw_probs = np.mean(all_probs, axis=0)
            pooled_attn = np.mean(all_attns, axis=0)

            # Online EMA smoothing
            if self._smoothed_probs is None:
                self._smoothed_probs = raw_probs.copy()
            else:
                self._smoothed_probs = (
                    self.ema_alpha * raw_probs + (1.0 - self.ema_alpha) * self._smoothed_probs
                )
        else:
            raw_probs = self._smoothed_probs.copy()
            pooled_attn = np.zeros(self.window_size, dtype=np.float32)

        p_alert, p_low_vig, p_drowsy = self._smoothed_probs

        # Debounced State Machine
        if p_alert >= 0.50:
            self._consecutive_low_vig_frames = 0
            self._consecutive_drowsy_frames = 0
            state = 0
            state_name = "NORMAL (Alert)"
            alarm_level = 0
            alarm_active = False
        elif p_drowsy >= 0.45:
            self._consecutive_drowsy_frames += self.stride
            self._consecutive_low_vig_frames = 0
            if self._consecutive_drowsy_frames >= self._drowsy_thresh_frames:
                state = 2
                state_name = "CRITICAL (Drowsy)"
                alarm_level = 2
                alarm_active = True
            else:
                state = 1
                state_name = "MONITORING (Eye closure detected)"
                alarm_level = 0
                alarm_active = False
        elif p_low_vig >= 0.50:
            self._consecutive_low_vig_frames += self.stride
            self._consecutive_drowsy_frames = 0
            if self._consecutive_low_vig_frames >= self._low_vig_thresh_frames:
                state = 1
                state_name = "WARNING (Low Vigilant)"
                alarm_level = 1
                alarm_active = True
            else:
                state = 0
                state_name = "MONITORING (Minor gaze drift)"
                alarm_level = 0
                alarm_active = False
        else:
            state = int(np.argmax(self._smoothed_probs))
            state_name = Config.CLASS_NAMES[state]
            alarm_level = 0
            alarm_active = False

        return EnsembleResult(
            state=state,
            state_name=state_name,
            smoothed_probs=self._smoothed_probs.copy(),
            raw_probs=raw_probs,
            alarm_active=alarm_active,
            alarm_level=alarm_level,
            attention=pooled_attn,
            buffer_fill=1.0,
        )

    def reset(self) -> None:
        self._buffer.clear()
        self._frames_since_inference = 0
        self._smoothed_probs = None
        self._last_result = None
        self._consecutive_low_vig_frames = 0
        self._consecutive_drowsy_frames = 0

    @property
    def buffer_fill(self) -> float:
        return len(self._buffer) / self.window_size

