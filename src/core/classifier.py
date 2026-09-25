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

from core.model import FatigueClassifier
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

        self._model = self._load_model(Path(model_path))

    # ------------------------------------------------------------------
    def _load_model(self, path: Path) -> FatigueClassifier:
        if not path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {path}\n"
                "Train the model first:  python -m training.trainer"
            )
        ckpt = torch.load(path, map_location=self.device)
        cfg  = ckpt.get("config", {})
        model = FatigueClassifier(
            n_features = cfg.get("n_features", Config.N_FEATURES),
            hidden1    = cfg.get("hidden1",    Config.HIDDEN_SIZE),
            hidden2    = cfg.get("hidden2",    Config.HIDDEN_SIZE_2),
            fc_size    = cfg.get("fc_size",    Config.FC_SIZE),
            n_classes  = cfg.get("n_classes",  Config.N_CLASSES),
        ).to(self.device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model

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
        """Stack buffer → tensor → model forward → ClassifierResult."""
        window = np.stack(list(self._buffer), axis=0)  # (60, 8)
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
        """Fraction of window filled (0.0 – 1.0). Used for warm-up HUD."""
        return len(self._buffer) / self.window_size
