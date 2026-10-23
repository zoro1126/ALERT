"""
test_ensemble.py
================
Unit tests for Phase 4 EnsembleFatigueClassifier:
- Buffer accumulation & warm-up
- Probability soft-voting aggregation
- EMA temporal smoothing
- Two-tier debounced alarm state machine
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.classifier import EnsembleFatigueClassifier, EnsembleResult
from utils.config import Config


def test_ensemble_classifier_warmup():
    clf = EnsembleFatigueClassifier(model_paths=[], window_size=120, stride=15)
    dummy_feat = np.zeros(12, dtype=np.float32)

    # First 119 frames should return None
    for i in range(119):
        res = clf.update(dummy_feat)
        assert res is None
        assert clf.buffer_fill < 1.0

    # 120th frame should trigger inference
    res = clf.update(dummy_feat)
    assert res is not None
    assert isinstance(res, EnsembleResult)
    assert clf.buffer_fill == 1.0


def test_ensemble_debounced_state_machine():
    clf = EnsembleFatigueClassifier(model_paths=[], window_size=120, stride=15, fps=15.0)

    # Feed 120 frames to warm up
    for _ in range(120):
        clf.update(np.zeros(12, dtype=np.float32))

    # Test Alert condition
    clf._smoothed_probs = np.array([0.90, 0.05, 0.05], dtype=np.float32)
    res = clf._infer()
    assert res.state == 0
    assert "NORMAL" in res.state_name
    assert not res.alarm_active
    assert res.alarm_level == 0

    # Test Drowsy debounce (< 1.5s = < 22 frames -> MONITORING)
    clf._smoothed_probs = np.array([0.05, 0.10, 0.85], dtype=np.float32)
    res1 = clf._infer()  # 15 frames
    assert res1.alarm_level == 0
    assert not res1.alarm_active

    res2 = clf._infer()  # 30 frames >= 22 frames -> CRITICAL
    assert res2.alarm_level == 2
    assert res2.alarm_active
    assert res2.state == 2
    assert "CRITICAL" in res2.state_name

    # Alert resets state immediately
    clf._smoothed_probs = np.array([0.80, 0.10, 0.10], dtype=np.float32)
    res3 = clf._infer()
    assert res3.state == 0
    assert not res3.alarm_active
    assert res3.alarm_level == 0
