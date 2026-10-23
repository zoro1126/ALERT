"""
test_pipeline_integration.py
============================
End-to-end integration test verifying the full ALERT Phase 5 pipeline:
Calibrator -> FeatureExtractor (12 kinematic channels) -> EnsembleFatigueClassifier
-> Debounced Alarm Engine -> AlertManager -> HUD Overlay.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from core.alert_manager import AlertManager, AlertState
from core.calibrator import Calibrator
from core.classifier import EnsembleFatigueClassifier
from core.feature_extractor import FeatureExtractor
from ui import overlay as hud
from utils.config import Config


def test_full_pipeline_mock_stream(tmp_path):
    # 1. Calibrator verification with profile persistence
    calib_file = tmp_path / "test_calibration.json"
    cal = Calibrator(n_frames=10, save_path=calib_file)
    for _ in range(10):
        cal.add_frame(ear_avg=0.34, pitch=2.0, yaw=-1.5)
    assert cal.is_ready()
    thresh = cal.finalize()
    assert thresh > 0.0

    profile = Calibrator.load_profile(save_path=calib_file)
    assert profile is not None
    assert pytest.approx(profile["ear_base"], abs=1e-3) == 0.34
    assert pytest.approx(profile["pitch_base"], abs=1e-3) == 2.0
    assert pytest.approx(profile["yaw_base"], abs=1e-3) == -1.5

    # 2. Feature extractor baseline alignment
    extractor = FeatureExtractor(window_size=120)
    extractor.update_baseline(
        ear_base=profile["ear_base"],
        pitch_base=profile["pitch_base"],
        yaw_base=profile["yaw_base"],
    )
    assert extractor.ear_base == 0.34
    assert extractor.pitch_base == 2.0
    assert extractor.yaw_base == -1.5

    # 3. Ensemble Classifier loading and inference check
    ensemble = EnsembleFatigueClassifier(models_dir=str(_PROJECT_ROOT / "models"), device="cpu")
    assert len(ensemble._models) >= 3

    # Feed 120 normal frames
    dummy_normal_vec = np.array([
        1.0, 1.0, 1.0, 0.0,    # EAR L/R/avg rel, dEAR
        0.25, 0.0,             # MAR, dMAR
        0.0,                   # PERCLOS
        0.0, 0.0, 0.0,         # Pitch rel, dPitch, Yaw rel
        0.25, 0.15             # Blink rate, Blink dur
    ], dtype=np.float32)

    last_res = None
    for _ in range(120):
        last_res = ensemble.update(dummy_normal_vec)

    assert last_res is not None
    assert last_res.smoothed_probs.shape == (3,)
    assert pytest.approx(float(np.sum(last_res.smoothed_probs)), abs=1e-3) == 1.0

    # 4. AlertManager mapping check
    alert_mgr = AlertManager()
    state = alert_mgr.update_from_alarm_level(last_res.alarm_level)
    assert state in (AlertState.NORMAL, AlertState.WARNING, AlertState.CRITICAL)

    # Test direct level triggers
    assert alert_mgr.update_from_alarm_level(1) == AlertState.WARNING
    assert alert_mgr.update_from_alarm_level(2) == AlertState.CRITICAL
    assert alert_mgr.update_from_alarm_level(0) == AlertState.NORMAL

    # 5. HUD rendering verification
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    hud.draw_status_bar(
        dummy_frame,
        state=AlertState.NORMAL,
        label="NORMAL (Alert)",
        confidence=0.92,
        model_tag="9-Model Ensemble (GPU)",
    )
    hud.draw_prob_bars(dummy_frame, last_res.smoothed_probs)
    hud.draw_attention_strip(dummy_frame, last_res.attention)
    hud.draw_warmup_bar(dummy_frame, 0.75)
    hud.draw_feature_panel(
        dummy_frame,
        ear_l=0.34,
        ear_r=0.34,
        ear_avg=0.34,
        mar=0.25,
        perclos=0.02,
        pitch=2.0,
        yaw=-1.5,
        blink_rate=0.25,
        ear_threshold=0.255,
        d_ear=0.01,
        d_pitch=0.0,
        blink_dur=0.15,
    )
    # Check that frame pixels were written
    assert dummy_frame.sum() > 0
