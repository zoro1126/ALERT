"""
test_calibrator.py
==================
Unit tests for Calibrator:
- Baseline accumulation (EAR, pitch, yaw)
- Threshold calculation
- Profile serialization & reloading
- Reset behavior
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.calibrator import Calibrator


def test_calibrator_accumulation_and_finalize(tmp_path: Path):
    save_file = tmp_path / "calibration.json"
    n_frames = 10
    cal = Calibrator(n_frames=n_frames, multiplier=0.75, save_path=save_file)

    for i in range(n_frames):
        assert not cal.is_ready()
        cal.add_frame(ear_avg=0.30, pitch=-5.0, yaw=10.0)

    assert cal.is_ready()
    assert cal.progress() == 1.0

    threshold = cal.finalize()
    assert pytest.approx(threshold, rel=1e-3) == 0.30 * 0.75

    assert save_file.exists()
    data = json.loads(save_file.read_text())
    assert data["ear_threshold"] == pytest.approx(0.225, rel=1e-3)
    assert data["ear_open_mean"] == pytest.approx(0.30, rel=1e-3)
    assert data["pitch_base"] == pytest.approx(-5.0, rel=1e-3)
    assert data["yaw_base"] == pytest.approx(10.0, rel=1e-3)
    assert data["n_frames"] == 10


def test_calibrator_load_and_profile(tmp_path: Path):
    save_file = tmp_path / "calibration.json"
    cal = Calibrator(n_frames=5, multiplier=0.75, save_path=save_file)
    for _ in range(5):
        cal.add_frame(0.32, pitch=-2.5, yaw=4.0)
    thresh = cal.finalize()

    # Test load threshold
    loaded_thresh = Calibrator.load(save_file)
    assert loaded_thresh == pytest.approx(thresh, rel=1e-3)

    # Test load profile
    profile = Calibrator.load_profile(save_file)
    assert profile is not None
    assert profile["ear_threshold"] == pytest.approx(thresh, rel=1e-3)
    assert profile["ear_base"] == pytest.approx(0.32, rel=1e-3)
    assert profile["pitch_base"] == pytest.approx(-2.5, rel=1e-3)
    assert profile["yaw_base"] == pytest.approx(4.0, rel=1e-3)


def test_calibrator_reset(tmp_path: Path):
    save_file = tmp_path / "calibration.json"
    cal = Calibrator(n_frames=5, save_path=save_file)
    for _ in range(3):
        cal.add_frame(0.30)
    assert len(cal._samples) == 3

    cal.reset()
    assert len(cal._samples) == 0
    assert not cal.is_ready()
