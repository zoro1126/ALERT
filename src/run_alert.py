"""
run_alert.py
============
Entry point for the ALERT driver fatigue detection system.

Usage
-----
  # Run production 9-model heterogeneous ensemble (default):
  python src/run_alert.py

  # Force personal baseline recalibration (15-second setup):
  python src/run_alert.py --recalibrate

  # Run on specific camera index or CPU:
  python src/run_alert.py --camera 0 --device cpu

  # Run a specific single-model checkpoint:
  python src/run_alert.py --model models/ensemble_cnn_gru_fold3.pt

  # Run rule-based demo mode without models:
  python src/run_alert.py --demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from utils.config import Config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ALERT -- Adaptive Landmark Eye-state & Reaction-Time "
                    "Driver Fatigue Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run without a trained model (threshold-based rules). "
             "Good for testing camera / MediaPipe / HUD pipeline.",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Path to a specific .pt model checkpoint. Defaults to None (runs full ensemble).",
    )
    parser.add_argument(
        "--models-dir",
        default=str(Config.MODEL_DIR),
        help="Directory containing trained ensemble checkpoints (ensemble_*.pt).",
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=Config.CAMERA_INDEX,
        help="OpenCV camera index (0 = default webcam).",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Delete saved baseline profile and re-run the 15-second open-eye calibration.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device for neural inference.",
    )
    parser.add_argument(
        "--no-fps",
        action="store_true",
        help="Hide FPS counter in the HUD.",
    )
    args = parser.parse_args()

    # Handle --recalibrate: delete saved calibration profile
    if args.recalibrate:
        cal_path = Path(Config.CALIBRATION_PATH)
        if cal_path.exists():
            cal_path.unlink()
            print(f"[run_alert] Deleted saved calibration profile: {cal_path}")
        else:
            print("[run_alert] No saved calibration profile found; proceeding to calibrate.")

    # --- Demo mode (no model needed) ---
    if args.demo:
        from ui.dashboard import run_demo
        run_demo(
            camera_idx=args.camera,
            show_fps=not args.no_fps,
            force_recalibrate=args.recalibrate,
        )
        return

    # Determine mode: Single Checkpoint vs Heterogeneous Ensemble
    model_path = Path(args.model) if args.model else None
    models_dir = Path(args.models_dir)

    if model_path is not None:
        if not model_path.exists():
            print(f"[run_alert] ERROR: Specified model not found at {model_path}", file=sys.stderr)
            sys.exit(1)
        mode_str = f"Single Checkpoint ({model_path.name})"
    else:
        ensemble_ckpts = list(models_dir.glob("ensemble_*.pt"))
        if not ensemble_ckpts:
            # Fall back to default MODEL_PATH if no ensemble ckpts found
            if Config.MODEL_PATH.exists():
                model_path = Config.MODEL_PATH
                mode_str = f"Single Checkpoint ({model_path.name})"
            else:
                print(
                    f"[run_alert] ERROR: No ensemble checkpoints found in '{models_dir}'.\n"
                    "  Train the ensemble suite first:\n"
                    "    python scripts/train_ensemble.py\n",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            mode_str = f"Heterogeneous Ensemble ({len(ensemble_ckpts)} models in '{models_dir}')"

    print("=" * 65)
    print("  ALERT — Extreme Production Driver Fatigue Detection System")
    print(f"  Mode   : {mode_str}")
    print(f"  Device : {args.device.upper()}")
    print(f"  Camera : index {args.camera}")
    print("=" * 65)

    from ui.dashboard import run
    run(
        model_path=model_path,
        models_dir=models_dir,
        camera_idx=args.camera,
        show_fps=not args.no_fps,
        force_recalibrate=args.recalibrate,
        device=args.device,
    )


if __name__ == "__main__":
    main()
