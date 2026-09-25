"""
run_alert.py
============
Entry point for the ALERT driver fatigue detection system.

Usage
-----
  # From project root:
  python src/run_alert.py

  # Custom model or camera:
  python src/run_alert.py --model models/my_model.pt --camera 1

  # Re-run calibration (ignore saved threshold):
  python src/run_alert.py --recalibrate
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
        description="ALERT — Adaptive Landmark Eye-state & Reaction-Time "
                    "Driver Fatigue Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run without a trained model (threshold-based rules). "
             "Good for testing the camera/MediaPipe/HUD pipeline.",
    )
    parser.add_argument(
        "--model", "-m",
        default=str(Config.MODEL_PATH),
        help="Path to trained .pt model checkpoint.",
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
        help="Delete saved calibration and re-run the 10-second open-eye session.",
    )
    parser.add_argument(
        "--no-fps",
        action="store_true",
        help="Hide FPS counter in the HUD.",
    )
    args = parser.parse_args()

    # Handle --recalibrate: delete saved threshold file
    if args.recalibrate:
        cal_path = Path(Config.CALIBRATION_PATH)
        if cal_path.exists():
            cal_path.unlink()
            print(f"[run_alert] Deleted saved calibration: {cal_path}")
        else:
            print("[run_alert] No saved calibration found; proceeding to calibrate.")

    # --- Demo mode (no model needed) ---
    if args.demo:
        from ui.dashboard import run_demo
        run_demo(
            camera_idx=args.camera,
            show_fps=not args.no_fps,
        )
        return

    model_path = Path(args.model)
    if not model_path.exists():
        print(
            f"[run_alert] ERROR: Model not found at {model_path}\n"
            "  Train the model first:\n"
            "    python scripts/extract_cache.py\n"
            "    python -m training.trainer\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 55)
    print("  ALERT — Driver Fatigue Detection System")
    print(f"  Model  : {model_path}")
    print(f"  Camera : index {args.camera}")
    print("=" * 55)

    from ui.dashboard import run
    run(
        model_path=model_path,
        camera_idx=args.camera,
        show_fps=not args.no_fps,
    )


if __name__ == "__main__":
    main()
