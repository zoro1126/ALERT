import os
import joblib
import numpy as np
from typing import Dict, Any, Optional

from train_model import train_and_save_classifier

DEFAULT_MODEL_PATH = "fatigue_classifier.joblib"


class FatigueClassifier:
    """
    Classical Machine Learning Classifier for Driver Fatigue Level Grading (L0 - L3).
    Loads a trained Random Forest / XGBoost model artifact and predicts multiclass probabilities.
    Includes micro-sleep and severe head-droop safety heuristics (§9.5 of Research Doc).
    """

    CLASS_MAP = {
        0: "L0 Alert",
        1: "L1 Mild Fatigue",
        2: "L2 Moderate Fatigue",
        3: "L3 Severe Fatigue"
    }

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.scaler = None

        self.load_or_train_model()

    def load_or_train_model(self):
        """Loads the trained model artifact from disk or trains a new one if missing."""
        if not os.path.exists(self.model_path):
            print(f"[INFO] Model artifact '{self.model_path}' not found. Training model now...")
            train_and_save_classifier(self.model_path)

        try:
            artifact = joblib.load(self.model_path)
            self.model = artifact["model"]
            self.scaler = artifact["scaler"]
            print(f"[INFO] Loaded trained classifier successfully from '{self.model_path}'.")
        except Exception as e:
            print(f"[ERROR] Failed to load model artifact: {e}. Retraining...")
            train_and_save_classifier(self.model_path)
            artifact = joblib.load(self.model_path)
            self.model = artifact["model"]
            self.scaler = artifact["scaler"]

    def predict(self, feature_vector: np.ndarray, raw_features: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Predicts fatigue severity level (L0 - L3) from a 9-element feature vector.

        Args:
            feature_vector: 1D numpy array of 9 features [F1-F9].
            raw_features: Optional dict containing unscaled features for rule overrides.

        Returns:
            Dictionary with predicted_class_id, predicted_label, confidence, probabilities, and override status.
        """
        if feature_vector is None or len(feature_vector) == 0:
            return {
                "class_id": 0,
                "label": "L0 Alert",
                "confidence": 1.0,
                "probabilities": [1.0, 0.0, 0.0, 0.0],
                "is_override": False
            }

        # Reshape for single sample prediction
        X_in = feature_vector.reshape(1, -1)
        X_scaled = self.scaler.transform(X_in)

        # Predict class probabilities using trained RF/XGB model
        probs = self.model.predict_proba(X_scaled)[0]
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])

        is_override = False

        # --- SAFETY HEURISTIC OVERRIDES (§9.5 L3 Heuristic Override) ---
        # If head pitch < -25° OR PERCLOS > 50%, force L3 Severe Fatigue regardless of classifier
        if raw_features:
            pitch = raw_features.get("F8_Head_Pitch", 0.0)
            perclos = raw_features.get("F3_PERCLOS", 0.0)

            if pitch < -25.0 or perclos > 50.0:
                class_id = 3
                confidence = 1.0
                probs = [0.0, 0.0, 0.0, 1.0]
                is_override = True

        label = self.CLASS_MAP.get(class_id, "L0 Alert")

        return {
            "class_id": class_id,
            "label": label,
            "confidence": confidence,
            "probabilities": [float(p) for p in probs],
            "is_override": is_override
        }


if __name__ == "__main__":
    classifier = FatigueClassifier()

    # Test sample: Alert Driver (L0)
    sample_l0 = np.array([0.31, 0.015, 4.0, 18.0, 200.0, 0.21, 0, 0.0, 0.67], dtype=np.float32)
    res_l0 = classifier.predict(sample_l0)
    print("\n[TEST L0 ALERT]:", res_l0)

    # Test sample: Severe Fatigue Driver (L3)
    sample_l3 = np.array([0.12, 0.060, 60.0, 4.0, 700.0, 0.58, 4, -28.0, 4.8], dtype=np.float32)
    res_l3 = classifier.predict(sample_l3)
    print("[TEST L3 SEVERE]:", res_l3)
