import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

MODEL_FILENAME = "fatigue_classifier.joblib"


def generate_synthetic_fatigue_dataset(num_samples_per_class: int = 2500, random_seed: int = 42):
    """
    Generates a realistic synthetic physiological dataset based on Wierwille-Ellsworth drowsiness scale
    and PERCLOS-validated thresholds (§9.1 - §9.2 of Research Document).

    Features (F1 - F9):
      F1: EAR (Eye Aspect Ratio)
      F2: EAR_std (5s rolling std)
      F3: PERCLOS (% closed in 60s)
      F4: Blink Rate (blinks/min)
      F5: Blink Duration (ms)
      F6: MAR (Mouth Aspect Ratio)
      F7: Yawn Count (5 min rolling)
      F8: Head Pitch (deg)
      F9: MOE (Mouth-Over-Eye Ratio)

    Classes:
      0: L0 Alert
      1: L1 Mild Fatigue
      2: L2 Moderate Fatigue
      3: L3 Severe Fatigue
    """
    np.random.seed(random_seed)
    X_list = []
    y_list = []

    classes = [0, 1, 2, 3]

    for label in classes:
        n = num_samples_per_class

        if label == 0:  # L0 Alert
            ear = np.random.normal(loc=0.30, scale=0.025, size=n)
            ear = np.clip(ear, 0.26, 0.40)

            ear_std = np.random.normal(loc=0.015, scale=0.005, size=n)
            ear_std = np.clip(ear_std, 0.005, 0.03)

            perclos = np.random.normal(loc=5.0, scale=3.0, size=n)
            perclos = np.clip(perclos, 0.0, 14.5)

            blink_rate = np.random.normal(loc=17.0, scale=2.5, size=n)
            blink_rate = np.clip(blink_rate, 12.0, 22.0)

            blink_duration = np.random.normal(loc=200.0, scale=30.0, size=n)
            blink_duration = np.clip(blink_duration, 120.0, 280.0)

            mar = np.random.normal(loc=0.20, scale=0.03, size=n)
            mar = np.clip(mar, 0.12, 0.28)

            yawn_count = np.zeros(n, dtype=int)

            head_pitch = np.random.normal(loc=0.0, scale=3.0, size=n)
            head_pitch = np.clip(head_pitch, -7.0, 7.0)

        elif label == 1:  # L1 Mild Fatigue
            ear = np.random.normal(loc=0.25, scale=0.02, size=n)
            ear = np.clip(ear, 0.21, 0.28)

            ear_std = np.random.normal(loc=0.028, scale=0.008, size=n)
            ear_std = np.clip(ear_std, 0.015, 0.045)

            perclos = np.random.normal(loc=20.0, scale=2.5, size=n)
            perclos = np.clip(perclos, 15.0, 24.9)

            b_type = np.random.binomial(1, 0.5, size=n)
            blink_rate = np.where(b_type == 1, np.random.normal(11.0, 1.5, size=n), np.random.normal(24.0, 2.0, size=n))
            blink_rate = np.clip(blink_rate, 8.0, 28.0)

            blink_duration = np.random.normal(loc=300.0, scale=40.0, size=n)
            blink_duration = np.clip(blink_duration, 240.0, 380.0)

            mar = np.random.normal(loc=0.28, scale=0.05, size=n)
            mar = np.clip(mar, 0.18, 0.42)

            yawn_count = np.random.choice([0, 1], size=n, p=[0.7, 0.3])

            head_pitch = np.random.normal(loc=-4.0, scale=4.0, size=n)
            head_pitch = np.clip(head_pitch, -12.0, 5.0)

        elif label == 2:  # L2 Moderate Fatigue
            ear = np.random.normal(loc=0.20, scale=0.018, size=n)
            ear = np.clip(ear, 0.17, 0.23)

            ear_std = np.random.normal(loc=0.042, scale=0.01, size=n)
            ear_std = np.clip(ear_std, 0.025, 0.065)

            perclos = np.random.normal(loc=32.0, scale=3.5, size=n)
            perclos = np.clip(perclos, 25.0, 39.9)

            blink_rate = np.random.normal(loc=9.0, scale=2.0, size=n)
            blink_rate = np.clip(blink_rate, 5.0, 14.0)

            blink_duration = np.random.normal(loc=420.0, scale=50.0, size=n)
            blink_duration = np.clip(blink_duration, 340.0, 550.0)

            mar = np.random.normal(loc=0.42, scale=0.08, size=n)
            mar = np.clip(mar, 0.28, 0.60)

            yawn_count = np.random.choice([1, 2, 3], size=n, p=[0.5, 0.35, 0.15])

            head_pitch = np.random.normal(loc=-11.0, scale=4.5, size=n)
            head_pitch = np.clip(head_pitch, -20.0, -2.0)

        else:  # label == 3: L3 Severe Fatigue
            ear = np.random.normal(loc=0.14, scale=0.02, size=n)
            ear = np.clip(ear, 0.05, 0.18)

            ear_std = np.random.normal(loc=0.055, scale=0.015, size=n)
            ear_std = np.clip(ear_std, 0.03, 0.09)

            perclos = np.random.normal(loc=55.0, scale=8.0, size=n)
            perclos = np.clip(perclos, 40.0, 85.0)

            blink_rate = np.random.normal(loc=5.0, scale=2.0, size=n)
            blink_rate = np.clip(blink_rate, 1.0, 9.0)

            blink_duration = np.random.normal(loc=650.0, scale=120.0, size=n)
            blink_duration = np.clip(blink_duration, 480.0, 1200.0)

            mar = np.random.normal(loc=0.52, scale=0.10, size=n)
            mar = np.clip(mar, 0.35, 0.75)

            yawn_count = np.random.choice([2, 3, 4, 5, 6], size=n, p=[0.2, 0.3, 0.25, 0.15, 0.1])

            head_pitch = np.random.normal(loc=-22.0, scale=6.0, size=n)
            head_pitch = np.clip(head_pitch, -40.0, -12.0)

        # Derived F9: MOE (Mouth-Over-Eye Ratio = MAR / EAR)
        moe = mar / np.maximum(ear, 0.05)

        X_sub = np.column_stack([
            ear, ear_std, perclos, blink_rate,
            blink_duration, mar, yawn_count,
            head_pitch, moe
        ])

        y_sub = np.full(n, label, dtype=int)

        X_list.append(X_sub)
        y_list.append(y_sub)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    return X, y


def train_and_save_classifier(output_path: str = MODEL_FILENAME):
    print("[INFO] Generating synthetic physiological training dataset (10,000 samples)...")
    X, y = generate_synthetic_fatigue_dataset(num_samples_per_class=2500)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 30 estimators, max_depth=8, n_jobs=1 for ultra-fast <2.5ms real-time inference
    print(f"[INFO] Training Ultra-Fast 30-Tree Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=30,
        max_depth=8,
        random_state=42,
        n_jobs=1
    )
    rf_model.fit(X_train_scaled, y_train)

    y_pred = rf_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[EVALUATION RESULTS] Fast Random Forest Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=["L0 Alert", "L1 Mild", "L2 Moderate", "L3 Severe"]))

    artifact = {
        "model": rf_model,
        "scaler": scaler,
        "class_labels": {0: "L0 Alert", 1: "L1 Mild", 2: "L2 Moderate", 3: "L3 Severe"},
        "feature_names": [
            "F1_EAR", "F2_EAR_std", "F3_PERCLOS", "F4_Blink_Rate",
            "F5_Blink_Duration_ms", "F6_MAR", "F7_Yawn_Count",
            "F8_Head_Pitch", "F9_MOE"
        ]
    }

    joblib.dump(artifact, output_path)
    print(f"[INFO] Fast Model artifact successfully saved to '{output_path}'.")


if __name__ == "__main__":
    train_and_save_classifier()
