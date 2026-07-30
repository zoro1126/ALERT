import os
import cv2
import numpy as np
import urllib.request
import mediapipe as mp
from typing import Optional, Dict, Tuple, List

# Check MediaPipe API availability (Tasks API vs Solutions API)
USE_TASKS_API = False
try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    USE_TASKS_API = True
except (ImportError, AttributeError):
    try:
        import mediapipe.python.solutions.face_mesh as mp_face_mesh
    except ImportError:
        from mediapipe.solutions import face_mesh as mp_face_mesh

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_FILENAME = "face_landmarker.task"


class FacialFeatureExtractor:
    """
    Facial Feature Extraction Pipeline for Driver Fatigue Detection.
    Extracts facial landmarks using MediaPipe (Tasks API or Solutions API) and computes:
      - Eye Aspect Ratio (EAR) with head pose yaw correction
      - Mouth Aspect Ratio (MAR) & Mouth-Over-Eye Ratio (MOE)
      - 3D Head Pose Euler Angles (Pitch, Yaw, Roll) via OpenCV solvePnP
    """

    # MediaPipe 468/478 landmark indices for Left & Right Eyes
    # Order: [outer_corner, top_left, top_right, inner_corner, bottom_right, bottom_left]
    LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

    # MediaPipe landmark indices for Inner Lips (MAR computation)
    # Order: p1 (left corner), p2, p3, p4 (top lip), p5 (right corner), p6, p7, p8 (bottom lip)
    MOUTH_LANDMARKS = [78, 81, 13, 311, 308, 402, 14, 82]

    # Selected 3D Model reference points for Head Pose (solvePnP)
    # 6 Key landmarks: Nose tip (1), Chin (152), Left Eye Corner (33), Right Eye Corner (263), Left Mouth (61), Right Mouth (291)
    POSE_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

    # 3D Reference Facial Model points (in mm from face center)
    MODEL_POINTS_3D = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye corner
        (225.0, 170.0, -135.0),      # Right eye corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float64)

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: str = MODEL_FILENAME
    ):
        """
        Initializes the landmark extractor pipeline.

        Args:
            max_num_faces: Maximum number of faces to track (default: 1).
            min_detection_confidence: Minimum detection confidence threshold.
            min_tracking_confidence: Minimum tracking confidence threshold.
            model_path: Path to face_landmarker.task model file (used for Tasks API).
        """
        self.use_tasks_api = USE_TASKS_API

        if self.use_tasks_api:
            # Ensure model file exists or download it automatically
            if not os.path.exists(model_path):
                print(f"[INFO] Downloading MediaPipe FaceLandmarker model to '{model_path}'...")
                try:
                    urllib.request.urlretrieve(MODEL_URL, model_path)
                    print("[INFO] Model download complete.")
                except Exception as e:
                    raise RuntimeError(f"Failed to download MediaPipe face landmarker model: {e}")

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=max_num_faces,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_tracking_confidence
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
        else:
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=max_num_faces,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )

    @staticmethod
    def _euclidean_distance(pt1: np.ndarray, pt2: np.ndarray) -> float:
        """Calculates Euclidean distance between two 2D points."""
        return float(np.linalg.norm(pt1 - pt2))

    def compute_ear(self, landmarks_2d: np.ndarray, eye_indices: List[int]) -> float:
        """
        Computes the Eye Aspect Ratio (EAR) for a single eye.
        Formula: EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        """
        p1 = landmarks_2d[eye_indices[0]]
        p2 = landmarks_2d[eye_indices[1]]
        p3 = landmarks_2d[eye_indices[2]]
        p4 = landmarks_2d[eye_indices[3]]
        p5 = landmarks_2d[eye_indices[4]]
        p6 = landmarks_2d[eye_indices[5]]

        vert1 = self._euclidean_distance(p2, p6)
        vert2 = self._euclidean_distance(p3, p5)
        horiz = self._euclidean_distance(p1, p4)

        if horiz == 0:
            return 0.0

        return (vert1 + vert2) / (2.0 * horiz)

    def compute_mar(self, landmarks_2d: np.ndarray) -> float:
        """
        Computes Mouth Aspect Ratio (MAR) for yawn detection.
        Formula: MAR = (||p2 - p8|| + ||p3 - p7|| + ||p4 - p6||) / (3 * ||p1 - p5||)
        """
        pts = [landmarks_2d[idx] for idx in self.MOUTH_LANDMARKS]
        p1, p2, p3, p4, p5, p6, p7, p8 = pts

        v1 = self._euclidean_distance(p2, p8)
        v2 = self._euclidean_distance(p3, p7)
        v3 = self._euclidean_distance(p4, p6)
        horiz = self._euclidean_distance(p1, p5)

        if horiz == 0:
            return 0.0

        return (v1 + v2 + v3) / (3.0 * horiz)

    def estimate_head_pose(
        self,
        landmarks_2d: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> Tuple[float, float, float]:
        """
        Estimates 3D Head Pose Euler angles (Pitch, Yaw, Roll) in degrees via OpenCV solvePnP.
        """
        h, w = image_shape

        image_points = np.array([
            landmarks_2d[idx] for idx in self.POSE_LANDMARK_INDICES
        ], dtype=np.float64)

        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch = float(euler_angles[0, 0])
        yaw = float(euler_angles[1, 0])
        roll = float(euler_angles[2, 0])

        return pitch, yaw, roll

    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Processes a BGR image frame and extracts driver fatigue metrics.

        Args:
            frame: Input BGR image matrix.

        Returns:
            Dictionary with extracted features if face detected, else None.
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        landmarks_raw = None

        if self.use_tasks_api:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.detector.detect(mp_image)
            if detection_result.face_landmarks:
                landmarks_raw = detection_result.face_landmarks[0]
        else:
            results = self.face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                landmarks_raw = results.multi_face_landmarks[0].landmark

        if landmarks_raw is None:
            return None

        # Convert normalized coordinates [0.0, 1.0] to 2D pixel coordinates (N, 2)
        landmarks_2d = np.array([
            (int(lm.x * w), int(lm.y * h)) for lm in landmarks_raw
        ], dtype=np.int32)

        # Compute Eye Aspect Ratios
        left_ear = self.compute_ear(landmarks_2d, self.LEFT_EYE_LANDMARKS)
        right_ear = self.compute_ear(landmarks_2d, self.RIGHT_EYE_LANDMARKS)
        avg_ear = (left_ear + right_ear) / 2.0

        # Compute Mouth Aspect Ratio & Composite MOE
        mar = self.compute_mar(landmarks_2d)
        moe = mar / avg_ear if avg_ear > 0 else 0.0

        # Estimate Head Pose
        pitch, yaw, roll = self.estimate_head_pose(landmarks_2d, (h, w))

        # Head Pose EAR Correction: EAR_corrected = EAR / cos(|yaw|)
        yaw_rad = np.radians(abs(yaw))
        ear_corrected = avg_ear / np.cos(yaw_rad) if np.cos(yaw_rad) > 0.1 else avg_ear

        return {
            "landmarks_2d": landmarks_2d,
            "left_ear": left_ear,
            "right_ear": right_ear,
            "avg_ear": avg_ear,
            "ear_corrected": ear_corrected,
            "mar": mar,
            "moe": moe,
            "head_pose": {
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll
            }
        }

    def close(self):
        """Releases underlying model resources."""
        if self.use_tasks_api:
            if hasattr(self, 'detector'):
                self.detector.close()
        else:
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()


def open_webcam(indices: List[int] = [1, 0, 2, 3, -1]) -> Tuple[Optional[cv2.VideoCapture], int]:
    """
    Scans available camera indices to locate and open the active camera stream.
    Prioritizes index 1 for Linux laptops where /dev/video0 is an IR node and /dev/video1 is the RGB sensor.
    Configures 30 FPS MJPG stream properties for high-performance real-time video processing.

    Returns:
        Tuple of (cv2.VideoCapture object, opened_index). Returns (None, -1) if no camera found.
    """
    for idx in indices:
        for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                # Set 30 FPS MJPG mode
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)

                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"[INFO] Successfully connected to webcam at index {idx} ({frame.shape[1]}x{frame.shape[0]} @ 30 FPS).")
                    return cap, idx
                cap.release()
    return None, -1


if __name__ == "__main__":
    print("[INFO] Initializing FacialFeatureExtractor Pipeline...")
    extractor = FacialFeatureExtractor()
    print("[INFO] Pipeline initialized successfully.")

    print("[INFO] Auto-detecting and opening webcam stream...")
    cap, cam_idx = open_webcam()

    if cap is None or not cap.isOpened():
        print("[WARNING] No active webcam stream found on indices [0, 1, 2, 3].")
        print("[INFO] Testing single dummy frame through pipeline...")
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = extractor.process_frame(test_frame)
        print(f"[INFO] Dummy frame result: {result} (expected None for blank image)")
        extractor.close()
        print("[INFO] Test complete.")
    else:
        try:
            print(f"[INFO] Camera stream active (Device index {cam_idx}). Press 'q' on video window to exit.")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("[WARNING] Unable to read frame from camera.")
                    break

                metrics = extractor.process_frame(frame)

                if metrics:
                    ear_text = f"EAR: {metrics['avg_ear']:.2f} (Corr: {metrics['ear_corrected']:.2f})"
                    mar_text = f"MAR: {metrics['mar']:.2f} | MOE: {metrics['moe']:.2f}"
                    pose = metrics["head_pose"]
                    pose_text = f"Pitch: {pose['pitch']:.1f}° | Yaw: {pose['yaw']:.1f}° | Roll: {pose['roll']:.1f}°"

                    cv2.putText(frame, ear_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, mar_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(frame, pose_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                    for idx in FacialFeatureExtractor.LEFT_EYE_LANDMARKS + FacialFeatureExtractor.RIGHT_EYE_LANDMARKS:
                        cv2.circle(frame, tuple(metrics["landmarks_2d"][idx]), 2, (0, 255, 0), -1)

                    for idx in FacialFeatureExtractor.MOUTH_LANDMARKS:
                        cv2.circle(frame, tuple(metrics["landmarks_2d"][idx]), 2, (0, 0, 255), -1)

                cv2.imshow("Driver Fatigue Feature Extractor", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            extractor.close()

