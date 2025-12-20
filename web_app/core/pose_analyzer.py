import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import polars as pl
import joblib
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from collections import deque
from datetime import datetime
import os
import sys

# Add src directory to path to import project modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from data import BodyPart
from feature_extraction.pose_features import (
    AngleFeatureExtractor,
    DistanceFeatureExtractor,
    SpineAlignmentFeatureExtractor
)
from feature_extraction.base import FeatureExtractionPipeline


class PostureAnalyzer:
    """Core posture analysis class."""

    def __init__(self, model_type="xgb", project_root=None):
        """
        Initialize the posture analyzer.

        Args:
            model_type: Type of model to use ("xgb", "adaboost", "nn")
            project_root: Path to project root directory
        """
        self.model_type = model_type
        self.class_names = {0: "Ergonomic", 1: "Non-Ergonomic"}

        # Set project root
        if project_root is None:
            self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        else:
            self.project_root = project_root

        # Initialize data storage for analytics
        self.prediction_history = deque(maxlen=1000)
        self.timestamps = deque(maxlen=1000)

        # Load MoveNet model
        print("Loading MoveNet model...")
        try:
            model = hub.load(
                "https://www.kaggle.com/models/google/movenet/TensorFlow2/singlepose-thunder/4"
            )
            self.movenet = model.signatures["serving_default"]
            print("MoveNet model loaded successfully!")
        except Exception as e:
            print(f"Error loading MoveNet model: {e}")
            self.movenet = None

        # Load model and preprocessors
        self._load_model()
        self._load_preprocessors()

        # Initialize feature extractors
        self.feature_extractors = [
            AngleFeatureExtractor(),
            DistanceFeatureExtractor(),
            SpineAlignmentFeatureExtractor()
        ]
        self.feature_pipeline = FeatureExtractionPipeline(self.feature_extractors)

        print("Posture Analyzer initialized successfully!")

    def _load_model(self):
        """Load the trained model."""
        model_path = os.path.join(self.project_root, "models", self.model_type, f"{self.model_type}_combined")

        try:
            if self.model_type == "xgb":
                # XGBoost models are saved as .json files
                json_path = f"{model_path}.json"
                if os.path.exists(json_path):
                    self.model = xgb.Booster()
                    self.model.load_model(json_path)
                else:
                    # Fallback to .joblib
                    joblib_path = f"{model_path}.joblib"
                    self.model = joblib.load(joblib_path)
            elif self.model_type in ["adaboost", "nn"]:
                # Scikit-learn models are saved as .joblib files
                joblib_path = f"{model_path}.joblib"
                self.model = joblib.load(joblib_path)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            print(f"Loaded {self.model_type} combined model")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def _load_preprocessors(self):
        """Load preprocessors from the combined directory."""
        preprocessor_dir = os.path.join(
            self.project_root, "data", "processed", "combined", "preprocessors"
        )

        try:
            # Load feature scaler
            scaler_path = os.path.join(preprocessor_dir, "feature_scaler.joblib")
            if os.path.exists(scaler_path):
                self.feature_scaler = joblib.load(scaler_path)
                print(f"Loaded feature scaler - expects {self.feature_scaler.n_features_in_} features")
            else:
                print("Warning: Feature scaler not found")
                self.feature_scaler = None

            # Load feature info
            feature_info_path = os.path.join(preprocessor_dir, "feature_info.joblib")
            if os.path.exists(feature_info_path):
                self.feature_info = joblib.load(feature_info_path)
                print("Loaded feature info")
                print(f"Feature info keys: {list(self.feature_info.keys()) if isinstance(self.feature_info, dict) else 'Not a dict'}")
            else:
                print("Warning: Feature info not found")
                self.feature_info = None

            # Load group imputers
            group_imputers_path = os.path.join(preprocessor_dir, "group_imputers.joblib")
            if os.path.exists(group_imputers_path):
                self.group_imputers = joblib.load(group_imputers_path)
                print("Loaded group imputers")
            else:
                print("Warning: Group imputers not found")
                self.group_imputers = None

        except Exception as e:
            print(f"Error loading preprocessors: {e}")
            self.feature_scaler = None
            self.feature_info = None
            self.group_imputers = None

    def update_model_type(self, model_type):
        """Update the model type and reload."""
        self.model_type = model_type
        self._load_model()
        return f"Updated to {model_type} model"

    def preprocess_frame(self, frame):
        """Preprocess frame for MoveNet input."""
        if frame is None:
            return None

        # Convert to RGB if needed
        if len(frame.shape) == 3:
            if frame.shape[2] == 4:  # RGBA
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            elif frame.shape[2] == 3:  # Assume BGR
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
        else:
            frame_rgb = frame

        # Use TensorFlow's resize_with_pad to maintain aspect ratio
        frame_tensor = tf.cast(
            tf.image.resize_with_pad(frame_rgb, 256, 256), dtype=tf.int32
        )

        # Expand dimensions
        frame_tensor = tf.expand_dims(frame_tensor, axis=0)

        return frame_tensor

    def extract_keypoints(self, frame_tensor):
        """Extract keypoints using MoveNet."""
        if self.movenet is None:
            return None

        outputs = self.movenet(frame_tensor)
        keypoints = outputs['output_0'].numpy()

        # Reshape to (17, 3) - 17 keypoints with (y, x, confidence)
        keypoints = keypoints.reshape((17, 3))

        # Filter for the 13 keypoints we use (matching the training data)
        # COCO-17 indices: 0=nose, 5=left_shoulder, 6=right_shoulder, etc.
        used_indices = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        keypoints_filtered = keypoints[used_indices]

        return keypoints_filtered

    def get_center_point(self, landmarks, left_bodypart, right_bodypart):
        """Calculate center point between two landmarks."""
        left = landmarks[left_bodypart.value]
        right = landmarks[right_bodypart.value]
        center = (left + right) * 0.5
        return center

    def get_pose_size(self, landmarks, torso_size_multiplier=2.5):
        """Calculate pose size for normalization."""
        try:
            # Hips center
            hips_center = self.get_center_point(
                landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)

            # Shoulders center
            shoulders_center = self.get_center_point(
                landmarks, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER)

            # Torso size
            torso_size = np.linalg.norm(shoulders_center - hips_center)

            # Pose center
            pose_center = hips_center

            # Max distance from center to any landmark
            distances = np.linalg.norm(landmarks[:, :2] - pose_center, axis=1)
            max_dist = np.max(distances)

            # Pose size
            pose_size = max(torso_size * torso_size_multiplier, max_dist)

            return pose_size, pose_center
        except:
            return 1.0, np.array([0.5, 0.5])

    def normalize_landmarks(self, landmarks):
        """Normalize landmarks for pose embedding."""
        pose_size, pose_center = self.get_pose_size(landmarks)

        # Center and scale landmarks
        normalized = landmarks.copy()
        normalized[:, :2] = (landmarks[:, :2] - pose_center) / pose_size

        return normalized

    def landmarks_to_dataframe(self, landmarks):
        """Convert landmarks to DataFrame format expected by feature extractors - EXCLUDING visibility scores."""
        landmark_names = [
            'NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
            'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE',
            'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE'
        ]

        data = {}
        for i, name in enumerate(landmark_names):
            data[f'{name}_x'] = [landmarks[i, 1]]  # x coordinate
            data[f'{name}_y'] = [landmarks[i, 0]]  # y coordinate
            # EXCLUDED: visibility/confidence scores to match training data
            # data[f'{name}_visibility'] = [landmarks[i, 2]]

        return pl.DataFrame(data)

    def extract_features(self, landmarks):
        """Extract both normalized embeddings and engineered features to match training pipeline."""
        try:
            # 1. Create normalized pose embedding (26 features: 13 landmarks * 2 coords only, NO visibility)
            normalized_landmarks = self.normalize_landmarks(landmarks)
            embedding = normalized_landmarks[:, :2].flatten()  # Only x,y coordinates

            # 2. Extract engineered features (also without visibility scores)
            landmarks_df = self.landmarks_to_dataframe(landmarks)
            features_df = self.feature_pipeline.extract_all(landmarks_df)

            # Keep metadata columns separate
            metadata_cols = ["file_name", "class_name", "class_no", "view_type"]

            # Engineered feature columns (all columns except landmarks and metadata)
            feature_cols = [col for col in features_df.columns if col not in landmarks_df.columns + metadata_cols]
            engineered_features = features_df.select(feature_cols).to_numpy().flatten()

            print(f"Debug - Embedding shape (x,y only): {embedding.shape}")
            print(f"Debug - Engineered features shape: {engineered_features.shape}")
            print(f"Debug - Feature columns: {feature_cols}")

            # 3. Handle missing features based on feature_info if available
            if self.feature_info is not None and 'feature_names' in self.feature_info:
                expected_features = self.feature_info['feature_names']
                print(f"Debug - Expected features from training: {len(expected_features)}")

                # Create a mapping of current features to expected features
                current_feature_dict = {col: engineered_features[i] for i, col in enumerate(feature_cols)}

                # Create engineered features array matching training order
                aligned_engineered = []
                for feature_name in expected_features:
                    if feature_name in current_feature_dict:
                        aligned_engineered.append(current_feature_dict[feature_name])
                    else:
                        # Fill missing features with 0 or mean value
                        aligned_engineered.append(0.0)
                        print(f"Debug - Missing feature filled with 0: {feature_name}")

                engineered_features = np.array(aligned_engineered)
                print(f"Debug - Aligned engineered features shape: {engineered_features.shape}")

            # 4. Apply scaling if scaler is available
            if self.feature_scaler is not None:
                expected_scaler_features = self.feature_scaler.n_features_in_
                print(f"Debug - Scaler expects {expected_scaler_features} features, got {len(engineered_features)}")

                # Ensure we have the right number of features for the scaler
                if len(engineered_features) > expected_scaler_features:
                    # Truncate extra features
                    engineered_features = engineered_features[:expected_scaler_features]
                    print(f"Debug - Truncated to {len(engineered_features)} features")
                elif len(engineered_features) < expected_scaler_features:
                    # Pad with zeros
                    padding = np.zeros(expected_scaler_features - len(engineered_features))
                    engineered_features = np.concatenate([engineered_features, padding])
                    print(f"Debug - Padded to {len(engineered_features)} features")

                # Scale the engineered features
                scaled_engineered = self.feature_scaler.transform(engineered_features.reshape(1, -1)).flatten()
                print(f"Debug - Scaled features shape: {scaled_engineered.shape}")

                # Combine embedding (unscaled) with scaled engineered features
                combined_features = np.concatenate([embedding, scaled_engineered])
            else:
                # No scaler available, use raw engineered features
                combined_features = np.concatenate([embedding, engineered_features])

            print(f"Debug - Final combined features shape: {combined_features.shape}")
            return combined_features

        except Exception as e:
            print(f"Error extracting features: {e}")
            import traceback
            traceback.print_exc()

            # Return appropriate number of dummy features based on model expectations
            if self.model is not None:
                if self.model_type == "nn" and hasattr(self.model, 'n_features_in_'):
                    dummy_size = self.model.n_features_in_
                elif self.model_type == "xgb":
                    # XGBoost doesn't have n_features_in_, use common size
                    dummy_size = 57  # Based on the error message
                else:
                    dummy_size = 57  # Default fallback
            else:
                dummy_size = 57

            print(f"Debug - Returning {dummy_size} dummy features")
            return np.zeros(dummy_size)

    def predict_posture(self, features):
        """Make posture prediction using the loaded model."""
        if self.model is None:
            return 0, 0.5

        try:
            features = features.reshape(1, -1)
            print(f"Debug - Prediction input shape: {features.shape}")

            if self.model_type == "xgb":
                dmatrix = xgb.DMatrix(features)
                prediction = self.model.predict(dmatrix)[0]
                # XGBoost returns probability directly
                probability = float(prediction)
                class_prediction = int(probability > 0.5)
            else:
                # For sklearn models (adaboost, nn)
                probabilities = self.model.predict_proba(features)[0]
                class_prediction = int(probabilities[1] > 0.5)  # probability of class 1 (Non-Ergonomic)
                probability = float(probabilities[class_prediction])

            return class_prediction, probability

        except Exception as e:
            print(f"Error in prediction: {e}")
            return 0, 0.5

    def process_frame(self, frame):
        """Process a single frame and return prediction results."""
        if frame is None:
            return None, "No frame", 0.0, "⚠️ No Input", 0.0

        # Start timing
        start_time = datetime.now()

        try:
            # 1. Preprocess frame
            frame_tensor = self.preprocess_frame(frame)
            if frame_tensor is None:
                return frame, "Processing Error", 0.0, "❌ Error", 0.0

            # 2. Extract keypoints
            keypoints = self.extract_keypoints(frame_tensor)
            if keypoints is None:
                return frame, "Keypoint Error", 0.0, "❌ Error", 0.0

            # 3. Check if pose is detected (using confidence scores for detection, but not for features)
            confident_keypoints = keypoints[keypoints[:, 2] > 0.3]

            if len(confident_keypoints) >= 5:  # Need at least 5 confident keypoints
                # 4. Extract features (without visibility scores)
                features = self.extract_features(keypoints)

                # 5. Make prediction
                class_pred, confidence = self.predict_posture(features)
                posture_label = self.class_names[class_pred]

                # Store prediction for analytics
                current_time = datetime.now()
                self.prediction_history.append(class_pred)
                self.timestamps.append(current_time)

                # Create status emoji
                status_emoji = "✅ Good Posture" if class_pred == 0 else "❌ Poor Posture"

                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds() * 1000  # Convert to milliseconds

                return keypoints, posture_label, confidence, status_emoji, processing_time
            else:
                # Calculate processing time even for failure cases
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                return keypoints, "No Pose Detected", 0.0, "⚠️ No Pose", processing_time

        except Exception as e:
            print(f"Error processing frame: {e}")
            import traceback
            traceback.print_exc()

            # Calculate processing time even for errors
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return None, "Processing Error", 0.0, "❌ Error", processing_time

    def clear_history(self):
        """Clear prediction history."""
        self.prediction_history.clear()
        self.timestamps.clear()
