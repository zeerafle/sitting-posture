import pandas as pd
from typing import List, Dict, Any
from .base import FeatureExtractor


class ViewSpecificFeatureExtractor(FeatureExtractor):
    """Extract features specific to different camera views using pandas."""

    def __init__(self):
        super().__init__("view_specific_features")

    @staticmethod
    def _v(row, key, default: float = 0.0) -> float:
        val = row.get(key, default)
        try:
            if pd.isna(val):
                return float(default)
        except Exception:
            # If val does not support isna checks, fall back to conversion
            pass
        try:
            return float(val)
        except Exception:
            return float(default)

    def extract(self, landmarks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract view-specific features from a pandas DataFrame.

        Expected columns (subset may be used depending on view):
        - 'view_type'
        - 'LEFT_SHOULDER_x', 'LEFT_SHOULDER_y'
        - 'RIGHT_SHOULDER_x', 'RIGHT_SHOULDER_y'
        - 'LEFT_HIP_x', 'LEFT_HIP_y'
        - 'RIGHT_HIP_x', 'RIGHT_HIP_y'
        - 'NOSE_x', 'NOSE_y'
        """
        features: List[Dict[str, Any]] = []

        for _, row in landmarks_df.iterrows():
            view_type = row.get('view_type', 'front')
            if pd.isna(view_type):
                view_type = 'front'
            row_features: Dict[str, float] = {}

            if view_type == 'front':
                row_features.update(self._extract_front_view_features(row))
            elif view_type == 'left':
                row_features.update(self._extract_side_view_features(row, 'left'))
            elif view_type == 'right':
                row_features.update(self._extract_side_view_features(row, 'right'))
            else:
                # Unknown view; default zeros
                row_features = {
                    'front_shoulder_alignment': 0.0,
                    'front_hip_alignment': 0.0,
                    'front_head_tilt': 0.0,
                    'side_forward_lean': 0.0,
                    'side_spine_curve': 0.0,
                    'side_head_position': 0.0,
                }

            features.append(row_features)

        df = pd.DataFrame(features)
        return df.reindex(columns=self.get_feature_names(), fill_value=0.0)

    def get_feature_names(self) -> List[str]:
        return [
            'front_shoulder_alignment', 'front_hip_alignment', 'front_head_tilt',
            'side_forward_lean', 'side_spine_curve', 'side_head_position'
        ]

    def _extract_front_view_features(self, row: pd.Series) -> Dict[str, float]:
        """Extract features specific to front view."""
        features: Dict[str, float] = {}

        # Shoulder alignment in front view
        left_shoulder_y = self._v(row, 'LEFT_SHOULDER_y', 0.0)
        right_shoulder_y = self._v(row, 'RIGHT_SHOULDER_y', 0.0)
        features['front_shoulder_alignment'] = abs(left_shoulder_y - right_shoulder_y)

        # Hip alignment in front view
        left_hip_y = self._v(row, 'LEFT_HIP_y', 0.0)
        right_hip_y = self._v(row, 'RIGHT_HIP_y', 0.0)
        features['front_hip_alignment'] = abs(left_hip_y - right_hip_y)

        # Head tilt in front view
        nose_x = self._v(row, 'NOSE_x', 0.0)
        left_shoulder_x = self._v(row, 'LEFT_SHOULDER_x', 0.0)
        right_shoulder_x = self._v(row, 'RIGHT_SHOULDER_x', 0.0)
        shoulder_center_x = (left_shoulder_x + right_shoulder_x) / 2.0
        features['front_head_tilt'] = abs(nose_x - shoulder_center_x)

        # Fill side view features with zeros for front view
        features['side_forward_lean'] = 0.0
        features['side_spine_curve'] = 0.0
        features['side_head_position'] = 0.0

        return features

    def _extract_side_view_features(self, row: pd.Series, side: str) -> Dict[str, float]:
        """Extract features specific to side views."""
        features: Dict[str, float] = {}

        # Forward lean in side view
        nose_x = self._v(row, 'NOSE_x', 0.0)
        left_hip_x = self._v(row, 'LEFT_HIP_x', 0.0)
        right_hip_x = self._v(row, 'RIGHT_HIP_x', 0.0)
        hip_center_x = (left_hip_x + right_hip_x) / 2.0
        features['side_forward_lean'] = abs(nose_x - hip_center_x)

        # Spine curvature in side view
        nose_y = self._v(row, 'NOSE_y', 0.0)
        left_shoulder_y = self._v(row, 'LEFT_SHOULDER_y', 0.0)
        right_shoulder_y = self._v(row, 'RIGHT_SHOULDER_y', 0.0)
        shoulder_y = (left_shoulder_y + right_shoulder_y) / 2.0
        left_hip_y = self._v(row, 'LEFT_HIP_y', 0.0)
        right_hip_y = self._v(row, 'RIGHT_HIP_y', 0.0)
        hip_y = (left_hip_y + right_hip_y) / 2.0
        # Simple spine curvature measure
        features['side_spine_curve'] = abs((nose_y + hip_y) / 2.0 - shoulder_y)

        # Head position relative to shoulders
        left_shoulder_x2 = self._v(row, 'LEFT_SHOULDER_x', 0.0)
        right_shoulder_x2 = self._v(row, 'RIGHT_SHOULDER_x', 0.0)
        shoulder_center_x2 = (left_shoulder_x2 + right_shoulder_x2) / 2.0
        features['side_head_position'] = abs(nose_x - shoulder_center_x2)

        # Fill front view features with zeros for side view
        features['front_shoulder_alignment'] = 0.0
        features['front_hip_alignment'] = 0.0
        features['front_head_tilt'] = 0.0

        return features
