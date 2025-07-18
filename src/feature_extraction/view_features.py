import polars as pl
from typing import List
from .base import FeatureExtractor


class ViewSpecificFeatureExtractor(FeatureExtractor):
    """Extract features specific to different camera views."""

    def __init__(self):
        super().__init__("view_specific_features")

    def extract(self, landmarks_df: pl.DataFrame) -> pl.DataFrame:
        """Extract view-specific features."""
        features = []

        for row in landmarks_df.iter_rows(named=True):
            view_type = row.get('view_type', 'front')
            row_features = {}

            if view_type == 'front':
                row_features.update(self._extract_front_view_features(row))
            elif view_type == 'left':
                row_features.update(self._extract_side_view_features(row, 'left'))
            elif view_type == 'right':
                row_features.update(self._extract_side_view_features(row, 'right'))

            features.append(row_features)

        return pl.DataFrame(features)

    def get_feature_names(self) -> List[str]:
        return [
            'front_shoulder_alignment', 'front_hip_alignment', 'front_head_tilt',
            'side_forward_lean', 'side_spine_curve', 'side_head_position'
        ]

    def _extract_front_view_features(self, row) -> dict:
        """Extract features specific to front view."""
        features = {}

        # Shoulder alignment in front view
        left_shoulder_y = row.get('LEFT_SHOULDER_y', 0)
        right_shoulder_y = row.get('RIGHT_SHOULDER_y', 0)
        features['front_shoulder_alignment'] = abs(left_shoulder_y - right_shoulder_y) if (left_shoulder_y and right_shoulder_y) else 0.0

        # Hip alignment in front view
        left_hip_y = row.get('LEFT_HIP_y', 0)
        right_hip_y = row.get('RIGHT_HIP_y', 0)
        features['front_hip_alignment'] = abs(left_hip_y - right_hip_y) if (left_hip_y and right_hip_y) else 0.0

        # Head tilt in front view
        nose_x = row.get('NOSE_x', 0)
        shoulder_center_x = (row.get('LEFT_SHOULDER_x', 0) + row.get('RIGHT_SHOULDER_x', 0)) / 2
        features['front_head_tilt'] = abs(nose_x - shoulder_center_x) if (nose_x and shoulder_center_x) else 0.0

        # Fill side view features with zeros for front view
        features['side_forward_lean'] = 0.0
        features['side_spine_curve'] = 0.0
        features['side_head_position'] = 0.0

        return features

    def _extract_side_view_features(self, row, side) -> dict:
        """Extract features specific to side views."""
        features = {}

        # Forward lean in side view
        nose_x = row.get('NOSE_x', 0)
        hip_center_x = (row.get('LEFT_HIP_x', 0) + row.get('RIGHT_HIP_x', 0)) / 2
        features['side_forward_lean'] = abs(nose_x - hip_center_x) if (nose_x and hip_center_x) else 0.0

        # Spine curvature in side view
        nose_y = row.get('NOSE_y', 0)
        shoulder_y = (row.get('LEFT_SHOULDER_y', 0) + row.get('RIGHT_SHOULDER_y', 0)) / 2
        hip_y = (row.get('LEFT_HIP_y', 0) + row.get('RIGHT_HIP_y', 0)) / 2

        if all([nose_y, shoulder_y, hip_y]):
            # Simple spine curvature measure
            features['side_spine_curve'] = abs((nose_y + hip_y) / 2 - shoulder_y)
        else:
            features['side_spine_curve'] = 0.0

        # Head position relative to shoulders
        features['side_head_position'] = abs(nose_x - (row.get('LEFT_SHOULDER_x', 0) + row.get('RIGHT_SHOULDER_x', 0)) / 2) if nose_x else 0.0

        # Fill front view features with zeros for side view
        features['front_shoulder_alignment'] = 0.0
        features['front_hip_alignment'] = 0.0
        features['front_head_tilt'] = 0.0

        return features
