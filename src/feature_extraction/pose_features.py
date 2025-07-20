import polars as pl
import numpy as np
from typing import List
from .base import FeatureExtractor


class AngleFeatureExtractor(FeatureExtractor):
    """Extract angle features between body parts."""

    def __init__(self):
        super().__init__("angle_features")

    def extract(self, landmarks_df: pl.DataFrame) -> pl.DataFrame:
        """Extract angle features from pose landmarks."""
        features = []

        for row in landmarks_df.iter_rows(named=True):
            row_features = {}

            # Extract key joint angles
            row_features['shoulder_angle'] = self._calculate_shoulder_angle(row)
            row_features['hip_angle'] = self._calculate_hip_angle(row)
            # estrada et al.
            row_features['sewangle_a_left'], row_features['sewangle_b_left'], row_features['sewangle_c_left'] = self._calculate_SEWAngleABC_left(row)
            row_features['sewangle_a_right'], row_features['sewangle_b_right'], row_features['sewangle_c_right'] = self._calculate_SEWAngleABC_right(row)
            # lin et al.
            row_features['phi1'] = self._calculate_phi1_angle(row)
            row_features['phi2'] = self._calculate_phi2_angle(row)
            row_features['phi3'] = self._calculate_phi3_angle(row)
            row_features['phi4'] = self._calculate_phi4_angle(row)
            row_features['phi5'] = self._calculate_phi5_angle(row)

            features.append(row_features)

        return pl.DataFrame(features)

    def get_feature_names(self) -> List[str]:
        return [
            'shoulder_angle', 'hip_angle',
            'sewangle_a_left', 'sewangle_b_left', 'sewangle_c_left',
            'sewangle_a_right', 'sewangle_b_right', 'sewangle_c_right',
            'phi1', 'phi2', 'phi3', 'phi4', 'phi5'
        ]

    def _euclidean_distance(self, x1, y1, x2, y2) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def _safe_arccos(self, x):
        """Safely compute arccos to avoid NaN values."""
        return np.arccos(np.clip(x, -1.0, 1.0))

    def _calculate_shoulder_angle(self, row) -> float:
        """Calculate shoulder slope angle."""
        left_shoulder_y = row.get('LEFT_SHOULDER_y', 0)
        right_shoulder_y = row.get('RIGHT_SHOULDER_y', 0)
        left_shoulder_x = row.get('LEFT_SHOULDER_x', 0)
        right_shoulder_x = row.get('RIGHT_SHOULDER_x', 0)

        if all([left_shoulder_y, right_shoulder_y, left_shoulder_x, right_shoulder_x]):
            dy = right_shoulder_y - left_shoulder_y
            dx = right_shoulder_x - left_shoulder_x
            if dx != 0:
                return np.arctan(dy / dx) * 180 / np.pi
        return 0.0

    def _calculate_hip_angle(self, row) -> float:
        """Calculate hip alignment angle."""
        left_hip_y = row.get('LEFT_HIP_y', 0)
        right_hip_y = row.get('RIGHT_HIP_y', 0)
        left_hip_x = row.get('LEFT_HIP_x', 0)
        right_hip_x = row.get('RIGHT_HIP_x', 0)

        if all([left_hip_y, right_hip_y, left_hip_x, right_hip_x]):
            dy = right_hip_y - left_hip_y
            dx = right_hip_x - left_hip_x
            if dx != 0:
                return np.arctan(dy / dx) * 180 / np.pi
        return 0.0

    def _calculate_SEWAngleABC_left(self, row) -> tuple[float, float, float]:
        """Calculate SEW angle for left side."""
        left_shoulder_x = row.get('LEFT_SHOULDER_x', 0)
        left_shoulder_y = row.get('LEFT_SHOULDER_y', 0)
        left_elbow_x = row.get('LEFT_ELBOW_x', 0)
        left_elbow_y = row.get('LEFT_ELBOW_y', 0)
        left_wrist_x = row.get('LEFT_WRIST_x', 0)
        left_wrist_y = row.get('LEFT_WRIST_y', 0)

        sewangle_A, sewangle_B, sewangle_C = 0.0, 0.0, 0.0

        if all([left_shoulder_x, left_shoulder_y, left_elbow_x, left_elbow_y, left_wrist_x, left_wrist_y]):
            # shoulder to elbow distance
            a = self._euclidean_distance(left_shoulder_x, left_shoulder_y, left_elbow_x, left_elbow_y)
            # elbow to wrist distance
            b = self._euclidean_distance(left_elbow_x, left_elbow_y, left_wrist_x, left_wrist_y)
            # shoulder to wrist distance
            c = self._euclidean_distance(left_shoulder_x, left_shoulder_y, left_wrist_x, left_wrist_y)

            sewangle_B = self._safe_arccos(
                (a**2 + b**2 - c**2) / (2 * a * b)
            )

            sewangle_A = self._safe_arccos(
                (b**2 + c**2 - a**2) / (2 * b * c)
            )

            sewangle_C = 180 - sewangle_A - sewangle_B

        return sewangle_A, sewangle_B, sewangle_C

    def _calculate_SEWAngleABC_right(self, row) -> tuple[float, float, float]:
        """Calculate SEW angle for right side."""
        right_shoulder_x = row.get('RIGHT_SHOULDER_x', 0)
        right_shoulder_y = row.get('RIGHT_SHOULDER_y', 0)
        right_elbow_x = row.get('RIGHT_ELBOW_x', 0)
        right_elbow_y = row.get('RIGHT_ELBOW_y', 0)
        right_wrist_x = row.get('RIGHT_WRIST_x', 0)
        right_wrist_y = row.get('RIGHT_WRIST_y', 0)

        sewangle_A, sewangle_B, sewangle_C = 0.0, 0.0, 0.0

        if all([right_shoulder_x, right_shoulder_y, right_elbow_x, right_elbow_y, right_wrist_x, right_wrist_y]):
            # shoulder to elbow distance
            a = self._euclidean_distance(right_shoulder_x, right_shoulder_y, right_elbow_x, right_elbow_y)
            # elbow to wrist distance
            b = self._euclidean_distance(right_elbow_x, right_elbow_y, right_wrist_x, right_wrist_y)
            # shoulder to wrist distance
            c = self._euclidean_distance(right_shoulder_x, right_shoulder_y, right_wrist_x, right_wrist_y)

            sewangle_B = self._safe_arccos(
                (a**2 + b**2 - c**2) / (2 * a * b)
            )

            sewangle_A = self._safe_arccos(
                (b**2 + c**2 - a**2) / (2 * b * c)
            )

            sewangle_C = 180 - sewangle_A - sewangle_B

        return sewangle_A, sewangle_B, sewangle_C

    def _calculate_phi1_angle(self, row) -> float:
        """
        Calculate φ1: angle between neck-to-nose vector and vertical vector.

        φ1 is the angle between:
        - The vector connecting the neck joint to the nose joint
        - A vertical vector (perpendicular to horizontal plane)
        """
        # Get nose coordinates
        nose_x = row.get('NOSE_x', 0)
        nose_y = row.get('NOSE_y', 0)

        # Approximate neck position as midpoint between shoulders
        neck_x = (row.get('LEFT_SHOULDER_x', 0) + row.get('RIGHT_SHOULDER_x', 0)) / 2
        neck_y = (row.get('LEFT_SHOULDER_y', 0) + row.get('RIGHT_SHOULDER_y', 0)) / 2

        if all([nose_x, nose_y, neck_x, neck_y]):
            # Vector from neck to nose (v10)
            v10_x = nose_x - neck_x
            v10_y = nose_y - neck_y

            # Vector perpendicular to horizontal plane (v90) - pointing up
            # In image coordinates, "up" is negative y-direction
            v90_x = 0
            v90_y = -1

            # Calculate dot product
            dot_product = v10_x * v90_x + v10_y * v90_y

            # Calculate magnitudes
            magnitude_v10 = np.sqrt(v10_x**2 + v10_y**2)
            magnitude_v90 = 1  # Unit vector

            # Calculate cosine of the angle
            if magnitude_v10 > 0:
                cos_angle = dot_product / (magnitude_v10 * magnitude_v90)
                # Clamp to [-1, 1] to avoid numerical errors
                cos_angle = max(-1, min(1, cos_angle))
                # Convert to degrees
                phi1 = np.arccos(cos_angle) * 180 / np.pi
                return phi1

        return 0.0

    def _calculate_phi2_angle(self, row) -> float:
        """
        Calculate φ2: angle between right elbow-to-shoulder vector and horizontal vector.

        φ2 is the angle between:
        - The vector connecting the right elbow joint to the right shoulder joint (v32)
        - A vector at 180° to the horizontal plane (i.e., pointing directly right)
        """
        # Get right shoulder coordinates
        right_shoulder_x = row.get('RIGHT_SHOULDER_x', 0)
        right_shoulder_y = row.get('RIGHT_SHOULDER_y', 0)

        # Get right elbow coordinates
        right_elbow_x = row.get('RIGHT_ELBOW_x', 0)
        right_elbow_y = row.get('RIGHT_ELBOW_y', 0)

        if all([right_shoulder_x, right_shoulder_y, right_elbow_x, right_elbow_y]):
            # Vector from right elbow to right shoulder (v32)
            v32_x = right_shoulder_x - right_elbow_x
            v32_y = right_shoulder_y - right_elbow_y

            # Vector at 180° to horizontal plane (v180) - pointing directly right
            v180_x = 1
            v180_y = 0

            # Calculate dot product
            dot_product = v32_x * v180_x + v32_y * v180_y

            # Calculate magnitudes
            magnitude_v32 = np.sqrt(v32_x**2 + v32_y**2)
            magnitude_v180 = 1  # Unit vector

            # Calculate cosine of the angle
            if magnitude_v32 > 0:
                cos_angle = dot_product / (magnitude_v32 * magnitude_v180)
                # Clamp to [-1, 1] to avoid numerical errors
                cos_angle = max(-1, min(1, cos_angle))
                # Convert to degrees
                phi2 = np.arccos(cos_angle) * 180 / np.pi
                return phi2

        return 0.0

    def _calculate_phi4_angle(self, row) -> float:
        """
        Calculate φ4: angle between left elbow-to-shoulder vector and horizontal vector.

        φ4 is the angle between:
        - The vector connecting the left elbow joint to the left shoulder joint
        - A vector at 0° to the horizontal plane (i.e., pointing directly left)
        """
        # Get left shoulder coordinates
        left_shoulder_x = row.get('LEFT_SHOULDER_x', 0)
        left_shoulder_y = row.get('LEFT_SHOULDER_y', 0)

        # Get left elbow coordinates
        left_elbow_x = row.get('LEFT_ELBOW_x', 0)
        left_elbow_y = row.get('LEFT_ELBOW_y', 0)

        if all([left_shoulder_x, left_shoulder_y, left_elbow_x, left_elbow_y]):
            # Vector from left elbow to left shoulder
            v65_x = left_shoulder_x - left_elbow_x
            v65_y = left_shoulder_y - left_elbow_y

            # Vector at 0° to horizontal plane (pointing directly left)
            v0_x = -1  # Pointing left
            v0_y = 0

            # Calculate dot product
            dot_product = v65_x * v0_x + v65_y * v0_y

            # Calculate magnitudes
            magnitude_v32 = np.sqrt(v65_x**2 + v65_y**2)
            magnitude_v0 = 1  # Unit vector

            # Calculate cosine of the angle
            if magnitude_v32 > 0:
                cos_angle = dot_product / (magnitude_v32 * magnitude_v0)
                # Clamp to [-1, 1] to avoid numerical errors
                cos_angle = max(-1, min(1, cos_angle))
                # Convert to degrees
                phi2_left = np.arccos(cos_angle) * 180 / np.pi
                return phi2_left

        return 0.0

    def _calculate_phi3_angle(self, row) -> float:
        """
        Calculate φ3: angle between right wrist-to-elbow vector and vertical vector.

        φ3 is the angle between:
        - The vector connecting the right wrist joint to the right elbow joint (v43)
        - A vector at 90° to the horizontal plane (vertical vector)
        """
        # Get right elbow coordinates
        right_elbow_x = row.get('RIGHT_ELBOW_x', 0)
        right_elbow_y = row.get('RIGHT_ELBOW_y', 0)

        # Get right wrist coordinates
        right_wrist_x = row.get('RIGHT_WRIST_x', 0)
        right_wrist_y = row.get('RIGHT_WRIST_y', 0)

        if all([right_elbow_x, right_elbow_y, right_wrist_x, right_wrist_y]):
            # Vector from right wrist to right elbow (v43)
            v43_x = right_elbow_x - right_wrist_x
            v43_y = right_elbow_y - right_wrist_y

            # Vector perpendicular to horizontal plane (v90) - pointing up
            # In image coordinates, "up" is negative y-direction
            v90_x = 0
            v90_y = -1

            # Calculate dot product
            dot_product = v43_x * v90_x + v43_y * v90_y

            # Calculate magnitudes
            magnitude_v43 = np.sqrt(v43_x**2 + v43_y**2)
            magnitude_v90 = 1  # Unit vector

            # Calculate cosine of the angle
            if magnitude_v43 > 0:
                cos_angle = dot_product / (magnitude_v43 * magnitude_v90)
                # Clamp to [-1, 1] to avoid numerical errors
                cos_angle = max(-1, min(1, cos_angle))
                # Convert to degrees
                phi3 = np.arccos(cos_angle) * 180 / np.pi
                return phi3

        return 0.0

    def _calculate_phi5_angle(self, row) -> float:
        """
        Calculate φ5: angle between left wrist-to-elbow vector and vertical vector.

        φ5 is the angle between:
        - The vector connecting the left wrist joint to the left elbow joint (v76)
        - A vector at 90° to the horizontal plane (vertical vector)
        """
        # Get left elbow coordinates
        left_elbow_x = row.get('LEFT_ELBOW_x', 0)
        left_elbow_y = row.get('LEFT_ELBOW_y', 0)

        # Get left wrist coordinates
        left_wrist_x = row.get('LEFT_WRIST_x', 0)
        left_wrist_y = row.get('LEFT_WRIST_y', 0)

        if all([left_elbow_x, left_elbow_y, left_wrist_x, left_wrist_y]):
            # Vector from left wrist to left elbow (v76)
            v76_x = left_elbow_x - left_wrist_x
            v76_y = left_elbow_y - left_wrist_y

            # Vector perpendicular to horizontal plane (v90) - pointing up
            # In image coordinates, "up" is negative y-direction
            v90_x = 0
            v90_y = -1

            # Calculate dot product
            dot_product = v76_x * v90_x + v76_y * v90_y

            # Calculate magnitudes
            magnitude_v76 = np.sqrt(v76_x**2 + v76_y**2)
            magnitude_v90 = 1  # Unit vector

            # Calculate cosine of the angle
            if magnitude_v76 > 0:
                cos_angle = dot_product / (magnitude_v76 * magnitude_v90)
                # Clamp to [-1, 1] to avoid numerical errors
                cos_angle = max(-1, min(1, cos_angle))
                # Convert to degrees
                phi5 = np.arccos(cos_angle) * 180 / np.pi
                return phi5

        return 0.0

class DistanceFeatureExtractor(FeatureExtractor):
    """Extract distance-based features."""
    def __init__(self):
        super().__init__("distance_features")

    def extract(self, landmarks_df: pl.DataFrame) -> pl.DataFrame:
        """Extract distance features from pose landmarks."""
        features = []

        for row in landmarks_df.iter_rows(named=True):
            row_features = {}

            # Key distance measurements
            row_features['torso_length'] = self._torso_length(row)
            # estrada et al.
            row_features['nose_to_shoulder_left_distance'] = self._nose_to_shoulder_left_distance(row)
            row_features['nose_to_shoulder_right_distance'] = self._nose_to_shoulder_right_distance(row)
            row_features['shoulder_to_elbow_left_distance'] = self._shoulder_to_elbow_left_distance(row)
            row_features['shoulder_to_elbow_right_distance'] = self._shoulder_to_elbow_right_distance(row)
            row_features['elbow_to_wrist_left_distance'] = self._elbow_to_wrist_left_distance(row)
            row_features['elbow_to_wrist_right_distance'] = self._elbow_to_wrist_right_distance(row)
            row_features['wrist_to_shoulder_left_distance'] = self._wrist_to_shoulder_left_distance(row)
            row_features['wrist_to_shoulder_right_distance'] = self._wrist_to_shoulder_right_distance(row)
            row_features['shoulder_to_mid_left_distance'] = self._shoulder_to_mid_left_distance(row)
            row_features['shoulder_to_mid_right_distance'] = self._shoulder_to_mid_right_distance(row)
            row_features['mid_to_middle_hip_distance'] = self._mid_to_middle_hip_distance(row)
            row_features['middle_hip_to_shoulder_left_distance'] = self._middle_hip_to_shoulder_left_distance(row)
            row_features['middle_hip_to_shoulder_right_distance'] = self._middle_hip_to_shoulder_right_distance(row)
            row_features['nose_to_middle_shoulder_distance'] = self._nose_to_middle_shoulder_distance(row)

            features.append(row_features)

        return pl.DataFrame(features)

    def get_feature_names(self) -> List[str]:
        return [
            'torso_length',
            'nose_to_shoulder_left_distance', 'nose_to_shoulder_right_distance',
            'shoulder_to_elbow_left_distance', 'shoulder_to_elbow_right_distance',
            'elbow_to_wrist_left_distance', 'elbow_to_wrist_right_distance',
            'wrist_to_shoulder_left_distance', 'wrist_to_shoulder_right_distance',
            'shoulder_to_mid_left_distance', 'shoulder_to_mid_right_distance',
            'mid_to_middle_hip_distance', 'middle_hip_to_shoulder_left_distance',
            'middle_hip_to_shoulder_right_distance', 'nose_to_middle_shoulder_distance'
        ]

    def _euclidean_distance(self, x1, y1, x2, y2) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def _torso_length(self, row) -> float:
        """Distance from shoulders to hips."""
        shoulder_center_x = (row.get('LEFT_SHOULDER_x', 0) + row.get('RIGHT_SHOULDER_x', 0)) / 2
        shoulder_center_y = (row.get('LEFT_SHOULDER_y', 0) + row.get('RIGHT_SHOULDER_y', 0)) / 2
        hip_center_x = (row.get('LEFT_HIP_x', 0) + row.get('RIGHT_HIP_x', 0)) / 2
        hip_center_y = (row.get('LEFT_HIP_y', 0) + row.get('RIGHT_HIP_y', 0)) / 2

        if all([shoulder_center_x, shoulder_center_y, hip_center_x, hip_center_y]):
            return self._euclidean_distance(shoulder_center_x, shoulder_center_y, hip_center_x, hip_center_y)
        return 0.0

    def _hip_width(self, row) -> float:
        """Distance between hips."""
        left_x = row.get('LEFT_HIP_x', 0)
        left_y = row.get('LEFT_HIP_y', 0)
        right_x = row.get('RIGHT_HIP_x', 0)
        right_y = row.get('RIGHT_HIP_y', 0)

        if all([left_x, left_y, right_x, right_y]):
            return self._euclidean_distance(left_x, left_y, right_x, right_y)
        return 0.0

    def _nose_to_shoulder_left_distance(self, row) -> float:
        """Distance from nose to left shoulder."""
        nose_x = row.get('NOSE_x', 0)
        nose_y = row.get('NOSE_y', 0)
        left_shoulder_x = row.get('LEFT_SHOULDER_x', 0)
        left_shoulder_y = row.get('LEFT_SHOULDER_y', 0)

        if all([nose_x, nose_y, left_shoulder_x, left_shoulder_y]):
            return self._euclidean_distance(nose_x, nose_y, left_shoulder_x, left_shoulder_y)
        return 0.0

    def _nose_to_shoulder_right_distance(self, row) -> float:
        """Distance from nose to right shoulder."""
        nose_x = row.get('NOSE_x', 0)
        nose_y = row.get('NOSE_y', 0)
        right_shoulder_x = row.get('RIGHT_SHOULDER_x', 0)
        right_shoulder_y = row.get('RIGHT_SHOULDER_y', 0)

        if all([nose_x, nose_y, right_shoulder_x, right_shoulder_y]):
            return self._euclidean_distance(nose_x, nose_y, right_shoulder_x, right_shoulder_y)
        return 0.0

    def _shoulder_to_elbow_left_distance(self, row) -> float:
        """Distance from left shoulder to left elbow."""
        left_shoulder_x = row.get('LEFT_SHOULDER_x', 0)
        left_shoulder_y = row.get('LEFT_SHOULDER_y', 0)
        left_elbow_x = row.get('LEFT_ELBOW_x', 0)
        left_elbow_y = row.get('LEFT_ELBOW_y', 0)

        if all([left_shoulder_x, left_shoulder_y, left_elbow_x, left_elbow_y]):
            return self._euclidean_distance(left_shoulder_x, left_shoulder_y, left_elbow_x, left_elbow_y)
        return 0.0

    def _shoulder_to_elbow_right_distance(self, row) -> float:
        """Distance from right shoulder to right elbow."""
        right_shoulder_x = row.get('RIGHT_SHOULDER_x', 0)
        right_shoulder_y = row.get('RIGHT_SHOULDER_y', 0)
        right_elbow_x = row.get('RIGHT_ELBOW_x', 0)
        right_elbow_y = row.get('RIGHT_ELBOW_y', 0)

        if all([right_shoulder_x, right_shoulder_y, right_elbow_x, right_elbow_y]):
            return self._euclidean_distance(right_shoulder_x, right_shoulder_y, right_elbow_x, right_elbow_y)
        return 0.0

    def _elbow_to_wrist_left_distance(self, row) -> float:
        """Distance from left elbow to left wrist."""
        left_elbow_x = row.get('LEFT_ELBOW_x', 0)
        left_elbow_y = row.get('LEFT_ELBOW_y', 0)
        left_wrist_x = row.get('LEFT_WRIST_x', 0)
        left_wrist_y = row.get('LEFT_WRIST_y', 0)

        if all([left_elbow_x, left_elbow_y, left_wrist_x, left_wrist_y]):
            return self._euclidean_distance(left_elbow_x, left_elbow_y, left_wrist_x, left_wrist_y)
        return 0.0

    def _elbow_to_wrist_right_distance(self, row) -> float:
        """Distance from right elbow to right wrist."""
        right_elbow_x = row.get('RIGHT_ELBOW_x', 0)
        right_elbow_y = row.get('RIGHT_ELBOW_y', 0)
        right_wrist_x = row.get('RIGHT_WRIST_x', 0)
        right_wrist_y = row.get('RIGHT_WRIST_y', 0)

        if all([right_elbow_x, right_elbow_y, right_wrist_x, right_wrist_y]):
            return self._euclidean_distance(right_elbow_x, right_elbow_y, right_wrist_x, right_wrist_y)
        return 0.0

    def _wrist_to_shoulder_left_distance(self, row) -> float:
        """Distance from left wrist to left shoulder."""
        left_wrist_x = row.get('LEFT_WRIST_x', 0)
        left_wrist_y = row.get('LEFT_WRIST_y', 0)
        left_shoulder_x = row.get('LEFT_SHOULDER_x', 0)
        left_shoulder_y = row.get('LEFT_SHOULDER_y', 0)

        if all([left_wrist_x, left_wrist_y, left_shoulder_x, left_shoulder_y]):
            return self._euclidean_distance(left_wrist_x, left_wrist_y, left_shoulder_x, left_shoulder_y)
        return 0.0

    def _wrist_to_shoulder_right_distance(self, row) -> float:
        """Distance from right wrist to right shoulder."""
        right_wrist_x = row.get('RIGHT_WRIST_x', 0)
        right_wrist_y = row.get('RIGHT_WRIST_y', 0)
        right_shoulder_x = row.get('RIGHT_SHOULDER_x', 0)
        right_shoulder_y = row.get('RIGHT_SHOULDER_y', 0)

        if all([right_wrist_x, right_wrist_y, right_shoulder_x, right_shoulder_y]):
            return self._euclidean_distance(right_wrist_x, right_wrist_y, right_shoulder_x, right_shoulder_y)
        return 0.0

    def _shoulder_to_mid_left_distance(self, row) -> float:
        """Distance from left shoulder to thoracolumbar"""
        left_shoulder_x = row.get('LEFT_SHOULDER_x', 0)
        left_shoulder_y = row.get('LEFT_SHOULDER_y', 0)
        right_shoulder_x = row.get('RIGHT_SHOULDER_x', 0)
        right_shoulder_y = row.get('RIGHT_SHOULDER_y', 0)
        left_hip_x = row.get('LEFT_HIP_x', 0)
        left_hip_y = row.get('LEFT_HIP_y', 0)
        right_hip_x = row.get('RIGHT_HIP_x', 0)
        right_hip_y = row.get('RIGHT_HIP_y', 0)

        # get the x and y of the thoracolumbar point
        t_x = (left_shoulder_x + right_shoulder_x) / 2
        t_y = (left_shoulder_y + right_shoulder_y) / 2
        l_x = (left_hip_x + right_hip_x) / 2
        l_y = (left_hip_y + right_hip_y) / 2

        # Calculate thoracolumbar point as midway between thoracic and lumbar
        tl_x = t_x + (l_x - t_x) * 0.5
        tl_y = t_y + (l_y - t_y) * 0.5

        if left_shoulder_x and left_shoulder_y and tl_y:
            return self._euclidean_distance(left_shoulder_x, left_shoulder_y, tl_x, tl_y)
        return 0.0

    def _shoulder_to_mid_right_distance(self, row) -> float:
        """Distance from right shoulder to thoracolumbar"""
        left_shoulder_x = row.get('LEFT_SHOULDER_x', 0)
        left_shoulder_y = row.get('LEFT_SHOULDER_y', 0)
        right_shoulder_x = row.get('RIGHT_SHOULDER_x', 0)
        right_shoulder_y = row.get('RIGHT_SHOULDER_y', 0)
        left_hip_x = row.get('LEFT_HIP_x', 0)
        left_hip_y = row.get('LEFT_HIP_y', 0)
        right_hip_x = row.get('RIGHT_HIP_x', 0)
        right_hip_y = row.get('RIGHT_HIP_y', 0)

        # get the x and y of the thoracolumbar point
        t_x = (left_shoulder_x + right_shoulder_x) / 2
        t_y = (left_shoulder_y + right_shoulder_y) / 2
        l_x = (left_hip_x + right_hip_x) / 2
        l_y = (left_hip_y + right_hip_y) / 2

        # Calculate thoracolumbar point as midway between thoracic and lumbar
        tl_x = t_x + (l_x - t_x) * 0.5
        tl_y = t_y + (l_y - t_y) * 0.5

        if right_shoulder_x and right_shoulder_y and tl_y:
            return self._euclidean_distance(right_shoulder_x, right_shoulder_y, tl_x, tl_y)
        return 0.0

    def _mid_to_middle_hip_distance(self, row) -> float:
        """Distance from thoracolumbar to middle hip."""
        left_shoulder_x = row.get('LEFT_SHOULDER_x', 0)
        left_shoulder_y = row.get('LEFT_SHOULDER_y', 0)
        right_shoulder_x = row.get('RIGHT_SHOULDER_x', 0)
        right_shoulder_y = row.get('RIGHT_SHOULDER_y', 0)
        left_hip_x = row.get('LEFT_HIP_x', 0)
        left_hip_y = row.get('LEFT_HIP_y', 0)
        right_hip_x = row.get('RIGHT_HIP_x', 0)
        right_hip_y = row.get('RIGHT_HIP_y', 0)

        # get the x and y of the thoracolumbar point
        t_x = (left_shoulder_x + right_shoulder_x) / 2
        t_y = (left_shoulder_y + right_shoulder_y) / 2
        l_x = (left_hip_x + right_hip_x) / 2
        l_y = (left_hip_y + right_hip_y) / 2

        # Calculate thoracolumbar point as midway between thoracic and lumbar
        tl_x = t_x + (l_x - t_x) * 0.5
        tl_y = t_y + (l_y - t_y) * 0.5

        # Calculate middle hip point as midway between left and right hips
        middle_hip_x = (left_hip_x + right_hip_x) / 2
        middle_hip_y = (left_hip_y + right_hip_y) / 2

        if tl_x and tl_y and middle_hip_x and middle_hip_y:
            return self._euclidean_distance(tl_x, tl_y, middle_hip_x, middle_hip_y)
        return 0.0

    def _middle_hip_to_shoulder_left_distance(self, row) -> float:
        """Distance from middle hip to left shoulder."""
        left_shoulder_x = row.get('LEFT_SHOULDER_x', 0)
        left_shoulder_y = row.get('LEFT_SHOULDER_y', 0)
        left_hip_x = row.get('LEFT_HIP_x', 0)
        left_hip_y = row.get('LEFT_HIP_y', 0)
        right_hip_x = row.get('RIGHT_HIP_x', 0)
        right_hip_y = row.get('RIGHT_HIP_y', 0)

        # Calculate middle hip point as midway between left and right hips
        middle_hip_x = (left_hip_x + right_hip_x) / 2
        middle_hip_y = (left_hip_y + right_hip_y) / 2

        if left_shoulder_x and left_shoulder_y and middle_hip_x and middle_hip_y:
            return self._euclidean_distance(middle_hip_x, middle_hip_y, left_shoulder_x, left_shoulder_y)
        return 0.0

    def _middle_hip_to_shoulder_right_distance(self, row) -> float:
        """Distance from middle hip to right shoulder."""
        right_shoulder_x = row.get('RIGHT_SHOULDER_x', 0)
        right_shoulder_y = row.get('RIGHT_SHOULDER_y', 0)
        left_hip_x = row.get('LEFT_HIP_x', 0)
        left_hip_y = row.get('LEFT_HIP_y', 0)
        right_hip_x = row.get('RIGHT_HIP_x', 0)
        right_hip_y = row.get('RIGHT_HIP_y', 0)

        # Calculate middle hip point as midway between left and right hips
        middle_hip_x = (left_hip_x + right_hip_x) / 2
        middle_hip_y = (left_hip_y + right_hip_y) / 2

        if right_shoulder_x and right_shoulder_y and middle_hip_x and middle_hip_y:
            return self._euclidean_distance(middle_hip_x, middle_hip_y, right_shoulder_x, right_shoulder_y)
        return 0.0

    def _nose_to_middle_shoulder_distance(self, row) -> float:
        """Distance from nose to middle shoulder."""
        nose_x = row.get('NOSE_x', 0)
        nose_y = row.get('NOSE_y', 0)
        left_shoulder_x = row.get('LEFT_SHOULDER_x', 0)
        left_shoulder_y = row.get('LEFT_SHOULDER_y', 0)
        right_shoulder_x = row.get('RIGHT_SHOULDER_x', 0)
        right_shoulder_y = row.get('RIGHT_SHOULDER_y', 0)

        # Calculate middle shoulder point as midway between left and right shoulders
        middle_shoulder_x = (left_shoulder_x + right_shoulder_x) / 2
        middle_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2

        if nose_x and nose_y and middle_shoulder_x and middle_shoulder_y:
            return self._euclidean_distance(nose_x, nose_y, middle_shoulder_x, middle_shoulder_y)
        return 0.0


class SymmetryFeatureExtractor(FeatureExtractor):
    """Extract symmetry-based features."""

    def __init__(self):
        super().__init__("symmetry_features")

    def extract(self, landmarks_df: pl.DataFrame) -> pl.DataFrame:
        """Extract symmetry features from pose landmarks."""
        features = []

        for row in landmarks_df.iter_rows(named=True):
            row_features = {}

            # Symmetry measurements
            row_features['shoulder_symmetry'] = self._shoulder_symmetry(row)
            row_features['hip_symmetry'] = self._hip_symmetry(row)
            row_features['overall_body_symmetry'] = self._overall_body_symmetry(row)

            features.append(row_features)

        return pl.DataFrame(features)

    def get_feature_names(self) -> List[str]:
        return ['shoulder_symmetry', 'hip_symmetry', 'overall_body_symmetry']

    def _shoulder_symmetry(self, row) -> float:
        """Calculate shoulder height symmetry."""
        left_y = row.get('LEFT_SHOULDER_y', 0)
        right_y = row.get('RIGHT_SHOULDER_y', 0)

        if left_y and right_y:
            return abs(left_y - right_y)
        return 0.0

    def _hip_symmetry(self, row) -> float:
        """Calculate hip height symmetry."""
        left_y = row.get('LEFT_HIP_y', 0)
        right_y = row.get('RIGHT_HIP_y', 0)

        if left_y and right_y:
            return abs(left_y - right_y)
        return 0.0

    def _overall_body_symmetry(self, row) -> float:
        """Calculate overall body symmetry score."""
        # This could be a weighted combination of multiple symmetry measures
        shoulder_sym = self._shoulder_symmetry(row)
        hip_sym = self._hip_symmetry(row)

        return (shoulder_sym + hip_sym) / 2


class SpineAlignmentFeatureExtractor(FeatureExtractor):
    """Extract features related to spine alignment and posture."""

    def __init__(self):
        super().__init__("spine_alignment_features")

    def extract(self, landmarks_df: pl.DataFrame) -> pl.DataFrame:
        """Extract spine alignment features from pose landmarks."""
        features = []

        for row in landmarks_df.iter_rows(named=True):
            row_features = {}

            # Calculate the Y-coordinates for the three spine points
            # T (Thoracic) - Use midpoint between shoulders
            t_y = (row.get('LEFT_SHOULDER_y', 0) + row.get('RIGHT_SHOULDER_y', 0)) / 2

            # L (Lumbar) - Use midpoint between hips
            l_y = (row.get('LEFT_HIP_y', 0) + row.get('RIGHT_HIP_y', 0)) / 2

            # TL (Thoraco-Lumbar) - Approximate as midpoint between T and L
            # You can adjust this ratio if you have a better anatomical approximation
            tl_y = t_y + (l_y - t_y) * 0.5  # Using 0.5 for middle point

            # Calculate differences between spine points
            row_features['t_tl_diff_y'] = abs(t_y - tl_y)  # Difference between T and TL
            row_features['tl_l_diff_y'] = abs(tl_y - l_y)  # Difference between TL and L
            row_features['t_l_diff_y'] = abs(t_y - l_y)    # Difference between T and L

            features.append(row_features)

        return pl.DataFrame(features)

    def get_feature_names(self) -> List[str]:
        return ['t_tl_diff_y', 'tl_l_diff_y', 't_l_diff_y']
