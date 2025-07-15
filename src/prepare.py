import os

import polars as pl
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

from data import BodyPart


def get_center_point(landmarks, left_bodypart, right_bodypart):
    """Calculates the center point of the two given landmarks."""

    left = tf.gather(landmarks, left_bodypart.value, axis=1)
    right = tf.gather(landmarks, right_bodypart.value, axis=1)
    center = left * 0.5 + right * 0.5
    return center


def get_pose_size(landmarks, torso_size_multiplier=2.5):
    """Calculates pose size.
    It is the maximum of two values:
      * Torso size multiplied by `torso_size_multiplier`
      * Maximum distance from pose center to any pose landmark
    """
    # Hips center
    hips_center = get_center_point(
        landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)

    # Shoulders center
    shoulders_center = get_center_point(
        landmarks, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER
    )

    # Torso size as the minimum body size
    torso_size = tf.linalg.norm(shoulders_center - hips_center)

    # Pose center
    pose_center_new = get_center_point(
        landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    pose_center_new = tf.expand_dims(pose_center_new, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to
    # perform substraction
    pose_center_new = tf.broadcast_to(
        pose_center_new, [tf.size(landmarks) // (13 * 2), 13, 2]
    )

    # Dist to pose center
    d = tf.gather(landmarks - pose_center_new, 0,
                  axis=0, name="dist_to_pose_center")
    # Max dist to pose center
    max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))

    # Normalize scale
    pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)

    return pose_size


def get_pose_size_side_view(landmarks, torso_size_multiplier=2.5):
    """Enhanced pose size calculation that's more robust for side views"""
    # Original torso size calculation
    hips_center = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    shoulders_center = get_center_point(landmarks, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER)
    torso_size = tf.linalg.norm(shoulders_center - hips_center)

    # For side view, also consider vertical span as reference
    # Get head and hip points for vertical reference
    nose = tf.gather(landmarks, BodyPart.NOSE.value, axis=1)
    head_to_hip_distance = tf.linalg.norm(nose - hips_center)

    # Use the more stable reference between torso and head-to-hip
    # This helps when shoulder width is compressed in side view
    stable_torso_size = tf.maximum(torso_size, head_to_hip_distance * 0.6)

    # Calculate pose center and max distance as before
    pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    pose_center = tf.expand_dims(pose_center, axis=1)
    pose_center = tf.broadcast_to(pose_center, [tf.size(landmarks) // (13 * 2), 13, 2])

    d = tf.gather(landmarks - pose_center, 0, axis=0)
    max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))

    # Use the stable torso size for scaling
    pose_size = tf.maximum(stable_torso_size * torso_size_multiplier, max_dist)

    return pose_size


def normalize_pose_landmarks(landmarks):
    """Normalizes the landmarks translation by moving the pose center to (0,0)
    and scaling it to a constant pose size.
    """
    # Move landmarks so that the pose center becomes (0,0)
    pose_center = get_center_point(
        landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    pose_center = tf.expand_dims(pose_center, axis=1)
    # Broadcast the pose center to the same size as the landmark vector
    # to perform substraction
    pose_center = tf.broadcast_to(
        pose_center, [tf.size(landmarks) // (13 * 2), 13, 2])
    landmarks = landmarks - pose_center

    # Scale the landmarks to a constant pose size
    pose_size = get_pose_size(landmarks)
    landmarks /= pose_size

    return landmarks


def normalize_pose_landmarks_side_view(landmarks):
    """Normalizes landmarks with better handling for side views"""
    # Step 1: Align shoulders horizontally (basic rotation normalization)
    left_shoulder = tf.gather(landmarks, BodyPart.LEFT_SHOULDER.value, axis=1)
    right_shoulder = tf.gather(landmarks, BodyPart.RIGHT_SHOULDER.value, axis=1)

    # Calculate shoulder angle and rotate to make shoulders horizontal
    shoulder_vector = right_shoulder - left_shoulder
    angle = tf.atan2(shoulder_vector[:, 1], shoulder_vector[:, 0])

    # Create rotation matrix for 2D rotation
    cos_angle = tf.cos(-angle)
    sin_angle = tf.sin(-angle)

    # Apply rotation to all landmarks
    x_rotated = landmarks[:, :, 0] * cos_angle - landmarks[:, :, 1] * sin_angle
    y_rotated = landmarks[:, :, 0] * sin_angle + landmarks[:, :, 1] * cos_angle

    landmarks_rotated = tf.stack([x_rotated, y_rotated], axis=2)

    # Step 2: Center the pose around hip center
    pose_center = get_center_point(landmarks_rotated, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    pose_center = tf.expand_dims(pose_center, axis=1)
    pose_center = tf.broadcast_to(pose_center, [tf.size(landmarks_rotated) // (13 * 2), 13, 2])
    landmarks_centered = landmarks_rotated - pose_center

    # Step 3: Scale using side-view aware pose size
    pose_size = get_pose_size_side_view(landmarks_centered)
    landmarks_normalized = landmarks_centered / pose_size

    return landmarks_normalized


def landmarks_to_embedding(landmarks_and_scores):
    """Converts the input landmarks into a pose embedding with side view normalization."""
    # Reshape the flat input into a matrix with shape=(13, 3)
    reshaped_inputs = tf.reshape(np.array(landmarks_and_scores), (-1, 13, 3))

    # Use side-view aware normalization for 2D landmarks
    landmarks = normalize_pose_landmarks_side_view(reshaped_inputs[:, :, :2])

    # Flatten the normalized landmark coordinates into a vector
    return tuple(tf.reshape(landmarks, (13*2)).numpy())


df = pl.read_csv("data/data.csv")
df.head()

X = df.drop(["file_name", "class_name", "class_no", "view_type"])
X_embedding = X.map_rows(landmarks_to_embedding)
df_to_split = X_embedding.with_columns(
    df.select(
        pl.col('class_no').alias('labels'),
        pl.col('view_type').alias('views')
    )
)

for view in df['view_type'].unique().to_list():
    df_to_split_view = df_to_split.filter(pl.col('views') == view)
    df_to_split_view = df_to_split_view.drop('views')
    train, test = train_test_split(df_to_split_view, test_size=0.2, random_state=42)

    os.makedirs(f"data/processed/{view}", exist_ok=True)

    # write train and test
    train.write_csv(f"data/processed/{view}/train.csv")
    test.write_csv(f"data/processed/{view}/test.csv")
