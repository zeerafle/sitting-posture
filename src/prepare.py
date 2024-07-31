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


def landmarks_to_embedding(landmarks_and_scores):
    """Converts the input landmarks into a pose embedding."""
    # Reshape the flat input into a matrix with shape=(13, 3)
    reshaped_inputs = tf.reshape(np.array(landmarks_and_scores), (-1, 13, 3))

    # Normalize landmarks 2D
    landmarks = normalize_pose_landmarks(reshaped_inputs[:, :, :2])

    # Flatten the normalized landmark coordinates into a vector
    return tuple(tf.reshape(landmarks, (13*2)).numpy())


df = pl.read_csv("data/data.csv")
df.head()

X = df.drop(["file_name", "class_name", "class_no"])
X_embedding = X.map_rows(landmarks_to_embedding)
df_to_split = X_embedding.with_columns(
    df.select(
        pl.col('class_no').alias('labels')
    )
)

train, test = train_test_split(df_to_split, test_size=0.2, random_state=42)
test, val = train_test_split(test, test_size=0.5, random_state=42)

os.makedirs("data/processed", exist_ok=True)

# write train, val and test
train.write_csv("data/processed/train.csv")
val.write_csv("data/processed/val.csv")
test.write_csv("data/processed/test.csv")
