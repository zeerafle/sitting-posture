import os
import polars as pl
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    """Converts the input landmarks into a pose embedding with side view normalization."""
    # Reshape the flat input into a matrix with shape=(13, 3)
    reshaped_inputs = tf.reshape(np.array(landmarks_and_scores), (-1, 13, 3))

    landmarks = normalize_pose_landmarks(reshaped_inputs[:, :, :2])

    # Flatten the normalized landmark coordinates into a vector
    return tuple(tf.reshape(landmarks, (13*2)).numpy())


# Load the data with all features
df = pl.read_csv("data/data_with_features.csv")

# Split the dataframe into landmark coordinates and engineered features
landmark_cols = [col for col in df.columns if
                any(part in col for part in ["NOSE", "EYE", "EAR", "SHOULDER",
                                           "ELBOW", "WRIST", "HIP"])]

# Keep metadata columns separate
metadata_cols = ["file_name", "class_name", "class_no", "view_type"]

# Engineered feature columns (all columns except landmarks and metadata)
feature_cols = [col for col in df.columns if col not in landmark_cols + metadata_cols]

print(f"Number of landmark columns: {len(landmark_cols)}")
print(f"Number of engineered feature columns: {len(feature_cols)}")

# 1. Process landmark coordinates with the embedding function
landmarks_df = df.select(landmark_cols)
embeddings = landmarks_df.map_rows(landmarks_to_embedding)

# 2. Normalize engineered features using standard scaling
features_df = df.select(feature_cols)
features_array = features_df.to_numpy()

# Apply standard scaling to engineered features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features_array)
normalized_features_df = pl.DataFrame(normalized_features, schema=feature_cols)

# 3. Combine normalized embeddings and normalized engineered features
# First convert embeddings to a proper DataFrame with column names
embedding_cols = [f"embedding_{i}" for i in range(26)]  # 13 landmarks * 2 coordinates
embeddings_df = pl.DataFrame([list(e) for e in embeddings], schema=embedding_cols)

# Combine all normalized features
combined_features = pl.concat([embeddings_df, normalized_features_df], how="horizontal")

# 4. Add back the class labels
df_to_split = combined_features.with_columns(
    df.select(
        pl.col('class_no').alias('labels'),
        pl.col('view_type')
    )
)

for view in df.select('view_type').unique().to_series():
    df_to_split_view = df_to_split.filter(df["view_type"] == view).drop('view_type')

    # Split into train and test sets
    train, test = train_test_split(df_to_split_view, test_size=0.2, random_state=42)

    # Save the processed data
    os.makedirs(f"data/processed/{view}", exist_ok=True)
    train.write_csv(f"data/processed/{view}/train.csv")
    test.write_csv(f"data/processed/{view}/test.csv")

    # Save the scaler for future use (if needed for inference)
    import joblib
    os.makedirs(f"data/processed/{view}/scalers", exist_ok=True)
    joblib.dump(scaler, f"data/processed/{view}/scalers/feature_scaler.joblib")

print(f"Processed data saved with {len(combined_features.columns)} total features")
print(f"- {len(embedding_cols)} normalized landmark coordinates")
print(f"- {len(feature_cols)} normalized engineered features")
