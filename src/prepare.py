import os
import argparse
import pandas as pd
import polars as pl
from sklearn.impute import SimpleImputer
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


def prepare_data(combined=False):
    """Prepare data for training - either by view or combined"""

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

    # FIRST: Split into train/test before any preprocessing
    if combined:
        df_work = df.drop('view_type')
        train_df, test_df = train_test_split(df_work, test_size=0.2, random_state=42)
    else:
        # For individual views, we'll process them separately
        train_dfs = {}
        test_dfs = {}

        for view in df.select('view_type').unique().to_series():
            view_df = df.filter(df["view_type"] == view).drop('view_type')
            train_view, test_view = train_test_split(view_df, test_size=0.2, random_state=42)
            train_dfs[view] = train_view
            test_dfs[view] = test_view

    def process_dataset(train_data, test_data, save_path, view_name=""):
        """Process a single dataset (train/test pair)"""

        # Convert to pandas for imputation
        train_pandas = train_data.to_pandas()
        test_pandas = test_data.to_pandas()

        # Handle NaN values using imputation by class_name groups (on TRAINING data only)
        print(f"Handling NaN values for {view_name}...")

        # Check for NaNs in training data
        train_nan_count = train_pandas[feature_cols].isna().sum().sum()
        test_nan_count = test_pandas[feature_cols].isna().sum().sum()
        print(f"Training NaN count before imputation: {train_nan_count}")
        print(f"Test NaN count before imputation: {test_nan_count}")

        group_imputers = {}
        overall_imputer = None

        if train_nan_count > 0 or test_nan_count > 0:
            # Show NaN counts by column in training data
            nan_counts = train_pandas[feature_cols].isna().sum()
            nan_cols = nan_counts[nan_counts > 0]
            if len(nan_cols) > 0:
                print("Training columns with NaN values:")
                for col, count in nan_cols.items():
                    print(f"  - {col}: {count} NaNs")

            # Fit imputers on TRAINING data only
            imputed_train = train_pandas[feature_cols].copy()

            for class_name in train_pandas['class_name'].unique():
                group_mask = train_pandas['class_name'] == class_name

                if group_mask.sum() == 0:
                    continue

                print(f"Processing group: {class_name} ({group_mask.sum()} samples)")

                group_features = train_pandas.loc[group_mask, feature_cols]

                if len(group_features) > 0 and group_features.isna().any().any():
                    imputer = SimpleImputer(strategy='mean')
                    non_nan_cols = group_features.columns[group_features.notna().any()]

                    if len(non_nan_cols) > 0:
                        try:
                            # Fit on training data
                            imputer.fit(group_features[non_nan_cols])

                            # Transform training data
                            imputed_group = imputer.transform(group_features[non_nan_cols])
                            imputed_train.loc[group_mask, non_nan_cols] = imputed_group

                            # Store the fitted imputer
                            group_imputers[class_name] = {
                                'imputer': imputer,
                                'columns': non_nan_cols.tolist(),
                                'class_name': class_name
                            }

                        except Exception as e:
                            print(f"  Warning: Could not impute for group {class_name}: {e}")
                            # Fallback to overall mean
                            for col in non_nan_cols:
                                if group_features[col].isna().any():
                                    overall_mean = train_pandas[col].mean()
                                    if not np.isnan(overall_mean):
                                        imputed_train.loc[group_mask & train_pandas[col].isna(), col] = overall_mean
                                    else:
                                        imputed_train.loc[group_mask & train_pandas[col].isna(), col] = 0.0

            # Handle any remaining NaNs with overall imputer fitted on training data
            remaining_nans = imputed_train.isna().sum().sum()
            if remaining_nans > 0:
                print(f"Handling {remaining_nans} remaining NaN values with overall mean...")
                overall_imputer = SimpleImputer(strategy='mean')
                imputed_train = pd.DataFrame(
                    overall_imputer.fit_transform(imputed_train),
                    columns=feature_cols,
                    index=imputed_train.index
                )

            # Apply the SAME imputers to test data
            imputed_test = test_pandas[feature_cols].copy()

            for class_name in test_pandas['class_name'].unique():
                if class_name in group_imputers:
                    group_mask = test_pandas['class_name'] == class_name
                    if group_mask.sum() > 0:
                        imputer_info = group_imputers[class_name]
                        imputer = imputer_info['imputer']
                        cols = imputer_info['columns']

                        try:
                            imputed_group = imputer.transform(test_pandas.loc[group_mask, cols])
                            imputed_test.loc[group_mask, cols] = imputed_group
                        except Exception as e:
                            print(f"Warning: Could not apply imputer to test group {class_name}: {e}")

            # Apply overall imputer to test data if needed
            if overall_imputer is not None:
                imputed_test = pd.DataFrame(
                    overall_imputer.transform(imputed_test),
                    columns=feature_cols,
                    index=imputed_test.index
                )

            # Replace the original features
            train_pandas[feature_cols] = imputed_train
            test_pandas[feature_cols] = imputed_test

            # Final check
            train_nan_after = train_pandas[feature_cols].isna().sum().sum()
            test_nan_after = test_pandas[feature_cols].isna().sum().sum()
            print(f"Training NaN count after imputation: {train_nan_after}")
            print(f"Test NaN count after imputation: {test_nan_after}")

        # Convert back to polars
        train_data = pl.from_pandas(train_pandas)
        test_data = pl.from_pandas(test_pandas)

        # Process landmarks for both train and test
        train_landmarks_df = train_data.select(landmark_cols)
        test_landmarks_df = test_data.select(landmark_cols)

        train_embeddings = train_landmarks_df.map_rows(landmarks_to_embedding)
        test_embeddings = test_landmarks_df.map_rows(landmarks_to_embedding)

        # Normalize engineered features - FIT ON TRAIN, TRANSFORM BOTH
        train_features_array = train_data.select(feature_cols).to_numpy()
        test_features_array = test_data.select(feature_cols).to_numpy()

        # Check for zero variance columns in training data
        std_devs = np.std(train_features_array, axis=0)
        zero_std_columns = np.where(std_devs == 0)[0]

        if len(zero_std_columns) > 0:
            print(f"Warning: {len(zero_std_columns)} columns have zero variance in training data")
            for col_idx in zero_std_columns:
                print(f"  - {feature_cols[col_idx]}")

            # Add small epsilon to avoid division by zero in both train and test
            for col_idx in zero_std_columns:
                train_features_array[:, col_idx] += np.random.normal(0, 0.001, size=train_features_array.shape[0])
                test_features_array[:, col_idx] += np.random.normal(0, 0.001, size=test_features_array.shape[0])

        # Fit scaler on training data only
        scaler = StandardScaler()
        normalized_train_features = scaler.fit_transform(train_features_array)
        normalized_test_features = scaler.transform(test_features_array)

        # Final NaN checks
        train_nan_final = np.isnan(normalized_train_features).sum()
        test_nan_final = np.isnan(normalized_test_features).sum()
        print(f"Training NaN count after scaling: {train_nan_final}")
        print(f"Test NaN count after scaling: {test_nan_final}")

        # Replace any remaining NaNs
        if np.isnan(normalized_train_features).any():
            print("Replacing remaining training NaNs with 0.0...")
            normalized_train_features = np.nan_to_num(normalized_train_features, nan=0.0)

        if np.isnan(normalized_test_features).any():
            print("Replacing remaining test NaNs with 0.0...")
            normalized_test_features = np.nan_to_num(normalized_test_features, nan=0.0)

        # Create DataFrames
        normalized_train_df = pl.DataFrame(normalized_train_features, schema=feature_cols)
        normalized_test_df = pl.DataFrame(normalized_test_features, schema=feature_cols)

        # Combine embeddings and features
        embedding_cols = [f"embedding_{i}" for i in range(26)]
        train_embeddings_df = pl.DataFrame([list(e) for e in train_embeddings], schema=embedding_cols)
        test_embeddings_df = pl.DataFrame([list(e) for e in test_embeddings], schema=embedding_cols)

        train_combined = pl.concat([train_embeddings_df, normalized_train_df], how="horizontal")
        test_combined = pl.concat([test_embeddings_df, normalized_test_df], how="horizontal")

        # Add labels
        final_train = train_combined.with_columns(train_data.select(pl.col('class_no').alias('labels')))
        final_test = test_combined.with_columns(test_data.select(pl.col('class_no').alias('labels')))

        # Save data and preprocessors
        os.makedirs(save_path, exist_ok=True)
        final_train.write_csv(f"{save_path}/train.csv")
        final_test.write_csv(f"{save_path}/test.csv")

        # Save preprocessors
        os.makedirs(f"{save_path}/preprocessors", exist_ok=True)
        import joblib
        joblib.dump(scaler, f"{save_path}/preprocessors/feature_scaler.joblib")
        joblib.dump(group_imputers, f"{save_path}/preprocessors/group_imputers.joblib")
        if overall_imputer is not None:
            joblib.dump(overall_imputer, f"{save_path}/preprocessors/overall_imputer.joblib")

        joblib.dump({
            'feature_cols': feature_cols,
            'landmark_cols': landmark_cols,
            'embedding_cols': embedding_cols
        }, f"{save_path}/preprocessors/feature_info.joblib")

        print(f"{view_name} data saved with {len(train_combined.columns)} total features")
        return len(train_combined.columns)

    # Process datasets
    if combined:
        total_features = process_dataset(train_df, test_df, "data/processed/combined", "Combined")
    else:
        for view in train_dfs.keys():
            total_features = process_dataset(train_dfs[view], test_dfs[view], f"data/processed/{view}", view)

    print(f"- {26} normalized landmark coordinates")
    print(f"- {len(feature_cols)} normalized engineered features")
    print("- All preprocessors saved for inference use")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument("--combined", action="store_true", help="Prepare combined view data")
    args = parser.parse_args()

    prepare_data(combined=args.combined)
