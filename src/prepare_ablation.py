import os
import time
import argparse
import pandas as pd
from loguru import logger
import joblib

from data_preparation.imputer import Imputer
from data_preparation.scaler import FeatureScaler
from data_preparation.embedding import landmarks_to_embedding
from data_preparation.data_filter import filter_dataset, filter_features


def prepare_ablation(data_mode: str, feature_mode: str):
    """
    Prepare data for ablation experiments with specified data and feature modes.
    Uses grouped 5-fold cross validation setup.

    Args:
        data_mode: "real", "synthetic", or "all"
        feature_mode: "keypoints_only" or "all_features"
    """
    t0 = time.time()
    logger.info(f"Starting ablation data preparation (data_mode={data_mode}, feature_mode={feature_mode})")

    # Load the enriched landmarks data (with features)
    data_in = "data/data_with_features.csv"
    if not os.path.exists(data_in):
        raise FileNotFoundError(
            f"{data_in} not found. Please run feature extraction first."
        )

    df = pd.read_csv(data_in)
    logger.info(f"Loaded dataset with shape {df.shape[0]} rows and {df.shape[1]} columns")

    # Filter dataset based on data mode
    df_filtered = filter_dataset(df, mode=data_mode)
    logger.info(f"After filtering ({data_mode}): {df_filtered.shape[0]} rows")

    # Filter features based on feature mode
    meta_cols, landmark_cols, feature_cols = filter_features(df_filtered, feature_mode=feature_mode)
    logger.info(f"Feature filtering ({feature_mode}): {len(landmark_cols)} landmark cols, {len(feature_cols)} feature cols")

    # Check subject distribution after filtering
    subject_counts = df_filtered['subject_id'].value_counts()
    logger.info(f"Found {len(subject_counts)} unique subjects after filtering")
    logger.info(f"Samples per subject: min={subject_counts.min()}, max={subject_counts.max()}, mean={subject_counts.mean():.1f}")

    # Check class distribution by subject
    subject_class_info = df_filtered.groupby('subject_id')['class_no'].first()
    class_counts = subject_class_info.value_counts()
    logger.info(f"Subject distribution by class: {dict(class_counts)}")

    # Process features
    scaler = None
    imp = None
    X_features_df = pd.DataFrame(index=df_filtered.index)

    if feature_cols:
        # Handle missing values
        imp = Imputer()
        df_imputed = imp.fit_transform(df_filtered, feature_cols)

        # Scale features
        scaler = FeatureScaler()
        X_features = scaler.fit_transform(df_imputed[feature_cols].to_numpy())

        X_features_df = pd.DataFrame(
            X_features,
            index=df_imputed.index,
            columns=feature_cols
        )
        logger.info("Applied imputation and scaling to engineered features")
    else:
        df_imputed = df_filtered
        logger.info("No engineered features to process")

    # Create embeddings from landmarks
    embeddings = pd.DataFrame(
        df_imputed[landmark_cols].apply(landmarks_to_embedding, axis=1).tolist(),
        index=df_imputed.index,
        columns=[col for col in landmark_cols if not col.endswith("_score")]
    )
    logger.info("Created landmark embeddings")

    # Combine embeddings + engineered features
    X_combined = pd.concat([embeddings, X_features_df], axis=1) if not X_features_df.empty else embeddings

    # Create final dataset with all required columns for LOSO
    final_df = pd.concat([
        X_combined,
        df_filtered[['subject_id', 'class_no', 'class_name', 'file_name', 'view_type']].reset_index(drop=True)
    ], axis=1)

    # Rename class_no to labels for consistency
    final_df = final_df.rename(columns={'class_no': 'labels'})

    # Verify data integrity
    logger.info("Verifying data integrity for ablation...")

    # Check that each subject has consistent class labels
    subject_label_consistency = final_df.groupby('subject_id')['labels'].nunique()
    inconsistent_subjects = subject_label_consistency[subject_label_consistency > 1]

    if len(inconsistent_subjects) > 0:
        logger.warning(f"Found {len(inconsistent_subjects)} subjects with inconsistent labels:")
        for subj_id, count in inconsistent_subjects.items():
            logger.warning(f"  Subject {subj_id}: {count} different labels")
    else:
        logger.info("All subjects have consistent class labels")

    # Check minimum subjects per class for cross-validation
    subjects_per_class = final_df.groupby('labels')['subject_id'].nunique()
    min_subjects = subjects_per_class.min()
    logger.info(f"Subjects per class: {dict(subjects_per_class)}")

    if min_subjects < 5:
        logger.warning(f"Minimum subjects per class ({min_subjects}) is less than 5. This may affect 5-fold CV.")

    # Create output directory
    output_dir = f"data/processed/ablation/{data_mode}_{feature_mode}"
    os.makedirs(output_dir, exist_ok=True)

    # Save complete dataset for ablation
    data_path = os.path.join(output_dir, "data.csv")
    final_df.to_csv(data_path, index=False)
    logger.info(f"Saved ablation dataset to {data_path}")

    # Save preprocessors
    preprocessors = {
        'scaler': scaler,
        'imputer': imp,
        'feature_cols': feature_cols,
        'landmark_cols': landmark_cols,
        'data_mode': data_mode,
        'feature_mode': feature_mode
    }
    preprocessor_path = os.path.join(output_dir, "preprocessors.joblib")
    joblib.dump(preprocessors, preprocessor_path)
    logger.info(f"Saved preprocessors to {preprocessor_path}")

    # Log final statistics
    logger.info(f"Final dataset shape: {final_df.shape}")
    logger.info(f"Feature columns: {len([col for col in final_df.columns if col not in ['subject_id', 'labels', 'class_name', 'file_name', 'view_type']])}")
    logger.info(f"Unique subjects: {final_df['subject_id'].nunique()}")
    logger.info(f"Class distribution: {dict(final_df['labels'].value_counts())}")

    logger.success(f"Ablation data preparation ({data_mode}_{feature_mode}) completed in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for ablation experiments")
    parser.add_argument(
        "--data-mode",
        type=str,
        choices=["all", "real", "synthetic"],
        required=True,
        help="Data mode: all (real+synthetic), real-only, or synthetic-only"
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        choices=["all_features", "keypoints_only"],
        required=True,
        help="Feature mode: all_features (keypoints+engineered) or keypoints_only"
    )

    args = parser.parse_args()
    prepare_ablation(args.data_mode, args.feature_mode)
