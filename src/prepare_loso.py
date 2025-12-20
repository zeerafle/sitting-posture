import os
import time
import pandas as pd
from loguru import logger

from data_preparation.imputer import Imputer
from data_preparation.scaler import FeatureScaler
from data_preparation.embedding import landmarks_to_embedding
from data_preparation.data_filter import filter_features, is_synthetic


def prepare_loso():
    """
    Prepare data for Leave-One-Subject-Out (LOSO) analysis using grouped 5-fold cross validation.
    Uses combined data only with all subjects and all features.
    """
    t0 = time.time()
    logger.info("Starting LOSO data preparation")

    # Load the enriched landmarks data (with features)
    data_in = "data/data_with_features.csv"
    if not os.path.exists(data_in):
        raise FileNotFoundError(
            f"{data_in} not found. Please run feature extraction first."
        )

    df = pd.read_csv(data_in)
    logger.info(f"Loaded dataset with shape {df.shape[0]} rows and {df.shape[1]} columns")

    # Use all data (real + synthetic) and all features for LOSO
    meta_cols, landmark_cols, feature_cols = filter_features(df, feature_mode="all_features")
    logger.info(f"Using {len(landmark_cols)} landmark cols, {len(feature_cols)} feature cols")

    # Check subject distribution
    subject_counts = df['subject_id'].value_counts()
    logger.info(f"Found {len(subject_counts)} unique subjects")
    logger.info(f"Samples per subject: min={subject_counts.min()}, max={subject_counts.max()}, mean={subject_counts.mean():.1f}")

    # Check synthetic vs real data distribution
    if 'file_name' in df.columns:
        synthetic_mask = df['file_name'].apply(is_synthetic)
        real_count = (~synthetic_mask).sum()
        synthetic_count = synthetic_mask.sum()
        logger.info(f"Data composition: {real_count} real samples, {synthetic_count} synthetic samples")

    # Check class distribution by subject
    subject_class_info = df.groupby('subject_id')['class_no'].first()
    class_counts = subject_class_info.value_counts()
    logger.info(f"Subject distribution by class: {dict(class_counts)}")

    # Process features
    scaler = None
    imp = None
    X_df = pd.DataFrame(index=df.index)

    if feature_cols:
        # Handle missing values
        imp = Imputer()
        df_imputed = imp.fit_transform(df, feature_cols)

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
        df_imputed = df
        X_features_df = pd.DataFrame(index=df.index)

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
        df[['subject_id', 'class_no', 'class_name', 'file_name', 'view_type']].reset_index(drop=True)
    ], axis=1)

    # Rename class_no to labels for consistency
    final_df = final_df.rename(columns={'class_no': 'labels'})

    # Verify no subject leakage in preparation
    logger.info("Verifying data integrity for LOSO...")

    # Check that each subject has exactly 2 class labels (ergonomic and non-ergonomic)
    subject_label_counts = final_df.groupby('subject_id')['labels'].nunique()
    subjects_without_both_classes = subject_label_counts[subject_label_counts != 2]

    if len(subjects_without_both_classes) > 0:
        logger.warning(f"Found {len(subjects_without_both_classes)} subjects without exactly 2 classes:")
        for subj_id, count in subjects_without_both_classes.items():
            subject_labels = final_df[final_df['subject_id'] == subj_id]['labels'].unique()
            logger.warning(f"  Subject {subj_id}: {count} class(es) - {subject_labels}")
    else:
        logger.info("All subjects have both ergonomic and non-ergonomic classes as expected")

    # Create output directory
    output_dir = "data/processed/loso"
    os.makedirs(output_dir, exist_ok=True)

    # Save complete dataset for LOSO
    data_path = os.path.join(output_dir, "data.csv")
    final_df.to_csv(data_path, index=False)
    logger.info(f"Saved LOSO dataset to {data_path}")

    # Save preprocessors
    if scaler is not None or imp is not None:
        import joblib
        preprocessors = {
            'scaler': scaler,
            'imputer': imp,
            'feature_cols': feature_cols,
            'landmark_cols': landmark_cols
        }
        preprocessor_path = os.path.join(output_dir, "preprocessors.joblib")
        joblib.dump(preprocessors, preprocessor_path)
        logger.info(f"Saved preprocessors to {preprocessor_path}")

    # Log final statistics
    logger.info(f"Final dataset shape: {final_df.shape}")
    logger.info(f"Feature columns: {len([col for col in final_df.columns if col not in ['subject_id', 'labels', 'class_name', 'file_name', 'view_type']])}")
    logger.info(f"Unique subjects: {final_df['subject_id'].nunique()}")
    logger.info(f"Class distribution: {dict(final_df['labels'].value_counts())}")

    logger.success(f"LOSO data preparation completed in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    prepare_loso()
