import os
import time
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from data_preparation.imputer import Imputer
from data_preparation.scaler import FeatureScaler
from data_preparation.embedding import landmarks_to_embedding
from data_preparation.data_filter import filter_features, is_synthetic


def prepare_standard_split():
    """
    Prepare data for standard train-test split using combined data only.
    Creates stratified split while preserving subject separation.
    """
    t0 = time.time()
    logger.info("Starting standard train-test split data preparation")

    # Load the enriched landmarks data (with features)
    data_in = "data/data_with_features.csv"
    if not os.path.exists(data_in):
        raise FileNotFoundError(
            f"{data_in} not found. Please run feature extraction first."
        )

    df = pd.read_csv(data_in)
    logger.info(f"Loaded dataset with shape {df.shape[0]} rows and {df.shape[1]} columns")

    # Use all data (real + synthetic) and all features for standard split
    meta_cols, landmark_cols, feature_cols = filter_features(df, feature_mode="all_features")
    logger.info(f"Using {len(landmark_cols)} landmark cols, {len(feature_cols)} feature cols")

    # Check synthetic vs real data distribution
    if 'file_name' in df.columns:
        synthetic_mask = df['file_name'].apply(is_synthetic)
        real_count = (~synthetic_mask).sum()
        synthetic_count = synthetic_mask.sum()
        logger.info(f"Data composition: {real_count} real samples, {synthetic_count} synthetic samples")

    # Create stratified train-test split by subject to prevent subject leakage
    # Group by subject and get one row per subject for stratification
    subject_info = df.groupby('subject_id').agg({
        'class_no': 'first',  # Assuming all samples from same subject have same class
    }).reset_index()

    # Split subjects into train/test (80/20)
    train_subjects, test_subjects = train_test_split(
        subject_info['subject_id'].values,
        test_size=0.2,
        random_state=42,
        stratify=subject_info['class_no'].values
    )

    logger.info(f"Split {len(train_subjects)} training subjects, {len(test_subjects)} test subjects")

    # Filter data based on subject split
    train_df = df[df['subject_id'].isin(train_subjects)].copy()
    test_df = df[df['subject_id'].isin(test_subjects)].copy()

    logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    # Process features
    scaler = None
    imp = None
    Xtr_df = pd.DataFrame(index=train_df.index)
    Xte_df = pd.DataFrame(index=test_df.index)

    if feature_cols:
        # Handle missing values
        imp = Imputer()
        train_imputed = imp.fit_transform(train_df, feature_cols)
        test_imputed = imp.transform(test_df, feature_cols)

        # Scale features
        scaler = FeatureScaler()
        Xtr = scaler.fit_transform(train_imputed[feature_cols].to_numpy())
        Xte = scaler.transform(test_imputed[feature_cols].to_numpy())

        Xtr_df = pd.DataFrame(
            Xtr,
            index=train_imputed.index,
            columns=feature_cols
        )
        Xte_df = pd.DataFrame(
            Xte,
            index=test_imputed.index,
            columns=feature_cols
        )
        logger.info("Applied imputation and scaling to engineered features")
    else:
        train_imputed = train_df
        test_imputed = test_df

    # Create embeddings from landmarks
    emb_tr = pd.DataFrame(
        train_imputed[landmark_cols].apply(landmarks_to_embedding, axis=1).tolist(),
        index=train_imputed.index,
        columns=[col for col in landmark_cols if not col.endswith("_score")]
    )
    emb_te = pd.DataFrame(
        test_imputed[landmark_cols].apply(landmarks_to_embedding, axis=1).tolist(),
        index=test_imputed.index,
        columns=[col for col in landmark_cols if not col.endswith("_score")]
    )
    logger.info("Created landmark embeddings")

    # Combine embeddings + engineered features
    out_tr = pd.concat([emb_tr, Xtr_df], axis=1) if not Xtr_df.empty else emb_tr
    out_te = pd.concat([emb_te, Xte_df], axis=1) if not Xte_df.empty else emb_te

    # Add labels
    if "class_no" in train_imputed.columns:
        out_tr["labels"] = train_imputed["class_no"].values
        out_te["labels"] = test_imputed["class_no"].values
    else:
        raise KeyError("'class_no' column not found for labels")

    # assign file_name and subject_id to train and test datasets
    out_tr["file_name"] = train_df["file_name"]
    out_tr["subject_id"] = train_df["subject_id"]
    out_te["file_name"] = test_df["file_name"]
    out_te["subject_id"] = test_df["subject_id"]

    # Create output directory
    output_dir = "data/processed/standard_split"
    os.makedirs(output_dir, exist_ok=True)

    # Save train and test datasets
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    out_tr.to_csv(train_path, index=False)
    out_te.to_csv(test_path, index=False)

    logger.info(f"Saved training data to {train_path}")
    logger.info(f"Saved test data to {test_path}")

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

    logger.success(f"Standard train-test split preparation completed in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    prepare_standard_split()
