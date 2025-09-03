import os
import pandas as pd
import polars as pl
from loguru import logger
from .loader import categorize_columns

def is_synthetic(filename):
    """Determine if a file is synthetic based on naming convention"""
    # Real data has no additional metadata in filename
    # Synthetic data has additional metadata like _flux_krea_00001_
    return '_' in os.path.basename(filename).split('.')[0][-12:]

def filter_dataset(df, mode='all'):
    """
    Filter dataset based on ablation mode

    Args:
        df: DataFrame with a 'filename' column
        mode: One of 'real', 'synthetic', or 'all' (real+synthetic)

    Returns:
        Filtered DataFrame
    """
    if mode == 'all':
        logger.info(f"Using all data (real + synthetic): {len(df)} samples")
        return df

    if isinstance(df, pd.DataFrame):
        if mode == 'real':
            filtered = df[~df['file_name'].apply(is_synthetic)]
            logger.info(f"Using real-only data: {len(filtered)} samples")
            return filtered
        elif mode == 'synthetic':
            filtered = df[df['file_name'].apply(is_synthetic)]
            logger.info(f"Using synthetic-only data: {len(filtered)} samples")
            return filtered
    elif isinstance(df, pl.DataFrame):
        if mode == 'real':
            # Alternative fix: Use with_column to create a boolean mask column
            filtered = df.with_columns(
                pl.col('file_name').map_elements(is_synthetic).alias('_is_synthetic')
            ).filter(pl.col('_is_synthetic').not_()).drop('_is_synthetic')
            logger.info(f"Using real-only data: {len(filtered)} samples")
            return filtered
        elif mode == 'synthetic':
            # Alternative fix: Use with_column to create a boolean mask column
            filtered = df.with_columns(
                pl.col('file_name').map_elements(is_synthetic).alias('_is_synthetic')
            ).filter(pl.col('_is_synthetic')).drop('_is_synthetic')
            logger.info(f"Using synthetic-only data: {len(filtered)} samples")
            return filtered

    raise ValueError(f"Unknown mode: {mode}. Use 'real', 'synthetic', or 'all'")

def filter_features(df, feature_mode='all_features'):
    """
    Filter features for ablation study

    Args:
        df: DataFrame containing both keypoints and engineered features
        feature_mode: One of 'keypoints_only' or 'all_features'

    Returns:
        Tuple of (metadata_columns, landmark_columns, feature_columns)
        where feature_columns will be empty if feature_mode='keypoints_only'
    """
    meta_cols, landmark_cols, feature_cols = categorize_columns(df)

    if feature_mode == 'keypoints_only':
        logger.info(f"Using keypoints-only features: {len(landmark_cols)} columns")
        return meta_cols, landmark_cols, []

    elif feature_mode == 'all_features':
        logger.info(f"Using all features: {len(landmark_cols)} keypoints + {len(feature_cols)} engineered features")
        return meta_cols, landmark_cols, feature_cols

    raise ValueError(f"Unknown feature mode: {feature_mode}. Use 'keypoints_only' or 'all_features'")
