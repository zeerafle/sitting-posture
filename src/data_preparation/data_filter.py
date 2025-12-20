import os
from typing import Tuple
import pandas as pd
from loguru import logger
from .loader import categorize_columns


def is_synthetic(filename) -> bool:
    """
    Determine if a file is synthetic based on naming convention.

    Real data has no additional metadata in the filename.
    Synthetic data typically includes extra tokens like '_flux_krea_00001_'.
    """
    if filename is None or (isinstance(filename, float) and pd.isna(filename)):
        return False
    try:
        base = os.path.basename(str(filename))
    except Exception:
        base = str(filename)
    stem = base.split(".")[0]
    return "_" in stem[-12:]

def filter_dataset(df, mode: str = "all") -> pd.DataFrame:
    """
    Filter dataset based on ablation mode.

    Args:
        df: DataFrame with a 'file_name' column
        mode: One of 'real', 'synthetic', or 'all' (real+synthetic)

    Returns:
        A pandas DataFrame filtered according to the selected mode
    """
    pdf = df.copy()

    if "file_name" not in pdf.columns:
        logger.warning("Column 'file_name' not found; returning data unchanged.")
        return pdf.copy()

    if mode == "all":
        logger.info(f"Using all data (real + synthetic): {len(pdf)} samples")
        return pdf.copy()

    syn_mask = pdf["file_name"].apply(is_synthetic)

    if mode == "real":
        filtered = pdf.loc[~syn_mask].copy()
        logger.info(f"Using real-only data: {len(filtered)} samples")
        return filtered
    elif mode == "synthetic":
        filtered = pdf.loc[syn_mask].copy()
        logger.info(f"Using synthetic-only data: {len(filtered)} samples")
        return filtered

    raise ValueError(f"Unknown mode: {mode}. Use 'real', 'synthetic', or 'all'")

def filter_features(df, feature_mode: str = "all_features") -> Tuple[list[str], list[str], list[str]]:
    """
    Filter features for ablation study.

    Args:
        df: DataFrame containing both keypoints and engineered features
        feature_mode: One of 'keypoints_only' or 'all_features'

    Returns:
        Tuple of (metadata_columns, landmark_columns, feature_columns)
        where feature_columns will be empty if feature_mode='keypoints_only'
    """
    pdf = df.copy()

    meta_cols, landmark_cols, feature_cols = categorize_columns(pdf)

    if feature_mode == "keypoints_only":
        logger.info(f"Using keypoints-only features: {len(landmark_cols)} columns")
        return meta_cols, landmark_cols, []

    elif feature_mode == "all_features":
        logger.info(
            f"Using all features: {len(landmark_cols)} keypoints + {len(feature_cols)} engineered features"
        )
        return meta_cols, landmark_cols, feature_cols

    raise ValueError(
        f"Unknown feature mode: {feature_mode}. Use 'keypoints_only' or 'all_features'"
    )
