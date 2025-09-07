import pandas as pd
from typing import List, Tuple


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a CSV dataset using pandas.

    Args:
        path: Path to the CSV file.

    Returns:
        A pandas DataFrame containing the loaded data.
    """
    # Keep defaults simple; callers can pre-clean the CSV if needed
    return pd.read_csv(path)


def categorize_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Categorize columns into metadata, landmark, and feature columns.

    Args:
        df: A pandas DataFrame.

    Returns:
        A tuple of (metadata_columns, landmark_columns, feature_columns)
    """
    metadata = ["file_name", "subject_id", "class_name", "class_no", "view_type"]
    landmark = [
        c
        for c in df.columns
        if any(k in c for k in ["NOSE", "EYE", "EAR", "SHOULDER", "ELBOW", "WRIST", "HIP"])
    ]
    features = [c for c in df.columns if c not in set(metadata + landmark)]
    return metadata, landmark, features
