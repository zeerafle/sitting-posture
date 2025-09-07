from typing import Dict, Tuple, Union
import pandas as pd
from sklearn.model_selection import train_test_split

# Type alias for return structure:
# - {"combined_full": DataFrame} when combined and loso are True
# - {"full": DataFrame} when loso is True
# - {"combined": (train_df, test_df)} when combined is True
# - {view: (train_df, test_df), ...} otherwise
SplitReturn = Dict[str, Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]]


def split_dataset(
    df: pd.DataFrame,
    combined: bool = False,
    test_size: float = 0.2,
    seed: int = 42,
) -> SplitReturn:
    """
    Split dataset into train/test sets using pandas.

    Behavior:
    - If combined only: drop 'view_type' and return a single train/test split under 'combined'
    - Otherwise: split per view_type and return a dict of {view: (train_df, test_df)}

    Parameters:
        df: Input pandas DataFrame
        combined: If True, ignore views and combine all samples
        loso: If True, return full dataset (for later LOSO handling)
        test_size: Fraction for test split
        seed: Random seed for reproducibility

    Returns:
        Dict with keys depending on mode as described above.
    """
    stratify_col = df["class_no"] if "class_no" in df.columns else None
    if combined:
        df2 = df.drop(columns=["view_type"], errors="ignore")
        train_df, test_df = train_test_split(df2, test_size=test_size, random_state=seed, stratify=stratify_col)
        return {"combined": (train_df, test_df)}

    out: SplitReturn = {}
    # Iterate through each unique view and split separately
    for view in df["view_type"].dropna().unique():
        sub = df[df["view_type"] == view].drop(columns=["view_type"], errors="ignore")
        stratify_sub = sub["class_no"] if "class_no" in sub.columns else None
        train_df, test_df = train_test_split(sub, test_size=test_size, random_state=seed, stratify=stratify_sub)
        out[str(view)] = (train_df, test_df)

    return out
