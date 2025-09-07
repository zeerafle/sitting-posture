import os
from typing import List

import pandas as pd
from loguru import logger

from data import BodyPart


class DataFrameCombiner:
    """
    Combine per-class, per-view keypoints CSVs into a single CSV with a unified header (pandas-only).

    Behavior:
    - Reads raw CSVs with no header for each (class, view) pair: "<class>_<view>.csv"
    - Prefixes the first column (assumed file name) with "<class>/<view>/"
    - Adds "class_no" (ordinal) and "class_name" (string) columns
    - Extracts "subject_id" from the file name (digits before the first underscore) and inserts it
      as the second column
    - Renames columns to the canonical header:
        ["file_name", "subject_id"] +
        [f"{BodyPart.name}_x", f"{BodyPart.name}_y", f"{BodyPart.name}_score" for each BodyPart] +
        ["view_type", "class_no", "class_name"]

    Notes:
    - This assumes the raw CSVs contain all landmark columns in the same ordering expected by the
      header above, plus a column that will become "view_type".
    - If the number of columns does not match the expected header length, the code will rename
      columns positionally up to the minimum of the two lengths and log a warning.
    """

    def __init__(self, csv_folder: str, pose_classes: List[str], view_types: List[str]) -> None:
        self.csv_folder = csv_folder
        self.pose_classes = pose_classes
        self.view_types = view_types

    def combine(self, output_path: str) -> pd.DataFrame:
        dfs: List[pd.DataFrame] = []

        for idx, cls in enumerate(self.pose_classes):
            for view in self.view_types:
                path = os.path.join(self.csv_folder, f"{cls}_{view}.csv")
                if not os.path.exists(path):
                    logger.warning(f"Missing CSV: {path}")
                    continue

                try:
                    # Raw CSVs have no header
                    df = pd.read_csv(path, header=None)
                except pd.errors.EmptyDataError:
                    logger.warning(f"Empty CSV: {path}")
                    continue
                except Exception as e:
                    logger.error(f"Failed to read CSV {path}: {e}")
                    continue

                if df.empty:
                    logger.warning(f"No rows in CSV: {path}")
                    continue

                # Prefix file name (assumed first column) with "<class>/<view>/"
                first_col = df.columns[0]
                df[first_col] = os.path.join(cls, view, "") + df[first_col].astype(str)

                # Add class metadata
                df["class_no"] = idx
                df["class_name"] = cls

                dfs.append(df)

        if not dfs:
            logger.error("No CSVs to combine.")
            return pd.DataFrame()

        # Concatenate all frames
        total = pd.concat(dfs, ignore_index=True)

        # Extract subject_id from file_name (digits before first underscore) and insert at col 1
        file_col = total.columns[0]
        subj = total[file_col].astype(str).str.extract(r"(\d+)_", expand=False)
        subject_id = pd.to_numeric(subj, errors="coerce").astype("Int64")
        total.insert(1, "subject_id", subject_id)

        # Build canonical header
        coords: List[str] = []
        for bp in BodyPart:
            coords.extend([f"{bp.name}_x", f"{bp.name}_y", f"{bp.name}_score"])
        header = ["file_name", "subject_id"] + coords + ["view_type", "class_no", "class_name"]

        # Rename columns by position up to the shortest length
        if len(header) != len(total.columns):
            logger.warning(
                f"Header length mismatch: expected {len(header)} columns, "
                f"got {len(total.columns)}. Renaming by position up to the shortest length."
            )
        rename_count = min(len(header), len(total.columns))
        rename_map = {total.columns[i]: header[i] for i in range(rename_count)}
        total = total.rename(columns=rename_map)

        # Ensure output directory exists and save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        total.to_csv(output_path, index=False)
        logger.success(f"Wrote combined CSV ({total.shape[0]}×{total.shape[1]}) to {output_path}")

        return total
