import os
import polars as pl
from loguru import logger
from data import BodyPart

class DataFrameCombiner:
    def __init__(self, csv_folder: str, pose_classes: list, view_types: list):
        self.csv_folder = csv_folder
        self.pose_classes = pose_classes
        self.view_types = view_types

    def combine(self, output_path: str):
        dfs = []
        for idx, cls in enumerate(self.pose_classes):
            for view in self.view_types:
                path = os.path.join(self.csv_folder, f"{cls}_{view}.csv")
                if not os.path.exists(path):
                    continue
                df = pl.read_csv(path, has_header=False)
                if df.is_empty():
                    continue
                df = df.with_columns([
                    pl.lit(idx).alias("class_no"),
                    pl.lit(cls).alias("class_name"),
                    (pl.lit(os.path.join(cls, view, "")) + pl.col(df.columns[0])).alias(df.columns[0])
                ])
                dfs.append(df)

        if not dfs:
            logger.error("No CSVs to combine.")
            return pl.DataFrame()

        total = pl.concat(dfs)
        total = total.with_columns(
            pl.col(total.columns[0]).str.extract(r"(\d+)_").cast(pl.Int64).alias("subject_id")
        ).select([total.columns[0], "subject_id"] + total.columns[1:])

        # build header
        coords = sum([[f"{bp.name}_x", f"{bp.name}_y", f"{bp.name}_score"] for bp in BodyPart], [])
        header = ["file_name","subject_id"] + coords + ["view_type","class_no","class_name"]
        total = total.rename({total.columns[i]: header[i] for i in range(len(header))})

        total.write_csv(output_path)
        logger.success(f"Wrote combined CSV ({total.shape[0]}×{total.shape[1]}) to {output_path}")
        return total
