import polars as pl

def load_dataset(path: str) -> pl.DataFrame:
    return pl.read_csv(path)

def categorize_columns(df: pl.DataFrame):
    metadata = ["file_name", "subject_id", "class_name", "class_no", "view_type"]
    landmark = [c for c in df.columns
                if any(k in c for k in ["NOSE","EYE","EAR","SHOULDER","ELBOW","WRIST","HIP"])]
    features = [c for c in df.columns if c not in metadata + landmark]
    return metadata, landmark, features
