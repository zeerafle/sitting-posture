from sklearn.model_selection import train_test_split
import polars as pl

def split_dataset(df: pl.DataFrame,
                  combined: bool=False,
                  loso: bool=False,
                  test_size: float=0.2,
                  seed: int=42):
    if loso:
        return {"full": df}
    if combined:
        df2 = df.drop("view_type")
        t, v = train_test_split(df2, test_size=test_size, random_state=seed)
        return {"combined": (pl.from_pandas(t), pl.from_pandas(v))}
    out = {}
    for view in df.select("view_type").unique().to_series():
        sub = df.filter(pl.col("view_type")==view).drop("view_type")
        t, v = train_test_split(sub.to_pandas(), test_size=test_size, random_state=seed)
        out[view] = (pl.from_pandas(t), pl.from_pandas(v))
    return out
