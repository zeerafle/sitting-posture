import os
import joblib
import pandas as pd


def save_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame, save_path: str, view: str) -> None:
    os.makedirs(os.path.join(save_path, view), exist_ok=True)
    train_df.to_csv(os.path.join(save_path, view, "train.csv"))
    test_df.to_csv(os.path.join(save_path, view, "test.csv"))


def save_preprocessors(scaler, imputer, feature_cols, landmark_cols, save_path: str) -> None:
    p = os.path.join(save_path, "preprocessors")
    os.makedirs(p, exist_ok=True)
    joblib.dump(scaler.scaler, os.path.join(p, "scaler.joblib"))
    joblib.dump(imputer.group_imp, os.path.join(p, "group_imputers.joblib"))
    if getattr(imputer, "overall_imp", None):
        joblib.dump(imputer.overall_imp, os.path.join(p, "overall_imputer.joblib"))
    info = {"feature_cols": feature_cols, "landmark_cols": landmark_cols}
    joblib.dump(info, os.path.join(p, "feature_info.joblib"))
