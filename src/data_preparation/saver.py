import os, joblib
import polars as pl

def save_datasets(train_df, test_df, save_path, view):
    os.makedirs(save_path, exist_ok=True)
    train_df.write_csv(f"{save_path}/train.csv")
    test_df.write_csv(f"{save_path}/test.csv")

def save_preprocessors(scaler, imputer, feature_cols, landmark_cols, save_path):
    p = os.path.join(save_path, "preprocessors")
    os.makedirs(p, exist_ok=True)
    joblib.dump(scaler.scaler, os.path.join(p,"scaler.joblib"))
    joblib.dump(imputer.group_imp, os.path.join(p,"group_imputers.joblib"))
    if imputer.overall_imp:
      joblib.dump(imputer.overall_imp, os.path.join(p,"overall_imputer.joblib"))
    info = {"feature_cols":feature_cols, "landmark_cols":landmark_cols}
    joblib.dump(info, os.path.join(p,"feature_info.joblib"))
