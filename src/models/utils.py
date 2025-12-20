import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn_evaluation import plot


def configure_logging(log_dir: str = ""):
    """Initialize Loguru with console + rotating file sinks."""
    if not log_dir:
        # assume this file lives in <project>/src/models/utils.py,
        # so project_root is three levels up
        project_root = Path(__file__).parents[2]
        log_dir_path = project_root / "logs"
    else:
        log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
    )
    logger.add(
        str(log_dir_path / "model_training_{time}.log"),
        rotation="500 MB",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )


def log_confusion_matrix(live, cm, class_names, title=None, cmap="Blues"):
    """Plots the confusion matrix."""
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 11
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        cmap=cmap,
        ax=ax,
        annot_kws={"fontsize": 15},
        xticklabels=class_names,
        yticklabels=class_names,
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    if title:
        ax.set_title(title)
    # log the confusion matrix
    plt.tight_layout()
    live.log_image("test/confusion_matrix.png", fig)


def log_roc_auc_curve(live, y_true, y_pred_proba):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    plot.ROC.from_raw_data(y_true, y_pred_proba, ax=ax)
    plt.tight_layout()
    live.log_image("test/roc_auc_curve.png", fig)


def load_data(view: str, data_path_suffix: str = ""):
    """
    Read train/test CSVs produced by prepare.py using pandas.

    Args:
        view: The view to load ('front', 'left', 'right', 'combined')
        data_path_suffix: Optional suffix for data path (e.g., 'real_keypoints_only')

    Returns:
        X_train (pd.DataFrame), X_test (pd.DataFrame),
        y_train (pd.Series), y_test (pd.Series)
    """
    # Construct the correct data path based on suffix
    if data_path_suffix:
        base_path = os.path.join("data/processed", data_path_suffix, view)
    else:
        base_path = os.path.join("data/processed", view)

    logger.info(f"Loading data from {base_path}")
    train = pd.read_csv(os.path.join(base_path, "train.csv"), index_col=0)
    test = pd.read_csv(os.path.join(base_path, "test.csv"), index_col=0)

    if "labels" not in train.columns or "labels" not in test.columns:
        raise KeyError("Expected 'labels' column in train/test CSVs.")

    X_train = train.drop(columns=["labels"])
    X_test = test.drop(columns=["labels"])
    y_train = train["labels"]
    y_test = test["labels"]
    return X_train, X_test, y_train, y_test


def bayes_search(
    estimator,
    search_spaces: dict,
    X,
    y,
    n_iter: int,
    cv,
    scoring: str,
    live
):
    """
    Run BayesSearchCV, log best params to dvclive, return (best_params, cv_results_).

    X, y can be pandas DataFrame/Series. They will be converted to numpy.
    """
    opt = BayesSearchCV(
        estimator=estimator,
        search_spaces=search_spaces,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=42,
        refit=False,
        n_jobs=-1,
        verbose=1,
    )
    # Ensure numpy arrays for skopt
    X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    y_np = y.to_numpy().ravel() if hasattr(y, "to_numpy") else np.ravel(y)
    opt.fit(X_np, y_np)
    best = opt.best_params_
    live.log_params(best)
    return best, opt.cv_results_


def extract_subject_ids(view: str, X_train, X_test, processed_root: str = "data/processed", data_path_suffix: str = ""):
    """
    From processed CSVs, pull out subject_id for train/test folds using pandas.

    X_train, X_test: pandas DataFrames (or objects with .index attribute)
    Returns:
        groups_train (np.ndarray), groups_test (np.ndarray)
    """
    if data_path_suffix:
        subdir = os.path.join(processed_root, data_path_suffix, view)
    else:
        subdir = os.path.join(processed_root, view)

    df = pd.read_csv(os.path.join(subdir, "data.csv"))

    if "subject_id" not in df.columns:
        raise KeyError("Expected 'subject_id' column in processed data.csv.")

    train_idx = X_train.index
    test_idx = X_test.index
    # check overlap
    assert not set(train_idx).intersection(test_idx), "Train and test sets overlap."

    gtr = df.loc[train_idx, "subject_id"].to_numpy()
    gte = df.loc[test_idx, "subject_id"].to_numpy()
    return gtr, gte


class NumpyEncoder(json.JSONEncoder):
    """Special JSON encoder for numpy types."""
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)
