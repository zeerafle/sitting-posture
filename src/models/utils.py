import os
import sys
import json
import yaml
import joblib
import numpy as np
import polars as pl
from loguru import logger
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn_evaluation import plot
from pathlib import Path


def configure_logging(log_dir: str = None):
    """Initialize Loguru with console + rotating file sinks."""
    if log_dir is None:
        # assume this file lives in <project>/src/models/utils.py,
        # so project_root is three levels up
        project_root = Path(__file__).parents[2]
        log_dir = project_root / "logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
    )
    logger.add(
        str(log_dir / "model_training_{time}.log"),
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


def load_data(base_path: str):
    """Read train/test CSVs produced by prepare.py."""
    train = pl.read_csv(os.path.join(base_path, "train.csv"))
    test  = pl.read_csv(os.path.join(base_path, "test.csv"))
    X_train = train.drop("labels")
    X_test  = test.drop("labels")
    y_train = train.select("labels")
    y_test  = test.select("labels")
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
    opt.fit(X.to_numpy(), np.ravel(y.to_numpy()))
    best = opt.best_params_
    live.log_params(best)
    return best, opt.cv_results_


def extract_subject_ids(view: str, X_train, X_test, processed_root: str = "data/processed"):
    """
    From your processed CSVs, pull out subject_id for train/test folds.
    """
    subdir = view if view != "combined" else "combined"
    df = pl.read_csv(os.path.join(processed_root, subdir, "data.csv"))
    train_idx = X_train.row_index
    test_idx  = X_test.row_index
    gtr = df["subject_id"].take(train_idx).to_numpy()
    gte = df["subject_id"].take(test_idx).to_numpy()
    return gtr, gte


class NumpyEncoder(json.JSONEncoder):
    """Special JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
