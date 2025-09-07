import os
import json
import yaml
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from dvclive import Live
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

from .utils import NumpyEncoder


def loso_training(trainer, data_mode=None):
    """
    Leave-One-Subject-Out for combined or per-view.
    Trains folds, logs per-subject metrics, then
    re-fits on all data and saves final model.

    Args:
        trainer: A BaseTrainer instance
        data_mode: Optional data mode path suffix (e.g., "real_keypoints_only")

    Returns:
        Dictionary with metrics from LOSO evaluation
    """
    # If data_mode is provided, use it, otherwise use trainer's data_path_suffix if available
    data_suffix = data_mode or getattr(trainer, "data_path_suffix", "") or ""

    # Determine which views to use
    if getattr(trainer, "train_combined", False):
        # Check if both loso and combined are true
        views = ["combined"]
    else:
        views = trainer.views
    all_metrics = {}

    for view in views:
        # setup paths
        subdir = view

        # Adjust data path based on data_suffix
        if data_suffix:
            data_path = os.path.join("data/processed", data_suffix, subdir)
            dvclive_loso = os.path.join(trainer.dvclive_dir, f"{subdir}_{data_suffix}_loso")
        else:
            data_path = os.path.join("data/processed", subdir)
            dvclive_loso = os.path.join(trainer.dvclive_dir, f"{subdir}_loso")

        os.makedirs(dvclive_loso, exist_ok=True)
        logger.info(f"Using data from {data_path}")

        # Get project root directory from the trainer's dvclive_dir (which is an absolute path)
        project_root = os.path.abspath(os.path.join(trainer.dvclive_dir, '..', '..'))

        # Check if the data file exists before trying to load it
        data_file_path = os.path.join(project_root, data_path, "data.csv")
        try:
            logger.info(f"Loading data from absolute path: {data_file_path}")
            df = pd.read_csv(data_file_path)
        except Exception as e:
            logger.error(f"Failed to load data from {data_file_path}: {e}")
            continue

        groups = df["subject_id"].to_numpy()
        X_df = df.drop(columns=["subject_id", "class_no", "class_name", "file_name", "view_type"])
        y = df["class_no"].to_numpy()

        # load best params from combined htcv if exists
        params_path = os.path.join(trainer.dvclive_dir, "combined", "params.yaml")
        if os.path.exists(params_path):
            with open(params_path, "r") as f:
                best_params = yaml.safe_load(f) or {}
            best_params.pop("htcv_best_score", None)
        else:
            best_params = {}
            logger.warning(f"No best‐params at {params_path}, using defaults.")

        # convert to numpy
        X = X_df.values
        y = np.ravel(y)

        # per‐subject CV
        cv = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=trainer.params.get("random_state", 42)
        )
        splits = list(cv.split(X, y, groups))

        metrics = {m: [] for m in ("accuracy", "precision", "recall", "f1", "roc_auc")}
        with Live(dvclive_loso) as live:
            if best_params:
                live.log_params(best_params)

            for fold, (tr_idx, te_idx) in enumerate(splits, start=1):
                subj = groups[te_idx[0]]
                logger.info(f"Fold {fold}/5 — holding out subject {subj}")

                Xtr, ytr = X[tr_idx], y[tr_idx]
                Xte, yte = X[te_idx], y[te_idx]

                # build and fit
                model = trainer.get_estimator()
                if best_params:
                    model.set_params(**best_params)
                model.fit(Xtr, ytr)

                # predict & score
                ypred = model.predict(Xte)
                yproba = trainer.get_y_pred_proba(model, Xte)
                acc = accuracy_score(yte, ypred)
                prec = precision_score(yte, ypred)
                rec = recall_score(yte, ypred)
                f1 = f1_score(yte, ypred)
                roc = roc_auc_score(yte, yproba)

                metrics["accuracy"].append(acc)
                metrics["precision"].append(prec)
                metrics["recall"].append(rec)
                metrics["f1"].append(f1)
                metrics["roc_auc"].append(roc)

                logger.info(
                    f"Subject {subj} — acc={acc:.3f}, f1={f1:.3f}, roc_auc={roc:.3f}"
                )

            # aggregate
            for k, vals in metrics.items():
                m, s = np.mean(vals), np.std(vals)
                live.log_metric(f"loso_mean_{k}", m, plot=False)
                live.log_metric(f"loso_std_{k}", s, plot=False)

            # save fold‐wise metrics
            json_path = os.path.join(dvclive_loso, "loso_metrics.json")
            with open(json_path, "w") as f:
                json.dump(metrics, f, indent=2, cls=NumpyEncoder)

            # Store metrics for return
            all_metrics = metrics

        # final model on all data
        logger.info(f"Retraining on all data for view={view}")
        final_model = trainer.get_estimator()
        if best_params:
            final_model.set_params(**best_params)
        final_model.fit(X, y)

        # Save with appropriate name based on data_suffix
        if data_suffix:
            model_path = os.path.join(trainer.models_dir, f"{trainer.model_name}_{view}_{data_suffix}_loso.joblib")
        else:
            model_path = os.path.join(trainer.models_dir, f"{trainer.model_name}_{view}_loso.joblib")

        logger.info(f"Saving final {trainer.model_name} LOSO model for {view} view")
        with open(model_path, "wb") as f:
            joblib.dump(final_model, f)
        logger.success(f"Model saved to {model_path}")
        logger.success(f"Completed LOSO evaluation for view: {view}")

    return all_metrics
