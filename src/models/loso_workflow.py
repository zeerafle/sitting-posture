import os
import json
import yaml
import joblib
import numpy as np
import polars as pl
from loguru import logger
from dvclive import Live
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

from .utils import NumpyEncoder

def loso_training(trainer):
    """
    Leave-One-Subject-Out for combined or per-view.
    Trains folds, logs per-subject metrics, then
    re-fits on all data and saves final model.
    """
    views = ["combined"] if trainer.train_combined else trainer.views
    for view in views:
        # setup paths
        subdir = view if view != "combined" else "combined"
        data_path = os.path.join("data/processed", subdir)
        dvclive_loso = os.path.join(trainer.dvclive_dir, f"{subdir}_loso")
        os.makedirs(dvclive_loso, exist_ok=True)

        # load full DF
        df = pl.read_csv(os.path.join('..', '..', '..', data_path, "data.csv"))
        groups = df["subject_id"].to_numpy()
        X_df = df.drop(["subject_id", "class_no", "class_name"])
        y_df = df.select("class_no")

        # load best params from combined htcv if exists
        params_path = os.path.join(trainer.dvclive_dir, "combined", "params.yaml")
        if os.path.exists(params_path):
            with open(params_path, "r") as f:
                best_params = yaml.safe_load(f)
            best_params.pop("htcv_best_score", None)
        else:
            best_params = {}
            logger.warning(f"No best‐params at {params_path}, using defaults.")

        # convert to numpy
        X = X_df.to_pandas().values
        y = np.ravel(y_df.to_pandas().values)

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
                ypred      = model.predict(Xte)
                yproba     = trainer.get_y_pred_proba(model, Xte)
                acc        = accuracy_score(yte, ypred)
                prec       = precision_score(yte, ypred)
                rec        = recall_score(yte, ypred)
                f1         = f1_score(yte, ypred)
                roc        = roc_auc_score(yte, yproba)

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

        # final model on all data
        logger.info(f"Retraining on all data for view={view}")
        final = trainer.get_estimator()
        if best_params:
            final.set_params(**best_params)
        final.fit(X, y)

        # Save with the specific name expected by DVC for combined LOSO
        model_path = os.path.join(trainer.models_dir, f"{trainer.model_name}_{view}_loso.joblib")
        logger.info(f"Saving final {trainer.model_name} LOSO model for {view} view")
        with open(model_path, "wb") as f:
            joblib.dump(model, f)
        logger.success(f"Model saved to {model_path}")
        logger.success(f"Completed LOSO evaluation for view: {view}")
