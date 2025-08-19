import os
import json
import numpy as np
import polars as pl
from codecarbon import OfflineEmissionsTracker
from dvclive import Live
from .evaluate import evaluate
from .utils    import load_data, bayes_search, extract_subject_ids

def run_standard(trainer):
    """
    Runs the standard train/test split path.
    """
    # decide which views to do
    views = ["combined"] if trainer.train_combined else trainer.views

    for view in views:
        Xtr, Xte, ytr, yte = load_data(view)
        # get subject‐IDs for cross‐validation
        groups_train, groups_test = extract_subject_ids(view, Xtr, Xte)
        _train_one_view(
            trainer, view,
            Xtr, Xte, ytr, yte,
            groups_train=groups_train,
            groups_test=groups_test
        )

def _train_one_view(
    trainer, view,
    X_train, X_test,
    y_train, y_test,
    groups_train=None, groups_test=None
):
    """
    Hyperparam search → final fit → emissions‐tracked train & inference → evaluate → save.
    """
    params        = trainer.params
    dvclive_path  = os.path.join(trainer.dvclive_dir, view)
    os.makedirs(dvclive_path, exist_ok=True)

    with Live(dvclive_path) as live:
        # 1) BayesSearchCV
        best_params, cv_results = bayes_search(
            trainer.get_estimator(),
            trainer.get_param_space(),
            X_train, y_train,
            params["n_iter"], params["cv"], params["scoring"],
            live
        )
        # persist cv_results …
        with open(os.path.join(dvclive_path,"htcv_results.json"),"w") as f:
            json.dump(cv_results, f, indent=2)

        # 2) final fit w/ emissions
        model = trainer.get_estimator()
        model.set_params(**best_params)
        with OfflineEmissionsTracker(save_to_file=False) as trk:
            model.fit(X_train, y_train.ravel())
        live.log_metric("train/emissions", trk.final_emissions_data.energy_consumed)

        # 3) inference w/ emissions
        with OfflineEmissionsTracker(save_to_file=False) as trk2:
            y_pred    = model.predict(X_test)
            y_proba   = model.predict_proba(X_test)[:,1]
        live.log_metric("test/emissions", trk2.final_emissions_data.energy_consumed)

        # 4) evaluate & log
        metrics = evaluate(
            model,
            X_train, X_test,
            y_train, y_test,
            y_pred, y_proba,
            live,
            groups_train=groups_train,
            groups_test=groups_test
        )
        with open(os.path.join(dvclive_path,"cv_results.json"),"w") as f:
            json.dump(metrics, f, indent=2)

    # 5) save final model
    trainer.save_model(model, f"{trainer.model_name}_{view}.joblib")


def run_loso(trainer):
    """
    Leave‐One‐Subject‐Out loop: for each subject, train & eval, then aggregate.
    """
    from .loso_workflow import loso_training
    loso_training(trainer)
