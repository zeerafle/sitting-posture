import os
import sys

from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from skopt import BayesSearchCV
from skopt.space import Integer

from dvclive import Live
import dvc.api
import joblib
import json
from codecarbon import OfflineEmissionsTracker


current_dir = os.path.dirname(os.path.abspath(__file__))
# move 2 level up
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
# add parent dir to path
sys.path.insert(0, parent_dir)

from models.evaluate import evaluate
from models.utils import load_data, NumpyEncoder

dvclive_path = os.path.join(parent_dir, "../dvclive/adaboost")
params = dvc.api.params_show()

X_train, X_test, y_train, y_test = load_data(os.path.join(parent_dir, "../data/processed"))

with Live(dvclive_path) as live:
    param_space = {
        "n_estimators": Integer(
            params["adaboost"]["n_estimators_min"],
            params["adaboost"]["n_estimators_max"],
        ),  # like the paper: varies by dataset size
    }

    # Setup Bayesian optimization
    opt = BayesSearchCV(
        estimator=AdaBoostClassifier(random_state=params["random_state"]),
        search_spaces=param_space,
        n_iter=params["n_iter"],
        cv=params["cv"],
        scoring=params["scoring"],
        random_state=params["random_state"],
        n_jobs=-1,
        verbose=1,
    )

    opt.fit(X_train, np.ravel(y_train))
    model = opt.best_estimator_

    # log hyperparameters
    live.log_params(opt.best_params_)
    live.log_param("htcv_best_score", float(opt.best_score_))
    htcv_results_json_path = os.path.join(dvclive_path, "htcv_results.json")
    with open(htcv_results_json_path, "w") as f:
        json.dump(opt.cv_results_, f, indent=4, cls=NumpyEncoder)
    live.log_artifact(htcv_results_json_path, type="htcv_results")

    # re-train the model with the best hyperparameters
    with OfflineEmissionsTracker(
        output_file=os.path.join(dvclive_path, "emissions.csv")
    ) as training_tracker:
        adaboost = AdaBoostClassifier(
            n_estimators=opt.best_params_["n_estimators"],
            random_state=params["random_state"],
        )
        model = adaboost.fit(X_train, np.ravel(y_train))
    live.log_artifact(
        os.path.join(dvclive_path, "emissions.csv"),
        type="emissions",
    )

    # log metrics
    live.log_metric(
        "estimator_weights_mean", float(np.mean(model.estimator_weights_))
    )
    live.log_metric(
        "feature_importance_mean", float(np.mean(model.feature_importances_))
    )

    # Save the trained model
    models_dir = os.path.join(parent_dir, "../models/adaboost/")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"adaboost.joblib")
    with open(model_path, "wb") as f:
        joblib.dump(model, f)

    # Classify pose in the TEST dataset using the trained model
    y_pred_proba = model.decision_function(X_test)

    # log emissions while inference
    with OfflineEmissionsTracker(
        output_file=os.path.join(
            dvclive_path, "emissions_inference.csv"
        )
    ) as inference_tracker:
        y_pred = model.predict(X_test)
    live.log_artifact(
        os.path.join(dvclive_path, "emissions_inference.csv"),
        type="emissions_inference",
    )

    cv_scores = evaluate(model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, live)

    cv_results_json_path = os.path.join(dvclive_path, "cv_results.json")
    with open(cv_results_json_path, "w") as f:
        json.dump(cv_scores, f, indent=4, cls=NumpyEncoder)
    live.log_artifact(cv_results_json_path, type="cv_results")
