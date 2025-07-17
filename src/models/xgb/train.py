import os
import sys

from xgboost import XGBClassifier
import numpy as np
from skopt import BayesSearchCV
from skopt.space import Categorical, Real

from dvclive import Live
import dvc.api
import json
from codecarbon import OfflineEmissionsTracker


current_dir = os.path.dirname(os.path.abspath(__file__))
# move 2 level up
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
# add parent dir to path
sys.path.insert(0, parent_dir)

from models.utils import NumpyEncoder, load_data
from models.evaluate import evaluate

dvclive_path = os.path.join(parent_dir, "../dvclive/xgb")
params = dvc.api.params_show()

X_train, X_test, y_train, y_test = load_data(os.path.join(parent_dir, "../data/processed"))

with Live(dvclive_path) as live:
    param_space = {
        "n_estimators": Categorical(params["xgb"]["n_estimators"]),
        "learning_rate": Real(
            params["xgb"]["learning_rate_min"], params["xgb"]["learning_rate_max"]
        ),
        "gamma": Real(params["xgb"]["gamma_min"], params["xgb"]["gamma_max"]),
        "max_depth": Categorical(params["xgb"]["max_depths"]),
        "min_child_weight": Categorical(params["xgb"]["min_child_weights"]),
        "subsample": Categorical(params["xgb"]["subsamples"]),
        "lambda": Categorical(params["xgb"]["regulation_lambdas"]),
        "alpha": Categorical(params["xgb"]["regulation_alphas"]),
    }

    # Setup Bayesian optimization
    opt = BayesSearchCV(
        estimator=XGBClassifier(
            random_state=params["random_state"],
        ),
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
    htcv_results_json_path = os.path.join(
        dvclive_path, "htcv_results.json"
    )
    with open(htcv_results_json_path, "w") as f:
        json.dump(opt.cv_results_, f, indent=4, cls=NumpyEncoder)
    live.log_artifact(htcv_results_json_path, type="cv_results")

    # re-train the model with the best hyperparameters
    # also track the emissions
    with OfflineEmissionsTracker(
        output_file=os.path.join(dvclive_path, "emissions.csv")
    ) as training_tracker:
        xgb = XGBClassifier(
            n_estimators=opt.best_params_["n_estimators"],
            learning_rate=opt.best_params_["learning_rate"],
            gamma=opt.best_params_["gamma"],
            max_depth=opt.best_params_["max_depth"],
            min_child_weight=opt.best_params_["min_child_weight"],
            subsample=opt.best_params_["subsample"],
            reg_lambda=opt.best_params_["lambda"],
            reg_alpha=opt.best_params_["alpha"],
            random_state=params["random_state"],
        )
        model = model.fit(X_train, np.ravel(y_train))
    live.log_artifact(
        os.path.join(dvclive_path, "emissions.csv"),
        type="emissions",
    )

    # log metrics
    live.log_metric(
        "feature_importance_mean", float(np.mean(model.feature_importances_))
    )

    # Save the trained model
    models_dir = os.path.join(parent_dir, "../models/xgb/")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"xgb.json")
    model.save_model(model_path)

    # Classify pose in the TEST dataset using the trained model
    y_pred_proba = model.predict_proba(X_test)[:, 1]

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
