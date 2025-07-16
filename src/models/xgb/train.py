import os
import sys

from xgboost import XGBClassifier
import numpy as np
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical, Real

from dvclive import Live
import dvc.api
import json
import time


current_dir = os.path.dirname(os.path.abspath(__file__))
# move 2 level up
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
# add parent dir to path
sys.path.insert(0, parent_dir)

from models.utils import NumpyEncoder, load_data
from models.evaluate import evaluate

params = dvc.api.params_show()

data = load_data(os.path.join(parent_dir, "../data/processed"))

for view in ["front", "left", "right"]:
    X_train = data[view]["X_train"]
    y_train = data[view]["y_train"]
    X_test = data[view]["X_test"]
    y_test = data[view]["y_test"]


    with Live(os.path.join(parent_dir, f"../dvclive/xgb/{view}")) as live:
        param_space = {
            "n_estimators": Categorical(params['xgb']["n_estimators"]),
            "learning_rate": Real(params['xgb']["learning_rate_min"], params['xgb']["learning_rate_max"]),
            "gamma": Real(params['xgb']["gamma_min"], params['xgb']["gamma_max"]),
            "max_depth": Categorical(params['xgb']["max_depths"]),
            "min_child_weight": Categorical(params['xgb']["min_child_weights"]),
            "subsample": Categorical(params['xgb']["subsamples"]),
            "lambda": Categorical(params['xgb']["regulation_lambdas"]),
            "alpha": Categorical(params['xgb']["regulation_alphas"]),
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
        cv_results_json_path = os.path.join(
            parent_dir, f"../dvclive/xgb/{view}/cv_results.json"
        )
        with open(cv_results_json_path, "w") as f:
            json.dump(opt.cv_results_, f, indent=4, cls=NumpyEncoder)
        live.log_artifact(cv_results_json_path, type="cv_results")

        # log metrics
        live.log_metric('feature_importance_mean', float(np.mean(model.feature_importances_)))

        # Save the trained model
        models_dir = os.path.join(parent_dir, "../models/xgb/")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"xgb_{view}.json")
        model.save_model(model_path)

        # Classify pose in the TEST dataset using the trained model
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        start_time = time.time()
        y_pred = model.predict(X_test)
        end_time = time.time()
        live.log_metric("inference_time", end_time - start_time)

        evaluate(model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, live)
