import os
import sys

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
import numpy as np

from dvclive import Live
import dvc.api
import json
import joblib
from codecarbon import OfflineEmissionsTracker


current_dir = os.path.dirname(os.path.abspath(__file__))
# move 2 level up
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
# add parent dir to path
sys.path.insert(0, parent_dir)

from models.utils import NumpyEncoder, load_data
from models.evaluate import evaluate

dvclive_path = os.path.join(parent_dir, "../dvclive/nn")
params = dvc.api.params_show()

X_train, X_test, y_train, y_test = load_data(os.path.join(parent_dir, "../data/processed"))


class MLPWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, layer1=10, learning_rate_init=0.001):
        self.layer1 = layer1
        self.learning_rate_init = learning_rate_init

    def fit(self, X, y):
        model = MLPClassifier(
            hidden_layer_sizes=[self.layer1],
            batch_size=params["nn"]["batch_size"],
            max_iter=params["nn"]["epochs"],
            random_state=params["random_state"],
            learning_rate_init=self.learning_rate_init,
        )
        model.fit(X, y)
        self.model = model
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)


with Live(dvclive_path) as live:
    param_space = {
        "layer1": Integer(
            params["nn"]["first_hidden_layer_sizes_min"],
            params["nn"]["first_hidden_layer_sizes_max"],
        ),
        "learning_rate_init": Categorical(params["nn"]["learning_rates"]),
    }
    # Setup Bayesian optimization
    opt = BayesSearchCV(
        estimator=MLPWrapper(),
        search_spaces=param_space,
        n_iter=params["n_iter"],
        cv=params["cv"],
        scoring=params["scoring"],
        random_state=params["random_state"],
        n_jobs=-1,
        verbose=1,
    )

    opt.fit(X_train, np.ravel(y_train))

    # log hyperparameters
    live.log_params(opt.best_params_)
    live.log_param("htcv_best_score", float(opt.best_score_))
    htcv_results_json_path = os.path.join(
        dvclive_path, "cv_results.json"
    )
    with open(htcv_results_json_path, "w") as f:
        json.dump(opt.cv_results_, f, indent=4, cls=NumpyEncoder)
    live.log_artifact(htcv_results_json_path, type="htcv_results")

    # re-train the model with the best hyperparameters
    with OfflineEmissionsTracker(
        output_file=os.path.join(dvclive_path, "emissions.csv")
    ) as training_tracker:
        mlp = MLPClassifier(
            hidden_layer_sizes=(
                opt.best_params_["layer1"],
            ),
            batch_size=params["nn"]["batch_size"],
            max_iter=params["nn"]["epochs"],
            random_state=params["random_state"],
            learning_rate_init=opt.best_params_["learning_rate_init"],
        )
        model = mlp.fit(X_train, np.ravel(y_train))
    live.log_artifact(
        os.path.join(dvclive_path, "emissions.csv"),
        type="emissions",
    )

    # log metrics
    if hasattr(model, "best_loss_") and model.best_loss_ is not None:
        live.log_metric("best_loss", float(model.best_loss_))
    if hasattr(model, "loss_curve_") and model.loss_curve_ is not None:
        for loss in model.loss_curve_:
            live.log_metric("loss_curve", float(loss))

    # Save the trained model
    models_dir = os.path.join(parent_dir, "../models/nn/")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"nn.joblib")
    with open(model_path, "wb") as f:
        joblib.dump(model, f)

   # Classify pose in the TEST dataset using the trained model
    y_pred_proba = model.predict_proba(X_test)[:, 1]

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
