import os
import sys
import json
from abc import ABC, abstractmethod

import dvc.api
import joblib
import numpy as np
from codecarbon import OfflineEmissionsTracker
from dvclive import Live
from skopt import BayesSearchCV

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from models.evaluate import evaluate
from models.utils import NumpyEncoder, load_data


class BaseTrainer(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.params = dvc.api.params_show()
        self.dvclive_path = os.path.join(parent_dir, f"../dvclive/{self.model_name}")
        self.models_dir = os.path.join(parent_dir, f"../models/{self.model_name}/")
        self.views = ['front', 'left', 'right']
        os.makedirs(self.models_dir, exist_ok=True)

    @abstractmethod
    def get_estimator(self):
        pass

    @abstractmethod
    def get_param_space(self):
        pass

    def save_model(self, model, view):
        model_path = os.path.join(self.models_dir, f"{self.model_name}_{view}.joblib")
        with open(model_path, "wb") as f:
            joblib.dump(model, f)

    def log_model_specific_metrics(self, model, live):
        pass

    def get_y_pred_proba(self, model, X_test):
        return model.predict_proba(X_test)[:, 1]

    def set_best_params(self, model, best_params):
        model.set_params(**best_params)
        return model

    def run(self):
        for view in self.views:
            X_train, X_test, y_train, y_test = load_data(
                os.path.join(parent_dir, f"../data/processed/{view}"),
            )

            dvclive_path_view = os.path.join(self.dvclive_path, view)

            with Live(dvclive_path_view) as live:
                opt = BayesSearchCV(
                    estimator=self.get_estimator(),
                    search_spaces=self.get_param_space(),
                    n_iter=self.params["n_iter"],
                    cv=self.params["cv"],
                    scoring=self.params["scoring"],
                    random_state=self.params["random_state"],
                    refit=False,
                    n_jobs=-1,
                    verbose=1,
                )

                opt.fit(X_train, np.ravel(y_train))
                best_params = opt.best_params_

                live.log_params(best_params)
                live.log_param("htcv_best_score", float(opt.best_score_))
                htcv_results_json_path = os.path.join(dvclive_path_view, "htcv_results.json")
                with open(htcv_results_json_path, "w") as f:
                    json.dump(opt.cv_results_, f, indent=4, cls=NumpyEncoder)

                model = self.get_estimator()
                model = self.set_best_params(model, best_params)

                with OfflineEmissionsTracker(save_to_file=False) as training_tracker:
                    model.fit(X_train, np.ravel(y_train))
                live.log_metric('train/score', float(model.score(X_train, np.ravel(y_train))), plot=False)
                live.log_metric('train/duration', training_tracker.final_emissions_data.duration, plot=False)
                live.log_metric('train/cpu_power', training_tracker.final_emissions_data.cpu_power, plot=False)
                live.log_metric('train/ram_power', training_tracker.final_emissions_data.ram_power, plot=False)
                live.log_metric('train/cpu_energy', training_tracker.final_emissions_data.cpu_energy, plot=False)
                live.log_metric('train/gpu_energy', training_tracker.final_emissions_data.gpu_energy, plot=False)
                live.log_metric('train/energy_consumed', training_tracker.final_emissions_data.energy_consumed, plot=False)

                self.log_model_specific_metrics(model, live)
                self.save_model(model, view)

                y_pred_proba = self.get_y_pred_proba(model, X_test)

                with OfflineEmissionsTracker(save_to_file=False) as inference_tracker:
                    y_pred = model.predict(X_test)
                live.log_metric('test/duration', inference_tracker.final_emissions_data.duration, plot=False)
                live.log_metric('test/cpu_power', inference_tracker.final_emissions_data.cpu_power, plot=False)
                live.log_metric('test/ram_power', inference_tracker.final_emissions_data.ram_power, plot=False)
                live.log_metric('test/cpu_energy', inference_tracker.final_emissions_data.cpu_energy, plot=False)
                live.log_metric('test/gpu_energy', inference_tracker.final_emissions_data.gpu_energy, plot=False)
                live.log_metric('test/energy_consumed', inference_tracker.final_emissions_data.energy_consumed, plot=False)

                cv_scores = evaluate(
                    model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, live
                )

                cv_results_json_path = os.path.join(dvclive_path_view, "cv_results.json")
                with open(cv_results_json_path, "w") as f:
                    json.dump(cv_scores, f, indent=4, cls=NumpyEncoder)
