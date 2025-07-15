import os
import sys

from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from skopt import BayesSearchCV
from skopt import Integer

from dvclive import Live
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))
# move 2 level up
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
# add parent dir to path
sys.path.insert(0, parent_dir)

from models.evaluate import evaluate
from models.utils import load_data

data = load_data(
    os.path.join(parent_dir, "../data/processed")
)

for view in ['front', 'left', 'right']:
    X_train = data[view]['X_train']
    y_train = data[view]['y_train']
    X_test = data[view]['X_test']
    y_test = data[view]['y_test']

    with Live(os.path.join(parent_dir, f'../dvclive/adaboost/{view}')) as live:
        param_space = {
            'n_estimators': Integer(10, 2000),  # like the paper: varies by dataset size
        }

        # Setup Bayesian optimization
        opt = BayesSearchCV(
            estimator=AdaBoostClassifier(random_state=42),
            search_spaces=param_space,
            n_iter=30,
            cv=5,
            scoring='accuracy',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        opt.fit(X_train, np.ravel(y_train))
        model = opt.best_estimator_

        # log hyperparameters
        live.log_params(opt.best_params_)
        live.log_params(opt.cv_results_)

        # log metrics
        live.log_metric('estimator_weights_mean', np.mean(model.estimator_weights_))
        live.log_metric('feature_importance_mean', np.mean(model.feature_importances_))

        # Save the trained model
        models_dir = os.path.join(parent_dir, '../models/adaboost/')
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f'adaboost_{view}.joblib')
        with open(model_path, 'wb') as f:
            joblib.dump(model, f)
        live.log_artifact(model_path, type='model')

        # Classify pose in the TEST dataset using the trained model
        y_pred_proba = model.decision_function(X_test)
        y_pred = model.predict(X_test)

        evaluate(model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, live)
