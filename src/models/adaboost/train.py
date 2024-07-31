import os
import sys

from sklearn.ensemble import AdaBoostClassifier
import numpy as np

from dvclive import Live
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))
# move 2 level up
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
# add parent dir to path
sys.path.insert(0, parent_dir)

from evaluate import evaluate
from models.utils import load_data

X_train, y_train, X_test, y_test, X_val, y_val = load_data(
    os.path.join(parent_dir, "../data/processed")
)

with Live(os.path.join(parent_dir, '../dvclive/adaboost')) as live:
    model = AdaBoostClassifier(n_estimators=500, algorithm='SAMME')
    model.fit(X_train, np.ravel(y_train))
    with open('adaboost.joblib', 'wb') as f:
        joblib.dump(model, f)
    live.log_artifact('adaboost.joblib', type='model')

    # Classify pose in the TEST dataset using the trained model
    y_pred_proba = model.decision_function(X_test)
    y_pred = model.predict(X_test)

    evaluate(model, y_test, y_pred, y_pred_proba, live)
