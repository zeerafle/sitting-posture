import os
import sys

import xgboost as xgb
import numpy as np

from dvclive import Live


current_dir = os.path.dirname(os.path.abspath(__file__))
# move 2 level up
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
# add parent dir to path
sys.path.insert(0, parent_dir)

from models.utils import load_data
from evaluate import evaluate

X_train, y_train, X_test, y_test, X_val, y_val = load_data(
    os.path.join(parent_dir, "../data/processed")
)

with Live(os.path.join(parent_dir, "../dvclive/xgb")) as live:
    clf = xgb.XGBClassifier()
    clf.fit(X_train, np.ravel(y_train))
    clf.save_model("model.json")
    live.log_artifact("model.json", type="model")

    # Classify pose in the TEST dataset using the trained model
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    evaluate(clf, y_test, y_pred, y_pred_proba, live)
