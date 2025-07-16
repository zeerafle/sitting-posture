import os
import sys
import time

from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np

from dvclive import Live
from dvclive.keras import DVCLiveCallback
import keras_tuner
import dvc.api
import json


current_dir = os.path.dirname(os.path.abspath(__file__))
# move 2 level up
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
# add parent dir to path
sys.path.insert(0, parent_dir)

from models.utils import NumpyEncoder, load_data
from models.evaluate import evaluate

params = dvc.api.params_show()

data = load_data(os.path.join(parent_dir, "../data/processed"))

def build_model(hp):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            units=hp.Int(
                "units",
                min_value=params['nn']['first_hidden_layer_sizes_min'],
                max_value=params['nn']['first_hidden_layer_sizes_max']
            ),
            activation='relu'
        ),
        tf.keras.layers.Dense(
            units=hp.Int(
                "units2",
                min_value=params['nn']['second_hidden_layer_sizes_min'],
                max_value=params['nn']['second_hidden_layer_sizes_max']
            ),
            activation='relu'
        ),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', params['nn']['learning_rates'])
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

class CVTuner(keras_tuner.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        # Get the hyperparameters for this trial
        hp = trial.hyperparameters

        # Perform k-fold cross-validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []

        for train_idx, val_idx in kfold.split(X_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            model = self.hypermodel.build(hp)
            history = model.fit(
                X_train_fold, y_train_fold,
                validation_data=(X_val_fold, y_val_fold),
                epochs=params['nn']['epochs'],
                verbose=0
            )
            cv_scores.append(max(history.history['val_accuracy']))

        # Return the mean CV score
        return {'score': np.mean(cv_scores)}

tuner = CVTuner(
    build_model,
    objective='score',
    max_trials=params['n_iter'],
    directory=os.path.join(parent_dir, "../tuner_results/nn"),
    project_name="nn_tuning_cv",
    overwrite=True
)

for view in ["front", "left", "right"]:
    X_train = data[view]["X_train"]
    y_train = data[view]["y_train"]
    X_test = data[view]["X_test"]
    y_test = data[view]["y_test"]

    with Live(os.path.join(parent_dir, f"../dvclive/nn/{view}")) as live:
        tuner.search(
            X_train, np.ravel(y_train),
            epochs=params['nn']['epochs'],
            validation_data=(X_test, np.ravel(y_test)),
            callbacks=[DVCLiveCallback(live)],
            verbose=1
        )
