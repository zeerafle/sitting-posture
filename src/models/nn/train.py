import os
import sys

import tensorflow as tf
from tensorflow.keras import layers, Sequential
import numpy as np

from dvclive import Live
from dvclive.keras import DVCLiveCallback


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

model = Sequential([
    layers.Dense(128, activation="relu", input_shape=(26,)),
    layers.Dropout(0.5),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),
    # need softmax activation for outputting probability
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

earlystopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)

with Live("../../../dvclive/nn", report=None) as live:
    history = model.fit(
        X_train.to_numpy(),
        y_train.to_numpy(),
        epochs=200,
        batch_size=16,
        validation_data=(X_val.to_numpy(), y_val.to_numpy()),
        callbacks=[earlystopping, DVCLiveCallback(live=live)],
    )
    model.save("model.keras")
    live.log_artifact("model.keras", type="model")

    # Classify pose in the TEST dataset using the trained model
    y_pred_proba = model.predict(X_test.to_numpy())
    # convert sigmoid probability to class index
    y_pred = np.where(y_pred_proba > 0.5, 1, 0)

    evaluate(model, y_test.to_numpy(), y_pred, y_pred_proba, live)
