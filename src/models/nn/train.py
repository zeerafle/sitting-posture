import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential

from dvclive import Live
from dvclive.keras import DVCLiveCallback

import polars as pl

current_dir = os.path.dirname(os.path.abspath(__file__))
# move 2 level up
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
# add parent dir to path
sys.path.insert(0, parent_dir)

from evaluate import evaluate

X_train = pl.read_csv(os.path.join(parent_dir, '..', "data/processed/X_train.csv"))
y_train = np.load(os.path.join(parent_dir, '..', "data/processed/y_train.npy"))
X_test = pl.read_csv(os.path.join(parent_dir, '..', "data/processed/X_test.csv"))
y_test = np.load(os.path.join(parent_dir, '..', "data/processed/y_test.npy"))
X_val = pl.read_csv(os.path.join(parent_dir, '..', "data/processed/X_val.csv"))
y_val = np.load(os.path.join(parent_dir, '..', "data/processed/y_val.npy"))

model = Sequential([
    layers.Dense(128, activation='relu', input_shape=(26,)),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    # need softmax activation for outputting probability
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 patience=20)

with Live('../../../dvclive/nn', report=None) as live:
    history = model.fit(X_train.to_numpy(), y_train,
                        epochs=200,
                        batch_size=16,
                        validation_data=(X_val.to_numpy(), y_val),
                        callbacks=[earlystopping, DVCLiveCallback(live=live)])
    model.save('model.keras')
    live.log_artifact('model.keras', type='model')

    evaluate(model, X_test, y_test, live)
