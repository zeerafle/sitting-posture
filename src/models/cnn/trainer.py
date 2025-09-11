import os
import sys
import tensorflow as tf
from loguru import logger
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", "..", '..'))
sys.path.insert(0, parent_dir)

from models.base_trainer import BaseTrainer

class CNNTrainer(BaseTrainer):
    def __init__(self, model_name="cnn", train_combined=True, use_loso=True):
        super().__init__(model_name, train_combined, use_loso)
        self.data_dir = "/teamspace/studios/01-data-download-kaggle/.cache/kagglehub/datasets/zeerafle/sitting-posture/versions/6"

    def get_estimator(self):
        """Return a CNN model based on MobileNetV2"""
        try:
            logger.info("Creating MobileNetV2 model...")

            mobilenet = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                weights='imagenet',
                include_top=False,
                pooling='avg'
            )

            model = tf.keras.Sequential([
                mobilenet,
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            # Freeze the base model
            for layer in mobilenet.layers:
                layer.trainable = False

            # Print model summary for debugging
            model.summary(print_fn=logger.info)

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            return model

        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise

    def get_param_space(self):
        """Return hyperparameter search space (not used for CNN)"""
        # For simplicity, we're not doing hyperparameter tuning in this example
        return {}

    def get_y_pred_proba(self, model, X):
        """Get prediction probabilities"""
        try:
            logger.debug(f"Making predictions with input type: {type(X)}")

            # Ensure X has the right shape and type
            if not isinstance(X, np.ndarray):
                logger.warning("Converting X to numpy array")
                X = np.array(X)

            # Ensure 4D tensor shape
            if len(X.shape) == 3:
                X = np.expand_dims(X, axis=0)

            return model.predict(X, verbose=1)
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
