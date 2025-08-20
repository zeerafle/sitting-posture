import os
import sys
import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from loguru import logger

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", "..", '..'))
sys.path.insert(0, parent_dir)

from models.base_trainer import BaseTrainer

class CNNTrainer(BaseTrainer):
    def __init__(self, model_name="cnn", train_combined=True, use_loso=True):
        super().__init__(model_name, train_combined, use_loso)
        self.data_dir = "/teamspace/studios/01-data-download-kaggle/.cache/kagglehub/datasets/zeerafle/sitting-posture/versions/6"
        # Load mapping of image paths to subject IDs
        self.subject_mapping = self._load_subject_mapping()

    def _load_subject_mapping(self):
        """Load mapping between images and subject IDs"""
        # Read subject info from data.csv
        df = pd.read_csv(os.path.join(parent_dir, "data", "data.csv"))
        # Create a dictionary mapping image paths to subject IDs
        mapping = {}

        # Create a mapping from filename to subject_id
        # The file_name column has relative paths like 'ergonomis/front/01_DSC02412.JPG_00001_.png'
        filename_to_subject = {row['file_name']: row['subject_id'] for _, row in df.iterrows()}

        # Walk through the dataset directory and match filenames to subject IDs
        for root, dirs, files in tf.io.gfile.walk(self.data_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = f"{root}/{file}"

                    # Try to find a matching entry in our mapping
                    for rel_path, subject_id in filename_to_subject.items():
                        # Check if the relative path is at the end of the absolute path
                        if file_path.endswith(rel_path) or file_path.endswith(rel_path.replace('\\', '/')):
                            mapping[file_path] = subject_id
                            break

        logger.info(f"Created mapping for {len(mapping)} images to {len(set(mapping.values()))} unique subjects")
        if len(mapping) == 0:
            logger.warning("No subject mappings found! Check if file paths match between data.csv and image files")

        return mapping

    def get_estimator(self):
        """Return a CNN model based on MobileNetV2"""
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

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def get_param_space(self):
        """Return hyperparameter search space (not used for CNN)"""
        # For simplicity, we're not doing hyperparameter tuning in this example
        return {}

    def get_y_pred_proba(self, model, X):
        """Get prediction probabilities"""
        return model.predict(X, verbose=0)
