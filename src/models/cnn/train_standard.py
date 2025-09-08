import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
from loguru import logger

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, parent_dir)

from models.base_trainer import BaseTrainer
from models.cnn.trainer import CNNTrainer


class CNNStandardTrainer(BaseTrainer):
    def __init__(self):
        super().__init__(
            model_name="cnn",
            train_combined=True,  # Always use combined data
            use_loso=False        # This is for standard split
        )
        self.cnn_trainer = CNNTrainer()

    def get_estimator(self):
        return self.cnn_trainer.get_estimator()

    def get_param_space(self):
        # CNN uses fixed architecture for standard training
        return {}

    def load_image_data(self, file_paths, labels):
        """Load and preprocess images"""
        def load_image(file_path):
            try:
                raw = tf.io.read_file(file_path)
                tensor = tf.io.decode_image(raw, channels=3)
                tensor = tf.cast(tensor, tf.float32) / 255.0
                tensor = tf.image.resize_with_pad(tensor, 224, 224)
                return tf.keras.applications.mobilenet_v2.preprocess_input(tensor)
            except Exception as e:
                logger.error(f"Error loading image {file_path}: {e}")
                # Return a black image as fallback
                return tf.zeros((224, 224, 3), dtype=tf.float32)

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        dataset = dataset.map(
            lambda path, label: (tf.py_function(load_image, [path], tf.float32), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

        return dataset

    def prepare_data(self):
        """Prepare CNN data from processed standard split"""
        # Load the subject-to-image mapping from data.csv
        data_csv_path = "data/data.csv"
        if not os.path.exists(data_csv_path):
            raise FileNotFoundError(f"Data CSV not found: {data_csv_path}")

        df = pd.read_csv(data_csv_path)

        # Load standard split subject assignments
        train_csv = "data/processed/standard_split/train.csv"
        test_csv = "data/processed/standard_split/test.csv"

        if not os.path.exists(train_csv) or not os.path.exists(test_csv):
            raise FileNotFoundError("Standard split data not found. Please run prepare_standard_split first.")

        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)

        # Get unique subject IDs from train/test splits
        train_subjects = set()
        test_subjects = set()

        # Extract subject IDs from the processed data
        # We need to map back to the original data to get file paths
        base_data_dir = "/teamspace/studios/01-data-download-kaggle/.cache/kagglehub/datasets/zeerafle/sitting-posture/versions/6"

        train_files = []
        train_labels = []
        test_files = []
        test_labels = []

        # Create mapping from processed data back to original images
        # This is complex because we need to map subjects back to their images

        # For simplicity, let's use the subject_mapping approach from CNNTrainer
        subject_mapping = self.cnn_trainer.subject_mapping

        # Reverse mapping: subject_id -> list of file paths
        subject_to_files = {}
        for file_path, subject_id in subject_mapping.items():
            if subject_id not in subject_to_files:
                subject_to_files[subject_id] = []
            subject_to_files[subject_id].append(file_path)

        # Get train/test subject splits (approximate from processed data)
        # Since the processed data doesn't directly contain subject IDs,
        # we'll use a different approach: split the CNN data using the same random state

        from sklearn.model_selection import train_test_split

        # Get all files and their labels
        all_files = []
        all_labels = []
        all_subjects = []

        for file_path, subject_id in subject_mapping.items():
            # Determine label from file path
            if "non-ergonomis" in file_path:
                label = 1  # non-ergonomic
            elif "ergonomis" in file_path:
                label = 0  # ergonomic
            else:
                continue  # skip files we can't classify

            all_files.append(file_path)
            all_labels.append(label)
            all_subjects.append(subject_id)

        # Convert to numpy arrays
        all_files = np.array(all_files)
        all_labels = np.array(all_labels)
        all_subjects = np.array(all_subjects)

        # Get unique subjects and their labels for stratification
        subject_df = pd.DataFrame({
            'subject_id': all_subjects,
            'label': all_labels
        })

        subject_labels = subject_df.groupby('subject_id')['label'].first()
        unique_subjects = subject_labels.index.values
        subject_label_values = subject_labels.values

        # Split subjects (same as in prepare_standard.py)
        train_subj, test_subj = train_test_split(
            unique_subjects,
            test_size=0.2,
            random_state=42,
            stratify=subject_label_values
        )

        # Get file indices for train/test subjects
        train_mask = np.isin(all_subjects, train_subj)
        test_mask = np.isin(all_subjects, test_subj)

        train_files = all_files[train_mask]
        train_labels = all_labels[train_mask]
        test_files = all_files[test_mask]
        test_labels = all_labels[test_mask]

        logger.info(f"CNN data prepared: {len(train_files)} train, {len(test_files)} test images")
        logger.info(f"Train subjects: {len(train_subj)}, Test subjects: {len(test_subj)}")

        return train_files, train_labels, test_files, test_labels

    def run(self):
        """Run the standard CNN training workflow"""
        from codecarbon import OfflineEmissionsTracker
        from dvclive import Live
        import time
        import json
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        logger.info("Starting CNN standard workflow")

        # Prepare data
        train_files, train_labels, test_files, test_labels = self.prepare_data()

        # Create datasets
        train_dataset = self.load_image_data(train_files, train_labels)
        test_dataset = self.load_image_data(test_files, test_labels)

        # Setup DVCLive logging
        dvclive_path = os.path.join(self.dvclive_dir, "standard")
        os.makedirs(dvclive_path, exist_ok=True)

        with Live(dvclive_path) as live:
            # Get model (no hyperparameter tuning for CNN in this version)
            model = self.get_estimator()

            # Track training emissions
            with OfflineEmissionsTracker(save_to_file=False) as train_tracker:
                start_time = time.time()

                # Train model
                history = model.fit(
                    train_dataset,
                    epochs=10,
                    validation_split=0.1,
                    verbose=1
                )

                training_time = time.time() - start_time

            live.log_metric("train/time_seconds", training_time)
            live.log_metric("train/emissions_kwh", train_tracker.final_emissions_data.energy_consumed)

            # Track inference emissions
            with OfflineEmissionsTracker(save_to_file=False) as inference_tracker:
                start_time = time.time()

                # Get predictions
                y_proba = model.predict(test_dataset, verbose=0).flatten()
                y_pred = (y_proba > 0.5).astype(int)

                inference_time = time.time() - start_time

            live.log_metric("test/time_seconds", inference_time)
            live.log_metric("test/emissions_kwh", inference_tracker.final_emissions_data.energy_consumed)

            # Calculate metrics
            test_accuracy = accuracy_score(test_labels, y_pred)
            test_precision = precision_score(test_labels, y_pred, average='binary')
            test_recall = recall_score(test_labels, y_pred, average='binary')
            test_f1 = f1_score(test_labels, y_pred, average='binary')
            test_roc_auc = roc_auc_score(test_labels, y_proba)

            # Log metrics
            live.log_metric("test_accuracy", test_accuracy)
            live.log_metric("test_precision", test_precision)
            live.log_metric("test_recall", test_recall)
            live.log_metric("test_f1", test_f1)
            live.log_metric("test_roc_auc", test_roc_auc)

            # Log training history
            if hasattr(history, 'history'):
                for epoch, acc in enumerate(history.history.get('accuracy', [])):
                    live.log_metric(f"train_accuracy_epoch_{epoch}", acc)
                for epoch, loss in enumerate(history.history.get('loss', [])):
                    live.log_metric(f"train_loss_epoch_{epoch}", loss)

            # Save test predictions
            predictions_df = pd.DataFrame({
                'y_true': test_labels,
                'y_pred': y_pred,
                'y_proba': y_proba
            })
            predictions_path = os.path.join(dvclive_path, "test_predictions.csv")
            predictions_df.to_csv(predictions_path, index=False)

            # Save metrics summary
            metrics = {
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'test_roc_auc': test_roc_auc,
                'training_time': training_time,
                'inference_time': inference_time
            }

            logger.info(f"CNN evaluation completed. Test accuracy: {test_accuracy:.4f}")

        # Save final trained model
        model_filename = f"{self.model_name}_standard.keras"
        model_path = os.path.join(self.models_dir, model_filename)
        model.save(model_path)
        logger.success(f"Saved CNN model to {model_path}")

        logger.success("CNN standard workflow completed successfully")
        return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN model with standard train-test split")
    args = parser.parse_args()

    trainer = CNNStandardTrainer()
    model, metrics = trainer.run()
