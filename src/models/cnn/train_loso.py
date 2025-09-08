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


class CNNLOSOTrainer(BaseTrainer):
    def __init__(self):
        super().__init__(
            model_name="cnn",
            train_combined=True,  # Always use combined data
            use_loso=True         # This is for LOSO
        )
        self.cnn_trainer = CNNTrainer()

    def get_estimator(self):
        return self.cnn_trainer.get_estimator()

    def get_param_space(self):
        # CNN uses fixed architecture for LOSO
        return {}

    def load_image(self, file_path):
        """Load and preprocess a single image"""
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

    def create_dataset(self, file_paths, labels, batch_size=32):
        """Create a TensorFlow dataset from file paths and labels"""
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        dataset = dataset.map(
            lambda path, label: (tf.py_function(self.load_image, [path], tf.float32), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    def prepare_data(self):
        """Prepare CNN data for LOSO evaluation"""
        base_data_dir = "/teamspace/studios/01-data-download-kaggle/.cache/kagglehub/datasets/zeerafle/sitting-posture/versions/6"

        # Get all image files and their labels
        all_files = []
        all_labels = []
        all_subjects = []

        # Use the subject mapping from CNNTrainer
        subject_mapping = self.cnn_trainer.subject_mapping

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

        logger.info(f"CNN LOSO data prepared: {len(all_files)} images from {len(np.unique(all_subjects))} subjects")

        return all_files, all_labels, all_subjects

    def run(self):
        """Run the LOSO CNN training workflow"""
        from codecarbon import OfflineEmissionsTracker
        from dvclive import Live
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        import time
        import json
        from models.utils import NumpyEncoder

        logger.info("Starting CNN LOSO workflow with grouped 5-fold cross validation")

        # Prepare data
        file_paths, labels, subjects = self.prepare_data()

        # Verify data distribution
        unique_subjects = np.unique(subjects)
        subject_labels = pd.Series(labels, index=subjects).groupby(level=0).first()
        class_counts = subject_labels.value_counts()
        logger.info(f"Subject distribution by class: {dict(class_counts)}")

        # Setup DVCLive logging
        dvclive_path = os.path.join(self.dvclive_dir, "loso")
        os.makedirs(dvclive_path, exist_ok=True)

        # Setup grouped 5-fold cross validation
        cv_folds = self.params.get("cv_folds", 5)
        skf = StratifiedGroupKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=self.params.get("random_state", 42)
        )

        # Get subject-level splits
        try:
            subject_splits = list(skf.split(unique_subjects, subject_labels.values, unique_subjects))
            logger.info(f"Successfully created {len(subject_splits)} folds for CNN LOSO")
        except ValueError as e:
            logger.error(f"Error creating grouped folds: {e}")
            # Fallback to manual splitting
            from sklearn.model_selection import StratifiedKFold
            skf_simple = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            subject_splits = list(skf_simple.split(unique_subjects, subject_labels.values))

        # Store fold-wise metrics
        fold_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': [],
            'fold_info': []
        }

        with Live(dvclive_path) as live:
            # Run cross-validation folds
            for fold_idx, (train_subject_idx, test_subject_idx) in enumerate(subject_splits, 1):
                logger.info(f"Processing fold {fold_idx}/{cv_folds}")

                # Get subjects for this fold
                train_subjects = unique_subjects[train_subject_idx]
                test_subjects = unique_subjects[test_subject_idx]

                # Verify no subject leakage
                subject_intersection = set(train_subjects) & set(test_subjects)
                if subject_intersection:
                    logger.error(f"SUBJECT LEAKAGE DETECTED in fold {fold_idx}!")
                    raise ValueError(f"Subject leakage detected in fold {fold_idx}")

                logger.info(f"Fold {fold_idx}: {len(train_subjects)} train subjects, {len(test_subjects)} test subjects")

                # Get sample indices for this fold
                train_mask = np.isin(subjects, train_subjects)
                test_mask = np.isin(subjects, test_subjects)

                X_train_paths = file_paths[train_mask]
                y_train = labels[train_mask]
                X_test_paths = file_paths[test_mask]
                y_test = labels[test_mask]

                # Log fold information
                fold_info = {
                    'fold': fold_idx,
                    'train_subjects': train_subjects.tolist(),
                    'test_subjects': test_subjects.tolist(),
                    'train_samples': len(X_train_paths),
                    'test_samples': len(X_test_paths),
                    'train_class_distribution': dict(pd.Series(y_train).value_counts()),
                    'test_class_distribution': dict(pd.Series(y_test).value_counts())
                }
                fold_metrics['fold_info'].append(fold_info)

                logger.info(f"Fold {fold_idx} - Train: {len(X_train_paths)} samples, Test: {len(X_test_paths)} samples")

                # Create datasets
                train_dataset = self.create_dataset(X_train_paths, y_train)
                test_dataset = self.create_dataset(X_test_paths, y_test)

                # Train model for this fold
                model = self.get_estimator()

                # Track training time and emissions
                with OfflineEmissionsTracker(save_to_file=False) as train_tracker:
                    start_time = time.time()

                    # Train with fewer epochs for LOSO to save time
                    model.fit(train_dataset, epochs=5, verbose=0)

                    training_time = time.time() - start_time

                # Track inference time and emissions
                with OfflineEmissionsTracker(save_to_file=False) as inference_tracker:
                    start_time = time.time()

                    y_proba_fold = model.predict(test_dataset, verbose=0).flatten()
                    y_pred_fold = (y_proba_fold > 0.5).astype(int)

                    inference_time = time.time() - start_time

                # Calculate metrics for this fold
                fold_accuracy = accuracy_score(y_test, y_pred_fold)
                fold_precision = precision_score(y_test, y_pred_fold, average='binary')
                fold_recall = recall_score(y_test, y_pred_fold, average='binary')
                fold_f1 = f1_score(y_test, y_pred_fold, average='binary')
                fold_roc_auc = roc_auc_score(y_test, y_proba_fold)

                # Store metrics
                fold_metrics['accuracy'].append(fold_accuracy)
                fold_metrics['precision'].append(fold_precision)
                fold_metrics['recall'].append(fold_recall)
                fold_metrics['f1'].append(fold_f1)
                fold_metrics['roc_auc'].append(fold_roc_auc)

                # Log fold metrics to DVCLive
                live.log_metric(f"fold_{fold_idx}/accuracy", fold_accuracy)
                live.log_metric(f"fold_{fold_idx}/precision", fold_precision)
                live.log_metric(f"fold_{fold_idx}/recall", fold_recall)
                live.log_metric(f"fold_{fold_idx}/f1", fold_f1)
                live.log_metric(f"fold_{fold_idx}/roc_auc", fold_roc_auc)
                live.log_metric(f"fold_{fold_idx}/train_time", training_time)
                live.log_metric(f"fold_{fold_idx}/inference_time", inference_time)
                live.log_metric(f"fold_{fold_idx}/train_emissions", train_tracker.final_emissions_data.energy_consumed)
                live.log_metric(f"fold_{fold_idx}/inference_emissions", inference_tracker.final_emissions_data.energy_consumed)

                logger.info(f"Fold {fold_idx} results - Acc: {fold_accuracy:.4f}, F1: {fold_f1:.4f}, ROC-AUC: {fold_roc_auc:.4f}")

            # Calculate aggregated metrics
            aggregated_metrics = {}
            for metric_name, values in fold_metrics.items():
                if metric_name != 'fold_info' and values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    aggregated_metrics[f"{metric_name}_mean"] = mean_val
                    aggregated_metrics[f"{metric_name}_std"] = std_val

                    # Log to DVCLive
                    live.log_metric(f"loso_mean_{metric_name}", mean_val)
                    live.log_metric(f"loso_std_{metric_name}", std_val)

            logger.info(f"CNN LOSO Results - Mean Accuracy: {aggregated_metrics['accuracy_mean']:.4f} ± {aggregated_metrics['accuracy_std']:.4f}")
            logger.info(f"CNN LOSO Results - Mean F1: {aggregated_metrics['f1_mean']:.4f} ± {aggregated_metrics['f1_std']:.4f}")

            # Save detailed fold metrics
            fold_metrics_path = os.path.join(dvclive_path, "fold_metrics.json")
            with open(fold_metrics_path, "w") as f:
                json.dump(fold_metrics, f, indent=2, cls=NumpyEncoder)

            # Save aggregated metrics
            aggregated_metrics_path = os.path.join(dvclive_path, "aggregated_metrics.json")
            with open(aggregated_metrics_path, "w") as f:
                json.dump(aggregated_metrics, f, indent=2, cls=NumpyEncoder)

        # Train final model on all data
        logger.info("Training final CNN model on complete dataset")
        final_model = self.get_estimator()
        final_dataset = self.create_dataset(file_paths, labels)
        final_model.fit(final_dataset, epochs=5, verbose=1)

        # Save final model
        model_filename = f"{self.model_name}_loso.keras"
        model_path = os.path.join(self.models_dir, model_filename)
        final_model.save(model_path)
        logger.success(f"Saved final CNN model to {model_path}")

        logger.success("CNN LOSO workflow completed successfully")
        return final_model, aggregated_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN model with LOSO grouped 5-fold cross validation")
    args = parser.parse_args()

    trainer = CNNLOSOTrainer()
    model, metrics = trainer.run()
