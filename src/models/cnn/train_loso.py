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
        self.data_dir = "/teamspace/studios/01-data-download-kaggle/.cache/kagglehub/datasets/zeerafle/sitting-posture/versions/6"

    def get_estimator(self):
        return self.cnn_trainer.get_estimator()

    def get_param_space(self):
        # CNN uses fixed architecture for LOSO
        return {}

    def preprocess_image(self, image_path, label):
        """Load and preprocess a single image for tf.data pipeline"""
        # Define a wrapper function that handles exceptions
        def _process_image_path_tensor(path_tensor):
            path_str = path_tensor.numpy().decode('utf-8')
            try:
                # Read the image file
                img = tf.io.read_file(path_tensor)
                # Decode the image
                img = tf.image.decode_image(img, channels=3, expand_animations=False)
                # Resize the image
                img = tf.image.resize(img, (224, 224))
                # Apply MobileNetV2 preprocessing
                img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
                return img
            except Exception as e:
                logger.error(f"Error processing image {path_str}: {str(e)}")
                # Return a placeholder image on error
                return tf.zeros((224, 224, 3), dtype=tf.float32)

        # Use tf.py_function to wrap the Python function
        img = tf.py_function(
            _process_image_path_tensor,
            [image_path],
            tf.float32
        )
        # Ensure the image has the right shape
        img.set_shape((224, 224, 3))
        # Cast label to int32
        label = tf.cast(label, tf.int32)
        return img, label

    def filter_valid_paths(self, path, label):
        """Filter function to check if file exists"""
        # Use py_function to wrap the file existence check
        file_exists = tf.py_function(
            lambda p: tf.constant(os.path.exists(p.numpy().decode('utf-8'))),
            [path],
            tf.bool
        )
        return file_exists

    def create_dataset(self, file_paths, labels, batch_size=32, is_training=True):
        """Create tf.data.Dataset for efficient data loading"""
        logger.debug(f"Creating dataset with {len(file_paths)} images")

        # Verify and filter file paths
        valid_count = 0
        for path in file_paths:
            if tf.io.gfile.exists(path):
                valid_count += 1

        logger.info(f"Found {valid_count} valid images out of {len(file_paths)} total paths")

        if valid_count == 0:
            raise ValueError(f"No valid image paths found! Check your data directory: {self.data_dir}")

        # Create dataset from tensors
        paths_tensor = tf.convert_to_tensor(file_paths, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(labels, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor))

        # Filter valid files
        dataset = dataset.filter(lambda path, label: self.filter_valid_paths(path, label))

        # Map preprocessing function to each element
        dataset = dataset.map(
            self.preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Apply training-specific transformations
        if is_training:
            # Cache to prevent re-execution of map for each epoch
            dataset = dataset.cache()
            # Shuffle with a large buffer
            dataset = dataset.shuffle(buffer_size=min(10000, len(file_paths)))
            # Apply data augmentation here if needed
            # dataset = dataset.map(data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)

        # Batch and prefetch
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def prepare_data(self):
        """Prepare CNN data from processed LOSO CSV file"""
        data_path = "data/processed/loso/data.csv"

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"LOSO CSV not found: {data_path}")

        # Load the LOSO dataset
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} records from LOSO CSV")

        # Check if file_name column exists
        if 'file_name' not in df.columns:
            raise ValueError("Required 'file_name' column not found in LOSO CSV")

        # Check if subject_id column exists
        if 'subject_id' not in df.columns:
            raise ValueError("Required 'subject_id' column not found in LOSO CSV")

        # Check if labels column exists
        if 'labels' not in df.columns:
            raise ValueError("Required 'labels' column not found in LOSO CSV")

        # Construct full file paths
        all_files = []
        all_labels = []
        all_subjects = []

        # Process the CSV data
        for _, row in df.iterrows():
            file_name = row['file_name']
            subject_id = row['subject_id']
            class_label = row['labels']

            # Construct the full image path
            image_path = os.path.join(self.data_dir, file_name)

            all_files.append(image_path)
            all_labels.append(class_label)
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
            raise ValueError(f"Error creating grouped folds: {e}")
            # Fallback to manual splitting
            # from sklearn.model_selection import StratifiedKFold
            # skf_simple = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            # subject_splits = list(skf_simple.split(unique_subjects, subject_labels.values))

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
                train_dataset = self.create_dataset(X_train_paths, y_train, is_training=True)
                test_dataset = self.create_dataset(X_test_paths, y_test, is_training=False)

                # Train model for this fold
                model = self.get_estimator()

                # Track training time and emissions
                with OfflineEmissionsTracker(save_to_file=False) as train_tracker:
                    start_time = time.time()

                    # PHASE 1: Initial training with frozen base model
                    logger.info(f"Fold {fold_idx} - Phase 1: Initial training with frozen base")
                    model.fit(train_dataset, epochs=1, verbose=1)

                    # PHASE 2: Fine-tuning with last block unfrozen
                    logger.info(f"Fold {fold_idx} - Phase 2: Fine-tuning with last block unfrozen")

                    # Unfreeze the last block and recompile with smaller learning rate
                    model = self.cnn_trainer.unfreeze_last_block(model)

                    # Train for 10-15 more epochs with early stopping
                    model.fit(
                        train_dataset,
                        epochs=1,  # Maximum number of epochs for fine-tuning
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(
                                monitor='loss',
                                patience=1,
                                restore_best_weights=True
                            )
                        ],
                        verbose=0
                    )
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

        logger.success("CNN LOSO workflow completed successfully")
        return aggregated_metrics


if __name__ == "__main__":
    # Configure logger to show more details
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    logger.info("Starting CNN LOSO trainer script")

    parser = argparse.ArgumentParser(description="Train CNN model with LOSO grouped 5-fold cross validation")
    args = parser.parse_args()

    try:
        trainer = CNNLOSOTrainer()
        metrics = trainer.run()
    except Exception as e:
        logger.exception("Fatal error during execution")
        sys.exit(1)
