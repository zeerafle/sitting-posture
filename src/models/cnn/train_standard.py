import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
from loguru import logger
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, parent_dir)

from models.base_trainer import BaseTrainer
from models.cnn.trainer import CNNTrainer


class CNNStandardTrainer(BaseTrainer):
    def __init__(self, use_loso=False):
        super().__init__(
            model_name="cnn",
            train_combined=True,  # Always use combined data
            use_loso=use_loso        # This is for standard split
        )
        self.cnn_trainer = CNNTrainer(use_loso=use_loso)
        self.data_dir = "/teamspace/studios/01-data-download-kaggle/.cache/kagglehub/datasets/zeerafle/sitting-posture/versions/6"

    def get_estimator(self):
        return self.cnn_trainer.get_estimator()

    def get_param_space(self):
        # CNN uses fixed architecture for standard training
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
        """Prepare CNN data from processed standard split CSV files"""
        # Load data from standard split CSV files
        train_csv_path = "data/processed/standard_split/train.csv"
        test_csv_path = "data/processed/standard_split/test.csv"

        if not os.path.exists(train_csv_path):
            raise FileNotFoundError(f"Train CSV not found: {train_csv_path}")
        if not os.path.exists(test_csv_path):
            raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")

        train_df = pd.read_csv(train_csv_path)
        test_df = pd.read_csv(test_csv_path)
        logger.info(f"Loaded {len(train_df)} training and {len(test_df)} testing records from standard split CSVs")

        # Prepare training and testing datasets
        train_files = []
        train_labels = []
        test_files = []
        test_labels = []

        # Process training data
        for _, row in train_df.iterrows():
            file_name = row['file_name']
            subject_id = row['subject_id']
            class_label = row['labels']

            # Construct the full image path
            image_path = os.path.join(self.data_dir, file_name)

            train_files.append(image_path)
            train_labels.append(class_label)

        # Process testing data
        for _, row in test_df.iterrows():
            file_name = row['file_name']
            subject_id = row['subject_id']
            class_label = row['labels']

            # Construct the full image path
            image_path = os.path.join(self.data_dir, file_name)

            test_files.append(image_path)
            test_labels.append(class_label)

        logger.info(f"CNN data prepared: {len(train_files)} train, {len(test_files)} test images")

        # Convert to numpy arrays
        train_files = np.array(train_files)
        train_labels = np.array(train_labels)
        test_files = np.array(test_files)
        test_labels = np.array(test_labels)

        return train_files, train_labels, test_files, test_labels

    def run(self):
        """Run the standard CNN training workflow"""
        from codecarbon import OfflineEmissionsTracker
        from dvclive import Live

        logger.info("Starting CNN standard workflow")

        # Print TensorFlow version for debugging
        logger.info(f"TensorFlow version: {tf.__version__}")
        logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

        # Prepare data
        train_files, train_labels, test_files, test_labels = self.prepare_data()

        # Create tf.data datasets for efficient loading
        batch_size = 32
        train_dataset = self.create_dataset(train_files, train_labels, batch_size, is_training=True)
        test_dataset = self.create_dataset(test_files, test_labels, batch_size, is_training=False)

        # Extract a small batch for test predictions later
        try:
            test_samples = next(iter(test_dataset.take(1)))
            test_images, test_labels_batch = test_samples
        except StopIteration:
            logger.warning("Could not extract test samples, dataset may be empty")
            # Create dummy tensors
            test_images = tf.zeros((1, 224, 224, 3))
            test_labels_batch = tf.zeros((1), dtype=tf.int32)

        # Get dataset sizes
        num_train_samples = len(train_files)
        num_test_samples = len(test_files)
        steps_per_epoch = int(np.ceil(num_train_samples / batch_size))
        validation_steps = int(np.ceil(num_test_samples / batch_size))

        logger.info(f"Created training dataset with {num_train_samples} samples ({steps_per_epoch} steps)")
        logger.info(f"Created testing dataset with {num_test_samples} samples ({validation_steps} steps)")

        # Setup DVCLive logging
        dvclive_path = os.path.join(self.dvclive_dir, "standard")
        os.makedirs(dvclive_path, exist_ok=True)

        with Live(dvclive_path) as live:
            # Get model (no hyperparameter tuning for CNN in this version)
            model = self.get_estimator()

            # Track training emissions
            with OfflineEmissionsTracker(save_to_file=False) as train_tracker:
                start_time = time.time()

                # Train model with tf.data dataset
                history = model.fit(
                    train_dataset,
                    validation_data=test_dataset,
                    epochs=10,
                    verbose=1,
                )

                training_time = time.time() - start_time

            live.log_metric("train/time_seconds", training_time)
            live.log_metric("train/emissions_kwh", train_tracker.final_emissions_data.energy_consumed)

            # Track inference emissions
            with OfflineEmissionsTracker(save_to_file=False) as inference_tracker:
                start_time = time.time()

                # Get predictions
                try:
                    logger.info("Starting prediction...")

                    # Check a sample before prediction using the saved test batch
                    logger.info(f"Test batch shape: {test_images.shape}, Test labels shape: {test_labels_batch.shape}")
                    sample_pred = model.predict(test_images[0:1], verbose=1)
                    logger.info(f"Sample prediction shape: {sample_pred.shape}, value: {sample_pred}")

                    # Collect all test predictions by batches
                    all_y_true = []
                    all_y_proba = []

                    # Use model.predict on the full test dataset
                    logger.info("Running predictions on all test data...")
                    y_proba_batched = model.predict(test_dataset, verbose=1)

                    # Collect all labels from test dataset
                    for _, labels in test_dataset:
                        all_y_true.extend(labels.numpy())

                    # Flatten predictions if needed
                    if len(y_proba_batched.shape) > 1:
                        y_proba = y_proba_batched.flatten()
                    else:
                        y_proba = y_proba_batched

                    # Get labels as numpy array
                    y_test = np.array(all_y_true)

                    # Convert probabilities to binary predictions
                    y_pred = (y_proba > 0.5).astype(int)

                    logger.info(f"Collected {len(y_test)} test samples and predictions")

                    # Log some debug info about predictions
                    logger.info(f"Made {len(y_proba)} predictions. Distribution: " +
                            f"{np.sum(y_pred == 0)} ergonomic, {np.sum(y_pred == 1)} non-ergonomic")
                except Exception as e:
                    logger.error(f"Error during prediction: {str(e)}")
                    raise

                inference_time = time.time() - start_time

            live.log_metric("test/time_seconds", inference_time)
            live.log_metric("test/emissions_kwh", inference_tracker.final_emissions_data.energy_consumed)

            # Calculate metrics
            test_accuracy = accuracy_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred, average='binary')
            test_recall = recall_score(y_test, y_pred, average='binary')
            test_f1 = f1_score(y_test, y_pred, average='binary')
            test_roc_auc = roc_auc_score(y_test, y_proba)

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
            # Ensure all arrays have the same length
            min_length = min(len(y_test), len(y_pred), len(y_proba))
            predictions_df = pd.DataFrame({
                'y_true': y_test[:min_length],
                'y_pred': y_pred[:min_length],
                'y_proba': y_proba[:min_length]
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
    # Configure logger to show more details
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    logger.info("Starting CNN standard trainer script")

    parser = argparse.ArgumentParser(description="Train CNN model with standard train-test split")
    args = parser.parse_args()

    try:
        trainer = CNNStandardTrainer()
        model, metrics = trainer.run()
    except Exception as e:
        logger.exception("Fatal error during execution")
        sys.exit(1)
