import os
import sys
import json
import yaml
import tensorflow as tf
import numpy as np
import pandas as pd
from loguru import logger
from dvclive import Live
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, parent_dir)

from models.cnn.trainer import CNNTrainer
from models.utils import NumpyEncoder

def load_image(file_name):
    """Load and preprocess image"""
    raw = tf.io.read_file(file_name)
    tensor = tf.io.decode_image(raw)
    tensor = tf.cast(tensor, tf.float32) / 255.0
    tensor = tf.image.resize_with_pad(tensor, 224, 224)
    return tf.keras.applications.mobilenet_v2.preprocess_input(tensor)

def create_dataset(file_names, labels, batch_size=32):
    """Create a TensorFlow dataset from file paths"""
    dataset = tf.data.Dataset.from_tensor_slices((file_names, labels))
    dataset = dataset.map(lambda file_name, label: (load_image(file_name), label))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def run_loso_training():
    """Run Leave-One-Subject-Out training for CNN model"""
    trainer = CNNTrainer(model_name="cnn", train_combined=True, use_loso=True)

    # Load data paths and labels
    ergonomic_list = []
    non_ergonomic_list = []
    for root, dirs, files in tf.io.gfile.walk(trainer.data_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                if "non-ergonomis" in root:
                    non_ergonomic_list.append(f"{root}/{file}")
                elif "ergonomis" in root:
                    ergonomic_list.append(f"{root}/{file}")

    # Create labels
    ergonomic_labels = [0] * len(ergonomic_list)
    non_ergonomic_labels = [1] * len(non_ergonomic_list)

    # Combine all data
    all_files = ergonomic_list + non_ergonomic_list
    all_labels = ergonomic_labels + non_ergonomic_labels

    # Get subject IDs for each file
    subjects = []
    for file_path in all_files:
        subject_id = trainer.subject_mapping.get(file_path, -1)
        subjects.append(subject_id)

    # If any subject is -1, we couldn't find mapping
    if -1 in subjects:
        logger.warning(f"Could not find subject ID for {subjects.count(-1)} images")

    # Convert to numpy arrays
    file_paths = np.array(all_files)
    labels = np.array(all_labels)
    subjects = np.array(subjects)

    # Setup DVC Live
    dvclive_path = os.path.join(trainer.dvclive_dir, "combined_loso")
    os.makedirs(dvclive_path, exist_ok=True)

    # Load best params if they exist (same pattern as other LOSO models)
    params_path = os.path.join(trainer.dvclive_dir, "combined", "params.yaml")
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            best_params = yaml.safe_load(f)
        best_params.pop("htcv_best_score", None)
    else:
        best_params = {}
        logger.warning(f"No best-params at {params_path}, using defaults.")

    # Setup StratifiedGroupKFold (same as other LOSO models)
    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=trainer.params.get("random_state", 42)
    )
    splits = list(cv.split(file_paths, labels, subjects))

    metrics = {m: [] for m in ("accuracy", "precision", "recall", "f1", "roc_auc")}
    with Live(dvclive_path) as live:
        if best_params:
            live.log_params(best_params)

        for fold, (tr_idx, te_idx) in enumerate(splits, start=1):
            # Get subject ID for logging (first test subject)
            test_subject = subjects[te_idx[0]]
            logger.info(f"Fold {fold}/5 - Testing on subject {test_subject}")

            # Split data by indices from StratifiedGroupKFold
            X_train_paths = file_paths[tr_idx]
            y_train = labels[tr_idx]
            X_test_paths = file_paths[te_idx]
            y_test = labels[te_idx]

            logger.info(f"Training set: {len(X_train_paths)} samples")
            logger.info(f"Test set: {len(X_test_paths)} samples")

            # Create datasets
            train_ds = create_dataset(X_train_paths, y_train)
            test_ds = create_dataset(X_test_paths, y_test)

            # Get model
            model = trainer.get_estimator()

            # Train model
            model.fit(train_ds, epochs=5, verbose=1)

            # Evaluate on test set
            y_pred_raw = model.predict(test_ds, verbose=0)
            y_pred = (y_pred_raw > 0.5).astype(int).flatten()
            y_pred_proba = y_pred_raw.flatten()

            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_pred_proba)

            # Log metrics for this subject
            metrics["accuracy"].append(acc)
            metrics["precision"].append(prec)
            metrics["recall"].append(rec)
            metrics["f1"].append(f1)
            metrics["roc_auc"].append(roc)

            logger.info(
                f"Subject {test_subject} - acc={acc:.3f}, f1={f1:.3f}, roc_auc={roc:.3f}"
            )

        # Log aggregate metrics
        for k, vals in metrics.items():
            m, s = np.mean(vals), np.std(vals)
            live.log_metric(f"loso_mean_{k}", m, plot=False)
            live.log_metric(f"loso_std_{k}", s, plot=False)

        # Save fold-wise metrics in same format as other models
        json_path = os.path.join(dvclive_path, "loso_metrics.json")
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2, cls=NumpyEncoder)
        logger.success(f"Saved LOSO metrics to {json_path}")

        # Train final model on all data
        logger.info("Training final model on all data")
        final_ds = create_dataset(file_paths, labels)
        final_model = trainer.get_estimator()
        final_model.fit(final_ds, epochs=5, verbose=1)

        # Save final model with the standard naming convention for LOSO models
        model_path = os.path.join(trainer.models_dir, f"{trainer.model_name}_combined_loso.keras")
        final_model.save(model_path)
        logger.success(f"Final model saved to {model_path}")

if __name__ == "__main__":
    run_loso_training()
