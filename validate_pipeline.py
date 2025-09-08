#!/usr/bin/env python3
"""
Pipeline Validation Script

This script validates the integrity of the sitting posture classification pipeline
by checking for required files, dependencies, and data consistency.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from loguru import logger

def check_file_exists(file_path, description=""):
    """Check if a file exists and log the result."""
    if os.path.exists(file_path):
        logger.success(f"✓ Found: {file_path} {description}")
        return True
    else:
        logger.error(f"✗ Missing: {file_path} {description}")
        return False

def check_directory_exists(dir_path, description=""):
    """Check if a directory exists and log the result."""
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        logger.success(f"✓ Found directory: {dir_path} {description}")
        return True
    else:
        logger.error(f"✗ Missing directory: {dir_path} {description}")
        return False

def validate_core_files():
    """Validate core pipeline files exist."""
    logger.info("=== Validating Core Pipeline Files ===")

    core_files = [
        ("dvc.yaml", "DVC pipeline definition"),
        ("params.yaml", "Pipeline parameters"),
        ("src/extract_keypoints.py", "Keypoint extraction"),
        ("src/featurize.py", "Feature engineering"),
        ("src/prepare_standard.py", "Standard split preparation"),
        ("src/prepare_loso.py", "LOSO data preparation"),
        ("src/prepare_ablation.py", "Ablation data preparation"),
    ]

    all_exist = True
    for file_path, description in core_files:
        if not check_file_exists(file_path, description):
            all_exist = False

    return all_exist

def validate_model_training_files():
    """Validate model training files exist."""
    logger.info("=== Validating Model Training Files ===")

    models = ["adaboost", "nn", "xgb", "cnn"]
    training_types = ["train_standard.py", "train_loso.py"]

    all_exist = True
    for model in models:
        for training_type in training_types:
            file_path = f"src/models/{model}/{training_type}"
            description = f"{model.upper()} {training_type.replace('.py', '').replace('_', ' ')}"
            if not check_file_exists(file_path, description):
                all_exist = False

    # Check CNN trainer
    if not check_file_exists("src/models/cnn/trainer.py", "CNN trainer class"):
        all_exist = False

    return all_exist

def validate_analysis_files():
    """Validate analysis and workflow files exist."""
    logger.info("=== Validating Analysis Files ===")

    analysis_files = [
        ("src/analysis/statistical_analysis.py", "Statistical model comparison"),
        ("src/analysis/select_best_model.py", "Best model selection"),
        ("src/analysis/generate_report.py", "Report generation"),
        ("src/models/standard_workflow.py", "Standard training workflow"),
        ("src/models/loso_workflow.py", "LOSO training workflow"),
        ("src/models/train_ablation.py", "Ablation training"),
        ("src/models/base_trainer.py", "Base trainer class"),
        ("src/models/evaluate.py", "Model evaluation utilities"),
        ("src/models/utils.py", "Training utilities"),
    ]

    all_exist = True
    for file_path, description in analysis_files:
        if not check_file_exists(file_path, description):
            all_exist = False

    return all_exist

def validate_data_preparation_modules():
    """Validate data preparation modules exist."""
    logger.info("=== Validating Data Preparation Modules ===")

    data_prep_files = [
        ("src/data_preparation/imputer.py", "Data imputation"),
        ("src/data_preparation/scaler.py", "Feature scaling"),
        ("src/data_preparation/embedding.py", "Landmark embeddings"),
        ("src/data_preparation/data_filter.py", "Data filtering"),
        ("src/data_preparation/saver.py", "Data saving utilities"),
    ]

    all_exist = True
    for file_path, description in data_prep_files:
        if not check_file_exists(file_path, description):
            all_exist = False

    return all_exist

def validate_statistical_modules():
    """Validate statistical analysis modules exist."""
    logger.info("=== Validating Statistical Analysis Modules ===")

    if not check_directory_exists("src/analysis/statistical", "Statistical analysis directory"):
        return False

    stat_files = [
        ("src/analysis/statistical/data_loader.py", "Data loading for statistics"),
        ("src/analysis/statistical/tests.py", "Statistical tests"),
        ("src/analysis/statistical/output.py", "Results output"),
        ("src/analysis/statistical/plots.py", "Statistical plots"),
    ]

    all_exist = True
    for file_path, description in stat_files:
        if not check_file_exists(file_path, description):
            all_exist = False

    return all_exist

def validate_dvc_pipeline():
    """Validate DVC pipeline structure."""
    logger.info("=== Validating DVC Pipeline Structure ===")

    if not check_file_exists("dvc.yaml", "DVC pipeline file"):
        return False

    try:
        import yaml
        with open("dvc.yaml", 'r') as f:
            dvc_config = yaml.safe_load(f)

        expected_stages = [
            "extract_keypoints",
            "featurize",
            "prepare_standard_split",
            "prepare_loso",
            "train_standard_split",
            "train_loso",
            "statistical_analysis_loso",
            "select_best_model",
            "prepare_ablation",
            "train_ablation",
            "generate_analysis_report"
        ]

        stages = dvc_config.get("stages", {})
        missing_stages = []

        for stage in expected_stages:
            if stage not in stages:
                missing_stages.append(stage)

        if missing_stages:
            logger.error(f"✗ Missing DVC stages: {missing_stages}")
            return False
        else:
            logger.success(f"✓ All expected DVC stages found ({len(expected_stages)} stages)")
            return True

    except Exception as e:
        logger.error(f"✗ Error parsing DVC pipeline: {e}")
        return False

def validate_parameters():
    """Validate parameter file structure."""
    logger.info("=== Validating Parameters ===")

    if not check_file_exists("params.yaml", "Parameters file"):
        return False

    try:
        import yaml
        with open("params.yaml", 'r') as f:
            params = yaml.safe_load(f)

        required_params = [
            "random_state",
            "n_iter",
            "cv",
            "cv_folds",
            "scoring",
            "adaboost",
            "nn",
            "xgb"
        ]

        missing_params = []
        for param in required_params:
            if param not in params:
                missing_params.append(param)

        if missing_params:
            logger.error(f"✗ Missing parameters: {missing_params}")
            return False
        else:
            logger.success(f"✓ All required parameters found")
            return True

    except Exception as e:
        logger.error(f"✗ Error parsing parameters: {e}")
        return False

def validate_import_structure():
    """Validate that key modules can be imported."""
    logger.info("=== Validating Import Structure ===")

    import_tests = [
        ("src.data_preparation.data_filter", "Data filtering module"),
        ("src.data_preparation.imputer", "Data imputation module"),
        ("src.models.base_trainer", "Base trainer class"),
        ("src.models.utils", "Training utilities"),
    ]

    # Add src to path for imports
    sys.path.insert(0, "src")

    all_imports_ok = True
    for module_path, description in import_tests:
        try:
            __import__(module_path)
            logger.success(f"✓ Successfully imported: {module_path} ({description})")
        except ImportError as e:
            logger.error(f"✗ Failed to import: {module_path} ({description}) - {e}")
            all_imports_ok = False
        except Exception as e:
            logger.warning(f"⚠ Import warning for {module_path}: {e}")

    return all_imports_ok

def validate_data_consistency():
    """Validate data file consistency."""
    logger.info("=== Validating Data Consistency ===")

    # Check if base data exists
    base_data_files = [
        "data/data.csv",
        "data/data_with_features.csv"
    ]

    data_exists = True
    for file_path in base_data_files:
        if not check_file_exists(file_path, "Base data file"):
            data_exists = False

    if not data_exists:
        logger.warning("⚠ Base data files not found - this is expected if data preparation hasn't been run")
        return True  # Not a failure, just not prepared yet

    # If data exists, validate structure
    try:
        df = pd.read_csv("data/data.csv")

        required_columns = ["subject_id", "class_no", "file_name"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.error(f"✗ Missing required columns in data.csv: {missing_columns}")
            return False

        logger.success(f"✓ Data structure validated: {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"  - Unique subjects: {df['subject_id'].nunique()}")
        logger.info(f"  - Class distribution: {dict(df['class_no'].value_counts())}")

        return True

    except Exception as e:
        logger.error(f"✗ Error validating data structure: {e}")
        return False

def validate_directory_structure():
    """Validate expected directory structure."""
    logger.info("=== Validating Directory Structure ===")

    expected_dirs = [
        ("src", "Source code"),
        ("src/models", "Model implementations"),
        ("src/analysis", "Analysis scripts"),
        ("src/data_preparation", "Data preparation modules"),
        ("data", "Data directory"),
        ("models", "Trained models directory"),
        ("dvclive", "DVCLive metrics directory"),
    ]

    all_exist = True
    for dir_path, description in expected_dirs:
        if not check_directory_exists(dir_path, description):
            all_exist = False

    return all_exist

def check_removed_files():
    """Check that cleanup was successful by verifying removed files don't exist."""
    logger.info("=== Validating Cleanup (Removed Files) ===")

    removed_files = [
        "src/prepare.py",
        "src/model_comparison.py",
        "src/roc_visualization.py",
        "src/run_ablation.py",
        "src/data.py",
        "src/models/training_workflow.py",
        "src/models/xgb/train.py",
        "src/models/nn/train.py",
        "src/models/adaboost/train.py",
    ]

    cleanup_successful = True
    for file_path in removed_files:
        if os.path.exists(file_path):
            logger.warning(f"⚠ Old file still exists: {file_path} (should have been removed)")
            cleanup_successful = False
        else:
            logger.success(f"✓ Confirmed removed: {file_path}")

    return cleanup_successful

def main():
    """Run all validation checks."""
    logger.info("🔍 Starting Pipeline Validation")
    logger.info("=" * 60)

    checks = [
        ("Core Files", validate_core_files),
        ("Model Training Files", validate_model_training_files),
        ("Analysis Files", validate_analysis_files),
        ("Data Preparation Modules", validate_data_preparation_modules),
        ("Statistical Modules", validate_statistical_modules),
        ("DVC Pipeline", validate_dvc_pipeline),
        ("Parameters", validate_parameters),
        ("Directory Structure", validate_directory_structure),
        ("Import Structure", validate_import_structure),
        ("Data Consistency", validate_data_consistency),
        ("Cleanup Verification", check_removed_files),
    ]

    results = {}
    for check_name, check_func in checks:
        logger.info("")
        try:
            results[check_name] = check_func()
        except Exception as e:
            logger.error(f"✗ Error during {check_name}: {e}")
            results[check_name] = False

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("🏁 VALIDATION SUMMARY")
    logger.info("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for check_name, passed_check in results.items():
        status = "✓ PASS" if passed_check else "✗ FAIL"
        logger.info(f"{status:<8} {check_name}")

    logger.info("")
    if passed == total:
        logger.success(f"🎉 ALL CHECKS PASSED ({passed}/{total})")
        logger.info("Pipeline is ready for execution!")
        return 0
    else:
        logger.error(f"❌ {total - passed} CHECKS FAILED ({passed}/{total} passed)")
        logger.info("Please address the issues above before running the pipeline.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
