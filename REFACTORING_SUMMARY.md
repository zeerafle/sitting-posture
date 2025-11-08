# Training Scripts Refactoring Summary

## Overview
Refactored the training scripts to perform cross-validation on all data (combining train/test splits) and made the codebase more modular and efficient by eliminating code duplication.

## Changes Made

### 1. Created Base Training Module (`src/models/base_trainer.py`)
A new modular training framework with:

- **`BaseTrainer`**: Base class with common functionality for all models
  - `train()`: Trains model with emission tracking
  - `predict()`: Makes predictions with emission tracking
  - `save_model()`: Saves models (supports .joblib and .json formats)
  - `log_model_specific_metrics()`: Hook for model-specific metrics (override in subclasses)
  - `log_predictions()`: Saves predictions to CSV
  - `log_cv_results()`: Logs cross-validation results
  - `log_emissions()`: Logs emission artifacts
  - `run_training_pipeline()`: Complete training pipeline orchestrator

- **Model-Specific Trainers**:
  - `AdaBoostTrainer`: Logs estimator weights and feature importances
  - `XGBTrainer`: Logs feature importances
  - `NNTrainer`: Logs best loss and loss curve

### 2. Updated Utility Functions (`src/models/utils.py`)
Added new function:
- **`load_all_data(base_path)`**: Loads and combines train/test data for cross-validation
- Kept **`load_data(base_path)`** for backward compatibility

### 3. Updated Evaluation Module (`src/models/evaluate.py`)
Added new function:
- **`evaluate_with_cv(model, X, y, live, cv=10)`**: Performs only cross-validation on all data
  - Uses `cross_validate()` for scoring metrics
  - Uses `cross_val_predict()` to generate predictions for confusion matrix and ROC curve
  - Logs CV mean and std for: accuracy, f1, precision, recall, roc_auc
  - Generates confusion matrix and ROC curve plots
- Kept **`evaluate()`** for backward compatibility

### 4. Refactored Training Scripts
All three training scripts now follow the same clean pattern:

#### Before (AdaBoost example - ~87 lines):
```python
# Manual data loading
X_train, X_test, y_train, y_test = load_data(...)

# Manual training with emission tracking
with OfflineEmissionsTracker(...) as tracker:
    model = adaboost.fit(X_train, np.ravel(y_train))

# Manual metric logging
live.log_metric("estimator_weights_mean", ...)
live.log_metric("feature_importance_mean", ...)

# Manual model saving
os.makedirs(models_dir, exist_ok=True)
with open(model_path, "wb") as f:
    joblib.dump(model, f)

# Manual prediction with emission tracking
with OfflineEmissionsTracker(...) as tracker:
    y_pred = model.predict(X_test)

# Manual prediction saving
np.savetxt(...)

# Manual evaluation
cv_scores = evaluate(...)

# Manual CV results logging
with open(cv_results_json_path, "w") as f:
    json.dump(cv_scores, f, ...)
```

#### After (all models - ~45 lines):
```python
# Load all data for CV
X, y = load_all_data(...)

# Initialize model
model = ModelClass(random_state=params["random_state"])

# Initialize trainer
trainer = ModelTrainer(
    model=model,
    model_name="model_name",
    dvclive_path=dvclive_path,
    models_dir=models_dir
)

# Run complete pipeline
trainer.run_training_pipeline(
    X=X,
    y=y,
    evaluate_fn=evaluate_with_cv,
    live=live,
    model_path=model_path
)
```

### Files Modified:
- ✅ `src/models/base_trainer.py` (NEW)
- ✅ `src/models/utils.py` (updated)
- ✅ `src/models/evaluate.py` (updated)
- ✅ `src/models/adaboost/train.py` (simplified from ~87 to ~45 lines)
- ✅ `src/models/nn/train.py` (simplified from ~83 to ~45 lines)
- ✅ `src/models/xgb/train.py` (simplified from ~77 to ~45 lines)

## Benefits

1. **Cross-Validation on All Data**: Now trains on all available data and uses cross-validation for evaluation instead of a fixed train/test split
2. **Eliminated Code Duplication**: Reduced ~240 lines of repetitive code to a single reusable base class
3. **Easier Maintenance**: Common functionality in one place - fix once, applies everywhere
4. **Consistent Behavior**: All models follow the same training pipeline
5. **Extensibility**: Easy to add new models by extending `BaseTrainer`
6. **Cleaner Training Scripts**: Each model's training script is now ~45 lines vs ~80+ lines
7. **Better Separation of Concerns**: Training logic separated from model-specific details

## Usage

To train any model:
```bash
python src/models/adaboost/train.py
python src/models/nn/train.py
python src/models/xgb/train.py
```

The scripts now:
1. Load all available data (train + test combined)
2. Train the model on all data with emission tracking
3. Perform 10-fold cross-validation for evaluation
4. Log all metrics, plots, and artifacts
5. Save the trained model

## Backward Compatibility

The old functions are preserved:
- `load_data()` still available for scripts that need separate train/test splits
- `evaluate()` still available for legacy evaluation with fixed test sets
