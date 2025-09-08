# Sitting Posture Classification - New Pipeline Documentation

## Overview

This document describes the completely rewritten pipeline that follows the new experimental procedures. The pipeline has been redesigned to use combined data only (ignoring view types) and implements a comprehensive evaluation methodology with statistical analysis and ablation studies.

## Pipeline Architecture

### Key Changes from Original Pipeline

1. **View Type Simplification**: Removed front/left/right view distinctions - all analyses use combined data
2. **Dual Evaluation Strategy**: Both standard train-test split AND LOSO evaluation
3. **Grouped Cross-Validation**: Uses StratifiedGroupKFold to prevent subject-level leakage
4. **Statistical Analysis Integration**: Automated statistical comparison with Friedman tests
5. **Best Model Selection**: Data-driven selection based on statistical significance
6. **Comprehensive Ablation Studies**: Systematic evaluation of data modes and feature sets

## Pipeline Stages

### 1. Data Preparation (Unchanged)
- `extract_keypoints`: Extract pose keypoints from images
- `featurize`: Generate engineered features from keypoints

### 2. Data Processing (New Structure)

#### Standard Split Preparation
```bash
dvc repro prepare_standard_split
```
- Creates stratified train-test split (80/20) by subject
- Prevents subject leakage between train/test sets
- Uses all data (real + synthetic) and all features
- Output: `data/processed/standard_split/`

#### LOSO Preparation
```bash
dvc repro prepare_loso
```
- Prepares data for grouped 5-fold cross validation
- Maintains subject metadata for proper grouping
- Output: `data/processed/loso/`

### 3. Model Training

#### Standard Training (with Hyperparameter Tuning)
```bash
dvc repro train_standard_split
```
- Trains all models (XGBoost, Neural Network, AdaBoost)
- Includes comprehensive hyperparameter optimization
- Uses stratified cross-validation for hyperparameter selection
- Outputs: 
  - `models/{model}/{model}_standard.joblib`
  - `dvclive/{model}/standard/`

#### LOSO Training (Grouped 5-Fold CV)
```bash
dvc repro train_loso
```
- Trains all models using grouped 5-fold cross validation
- Each fold holds out disjoint participants
- Prevents subject-level data leakage
- Uses best hyperparameters from standard training
- Outputs:
  - `models/{model}/{model}_loso.joblib`
  - `dvclive/{model}/loso/`

### 4. Statistical Analysis

#### LOSO Model Comparison
```bash
dvc repro statistical_analysis_loso
```
- Performs Friedman test for model comparison
- Calculates Kendall's W for effect size
- Includes pairwise comparisons and post-hoc tests
- Generates critical difference diagrams
- Outputs: `dvclive/analysis/loso/`

#### Best Model Selection
```bash
dvc repro select_best_model
```
- Automatically selects best model based on statistical analysis
- Uses configurable criteria (statistical significance, average rank, mean score)
- Outputs: `dvclive/analysis/best_model.json`

### 5. Ablation Studies

#### Ablation Data Preparation
```bash
dvc repro prepare_ablation
```
- Creates datasets for all ablation configurations:
  - Data modes: `real`, `synthetic`, `all`
  - Feature modes: `keypoints_only`, `all_features`
- Outputs: `data/processed/ablation/{data_mode}_{feature_mode}/`

#### Ablation Training
```bash
dvc repro train_ablation
```
- Trains best model on all ablation configurations
- Uses grouped 5-fold cross validation for fair comparison
- Outputs: `dvclive/ablation/{data_mode}_{feature_mode}/`

#### Ablation Statistical Analysis
```bash
dvc repro statistical_analysis_ablation_*
```
- Three types of ablation analysis:
  - `data_modes`: Compare real vs synthetic vs all data
  - `feature_modes`: Compare keypoints-only vs all features
  - `full`: Complete comparison of all configurations

### 6. Final Reporting

#### Comprehensive Report Generation
```bash
dvc repro generate_analysis_report
```
- Generates HTML report with all results
- Includes statistical analysis summaries
- Creates visualizations and performance plots
- Outputs: `dvclive/analysis/final_report/`

## Directory Structure

```
data/
├── processed/
│   ├── standard_split/          # Standard train-test split data
│   ├── loso/                    # LOSO prepared data
│   └── ablation/                # Ablation experiment data
│       ├── real_keypoints_only/
│       ├── real_all_features/
│       ├── synthetic_keypoints_only/
│       ├── synthetic_all_features/
│       ├── all_keypoints_only/
│       └── all_all_features/

models/
├── adaboost/
├── nn/
├── xgb/
└── ablation/                    # Best model variants for ablation

dvclive/
├── adaboost/
│   ├── standard/                # Standard evaluation results
│   └── loso/                    # LOSO evaluation results
├── nn/
├── xgb/
├── analysis/
│   ├── loso/                    # LOSO statistical analysis
│   ├── ablation/                # Ablation statistical analysis
│   ├── best_model.json         # Best model selection
│   └── final_report/            # Comprehensive report
└── ablation/                    # Ablation experiment results
```

## Key Features

### 1. Subject-Level Data Integrity
- **Grouped Cross-Validation**: Uses `StratifiedGroupKFold` with subject IDs as groups
- **Leakage Prevention**: Ensures no subject appears in both training and test sets
- **Stratification**: Maintains class balance across folds

### 2. Statistical Rigor
- **Friedman Test**: Non-parametric test for comparing multiple models
- **Post-hoc Analysis**: Nemenyi test for pairwise comparisons when significant
- **Effect Size**: Kendall's W for measuring agreement between rankings
- **Multiple Comparison Correction**: Appropriate corrections for pairwise tests

### 3. Automated Model Selection
- **Evidence-Based Selection**: Uses statistical significance and performance rankings
- **Configurable Criteria**: Can prioritize different selection strategies
- **Transparent Reasoning**: Documents selection rationale in output

### 4. Comprehensive Ablation Studies
- **Systematic Design**: Tests all combinations of data modes and feature sets
- **Fair Comparison**: Uses same best model architecture for all configurations
- **Statistical Validation**: Applies same rigorous testing to ablation results

## Usage Examples

### Running Complete Pipeline
```bash
# Run everything in sequence
dvc repro

# Or run specific stages
dvc repro prepare_standard_split
dvc repro train_standard_split
dvc repro train_loso
dvc repro statistical_analysis_loso
dvc repro select_best_model
dvc repro prepare_ablation
dvc repro train_ablation
dvc repro statistical_analysis_ablation_full
dvc repro generate_analysis_report
```

### Manual Model Training
```bash
# Train specific model with standard split
python src/models/xgb/train_standard.py

# Train specific model with LOSO
python src/models/xgb/train_loso.py

# Run ablation for specific configuration
python src/models/train_ablation.py \
  --data-mode all \
  --feature-mode all_features \
  --best-model-config dvclive/analysis/best_model.json
```

### Custom Statistical Analysis
```bash
# Analyze LOSO results
python src/analysis/statistical_analysis.py \
  --models adaboost nn xgb \
  --metric accuracy \
  --analysis-type loso \
  --output-dir dvclive/analysis/custom

# Analyze ablation results
python src/analysis/statistical_analysis.py \
  --models real_keypoints_only synthetic_keypoints_only all_keypoints_only \
  --metric f1 \
  --analysis-type ablation \
  --output-dir dvclive/analysis/ablation_custom
```

## Configuration

### Parameters (params.yaml)
```yaml
# Global parameters
random_state: 42      # For reproducibility
n_iter: 30           # Hyperparameter optimization iterations
cv: 5                # Cross-validation folds for hyperparameter tuning
cv_folds: 5          # Folds for LOSO grouped CV
scoring: accuracy    # Primary optimization metric

# Model-specific parameters
adaboost:
  n_estimators_min: 10
  n_estimators_max: 2000

nn:
  first_hidden_layer_sizes_min: 128
  first_hidden_layer_sizes_max: 512
  second_hidden_layer_sizes_min: 256
  second_hidden_layer_sizes_max: 1024
  learning_rates: [0.0001, 0.001, 0.01]
  batch_size: 32
  epochs: 50

xgb:
  gamma_min: 5.0
  gamma_max: 11.0
  learning_rate_min: 0.07
  learning_rate_max: 0.6
  n_estimators: [50, 100, 150]
  regulation_alphas: [0.00001, 0.01, 0.75]
  regulation_lambdas: [0.00001, 0.01, 0.45]
  min_child_weights: [1.5, 6, 10]
  subsamples: [0.6, 0.95]
  max_depths: [3, 6, 9]
```

## Output Interpretation

### Statistical Results
- **p-value < 0.05**: Significant differences between models
- **Kendall's W**: Agreement between rankings (0 = no agreement, 1 = perfect agreement)
- **Average Rank**: Lower is better (1 = best, 2 = second best, etc.)
- **Critical Difference**: Models within CD are not significantly different

### Model Selection
The best model is selected based on:
1. **Statistical Significance**: If Friedman test is significant, use average rank
2. **Performance**: If no significant differences, use highest mean score
3. **Robustness**: Consider standard deviation and confidence intervals

### Ablation Insights
- **Data Modes**: Shows impact of real vs synthetic vs combined data
- **Feature Modes**: Shows importance of engineered features vs keypoints alone
- **Interactions**: Full analysis reveals optimal combinations

## Troubleshooting

### Common Issues

1. **Subject Leakage Warnings**
   - Check that subject IDs are properly assigned
   - Verify StratifiedGroupKFold is working correctly
   - Review fold splits in detailed logs

2. **Insufficient Data for CV**
   - Ensure minimum 5 subjects per class
   - Check data filtering in ablation studies
   - Consider reducing CV folds if necessary

3. **Statistical Test Failures**
   - Verify all models have results
   - Check data format consistency
   - Ensure proper metric names

4. **Memory Issues**
   - Reduce hyperparameter search space
   - Use smaller neural network architectures
   - Enable early stopping

### Debugging Tips

1. **Enable Detailed Logging**
   ```python
   from loguru import logger
   logger.enable("__main__")
   ```

2. **Check Data Integrity**
   ```python
   # Verify no subject leakage
   train_subjects = set(train_df['subject_id'])
   test_subjects = set(test_df['subject_id'])
   assert len(train_subjects.intersection(test_subjects)) == 0
   ```

3. **Validate Fold Splits**
   ```python
   # Check StratifiedGroupKFold results
   for fold, (train_idx, test_idx) in enumerate(splits):
       train_subjs = set(groups[train_idx])
       test_subjs = set(groups[test_idx])
       print(f"Fold {fold}: {len(train_subjs)} train, {len(test_subjs)} test subjects")
   ```

## Migration from Old Pipeline

### Key Differences
1. **No View Types**: Remove `--combined` flags and view-specific logic
2. **New Training Scripts**: Use `train_standard.py` and `train_loso.py`
3. **Different Paths**: Data stored in `data/processed/` subdirectories
4. **Statistical Integration**: Built-in statistical analysis instead of separate scripts

### Migration Steps
1. **Backup Old Results**: Save existing `dvclive/` and `models/` directories
2. **Update Dependencies**: Ensure all required packages are installed
3. **Run New Pipeline**: Start with `dvc repro prepare_standard_split`
4. **Compare Results**: Validate that new results are consistent with expectations

## Best Practices

1. **Always Use Grouped CV**: Never split data without considering subject IDs
2. **Document Hyperparameters**: Save best parameters from standard training for LOSO
3. **Validate Statistically**: Don't rely on single metrics; use proper statistical tests
4. **Report Comprehensively**: Include confidence intervals and effect sizes
5. **Version Everything**: Use DVC for reproducible experiments

## Citation

When using this pipeline, please cite the methodology:

```
This analysis uses grouped k-fold cross validation with StratifiedGroupKFold 
to prevent subject-level data leakage, following best practices for 
person-independent evaluation in pose-based classification tasks. Statistical 
significance is assessed using Friedman tests with post-hoc Nemenyi correction 
for multiple comparisons.
```
