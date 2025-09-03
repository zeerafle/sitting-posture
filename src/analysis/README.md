# Statistical Analysis Framework

A modular, extensible framework for comparing machine learning models using non-parametric statistical tests. The framework supports Cross-Validation (CV), Leave-One-Subject-Out (LOSO), and ablation study results.

## Features

- **Modular Design**: Clean separation of data loading, statistical tests, and output handling
- **Multiple Analysis Types**: Supports individual views, combined views, LOSO, and ablation studies
- **Extensible**: Easy to add new statistical tests and data sources
- **Under 100 Lines**: Main script is concise while maintaining full functionality
- **Comprehensive Output**: Detailed results with effect sizes and post-hoc comparisons

## Architecture

```
src/analysis/
├── friedman_test.py          # Main entry point (77 lines)
└── statistical/              # Modular components
    ├── __init__.py          # Package initialization
    ├── data_loader.py       # Data loading utilities
    ├── tests.py             # Statistical tests implementation
    └── output.py            # Result formatting and saving
```

## Usage Examples

### 1. Individual View Analysis (Original Functionality)
Compare models across all three camera views:
```bash
uv run python src/analysis/friedman_test.py \
  --models adaboost nn xgb \
  --metric accuracy
```

### 2. Combined View Analysis
Compare models using combined view results:
```bash
uv run python src/analysis/friedman_test.py \
  --models adaboost nn xgb \
  --metric accuracy \
  --combined
```

### 3. LOSO Analysis
Compare models using Leave-One-Subject-Out results:
```bash
uv run python src/analysis/friedman_test.py \
  --models adaboost nn xgb \
  --metric accuracy \
  --loso
```

### 4. Ablation Study Analysis (New Feature)
Compare different data/feature configurations:
```bash
# Compare data modes with keypoints only
uv run python src/analysis/friedman_test.py \
  --models real_keypoints_only synthetic_keypoints_only all_keypoints_only \
  --metric accuracy \
  --ablation

# Compare feature types with all data
uv run python src/analysis/friedman_test.py \
  --models all_keypoints_only all_all_features \
  --metric f1 \
  --ablation

# Full ablation comparison
uv run python src/analysis/friedman_test.py \
  --models real_keypoints_only real_all_features synthetic_keypoints_only synthetic_all_features all_keypoints_only all_all_features \
  --metric accuracy \
  --ablation
```

## Supported Metrics

- `accuracy` - Classification accuracy
- `f1` - F1 score
- `precision` - Precision score
- `recall` - Recall score
- `roc_auc` - ROC AUC score

## Output

The framework provides:
- **Console Output**: Formatted statistical results with interpretation
- **JSON Files**: Structured results saved to `dvclive/analysis/`
- **Effect Sizes**: Kendall's W for practical significance
- **Post-hoc Tests**: Pairwise Wilcoxon tests with Bonferroni correction

### Sample Output
```
============================================================
FRIEDMAN TEST RESULTS - ABLATION ANALYSIS
============================================================
Models compared: real_keypoints_only, synthetic_keypoints_only, all_keypoints_only
Metric: accuracy
Number of models: 3
Number of observations: 5

Test Statistics:
  Friedman χ² statistic: 0.4000
  P-value: 0.818731
  Kendall's W (effect size): 0.0400

✗ NO SIGNIFICANT DIFFERENCE (p ≥ 0.05)
Effect Size: Small (Kendall's W = 0.0400)

Model Performance Summary:
  real_keypoints_only: Mean = 0.9131, Std = 0.0373
  synthetic_keypoints_only: Mean = 0.9252, Std = 0.0125
  all_keypoints_only: Mean = 0.9150, Std = 0.0234
```

## DVC Integration

The framework is integrated into the DVC pipeline with stages for:
- `analyze_friedman*` - Original CV and LOSO analyses
- `analyze_friedman_ablation_*` - Ablation study analyses

Run ablation analysis via DVC:
```bash
dvc repro analyze_friedman_ablation_data_modes
dvc repro analyze_friedman_ablation_features_f1
```

## Extending the Framework

### Adding New Statistical Tests
1. Add test functions to `statistical/tests.py`
2. Update `create_result_summary()` to include new metrics
3. Modify main script to call new tests

### Adding New Data Sources
1. Add loader functions to `statistical/data_loader.py`
2. Update `load_model_results()` to handle new analysis types
3. Add command line arguments for new sources

### Adding New Output Formats
1. Add formatting functions to `statistical/output.py`
2. Update `save_results()` to support new formats

## Requirements

- scipy >= 1.7.0 (for statistical tests)
- pandas >= 1.3.0 (for data handling)
- numpy >= 1.21.0 (for numerical operations)

## Error Handling

The framework handles common issues:
- **Missing files**: Graceful degradation with warnings
- **Insufficient data**: Clear error messages for statistical requirements
- **Mismatched lengths**: Automatic truncation with warnings
- **Invalid metrics**: Fallback to available data keys