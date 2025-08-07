import json
import os
import pandas as pd
from pathlib import Path

base_path = Path("/teamspace/studios/this_studio/sitting-posture/dvclive")

def read_metrics_from_dvclive():
    """
    Read metrics from all model-view combinations in the dvclive folder
    and return a structured DataFrame
    """
    # Define the models and view types available
    models = ["adaboost", "nn", "xgb"]
    view_types = ["front", "left", "right"]

    data = []

    for model in models:
        for view_type in view_types:
            metrics_file = base_path / model / view_type / "metrics.json"

            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)

                    # Extract test metrics
                    test_metrics = metrics.get('test', {})

                    row = {
                        'Model': model,
                        'View': view_type,
                        'Model_View': f"{model}_{view_type}",
                        'Accuracy': test_metrics.get('accuracy', 'N/A'),
                        'Precision': test_metrics.get('precision', 'N/A'),
                        'Recall': test_metrics.get('recall', 'N/A'),
                        'F1': test_metrics.get('f1', 'N/A'),
                        'ROC_AUC': test_metrics.get('roc_auc', 'N/A'),
                        'Duration': test_metrics.get('duration', 'N/A'),
                        'Energy_Consumed': test_metrics.get('energy_consumed', 'N/A'),
                        'CV_Accuracy_Mean': metrics.get('cv_accuracy_mean', 'N/A'),
                        'CV_F1_Mean': metrics.get('cv_f1_mean', 'N/A'),
                        'CV_ROC_AUC_Mean': metrics.get('cv_roc_auc_mean', 'N/A')
                    }
                    data.append(row)

                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Error reading {metrics_file}: {e}")
                    continue
            else:
                print(f"Metrics file not found: {metrics_file}")

    return pd.DataFrame(data)

def create_summary_tables(df):
    """
    Create various summary tables for better analysis
    """
    if df.empty:
        print("No data available for analysis")
        return

    # Main performance metrics
    performance_cols = ['Model_View', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']
    df_performance = df[performance_cols].copy()

    # Format numerical columns to percentages
    numeric_cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']
    for col in numeric_cols:
        df_performance[col] = df_performance[col].apply(
            lambda x: f"{x*100:.2f}%" if isinstance(x, (int, float)) else x
        )

    print("Model Performance Summary (Test Set)")
    print("=" * 100)
    print(df_performance.to_string(index=False))

    # Cross-validation summary
    cv_cols = ['Model_View', 'CV_Accuracy_Mean', 'CV_F1_Mean', 'CV_ROC_AUC_Mean']
    df_cv = df[cv_cols].copy()

    cv_numeric_cols = ['CV_Accuracy_Mean', 'CV_F1_Mean', 'CV_ROC_AUC_Mean']
    for col in cv_numeric_cols:
        df_cv[col] = df_cv[col].apply(
            lambda x: f"{x*100:.2f}%" if isinstance(x, (int, float)) else x
        )

    print("\n\nCross-Validation Performance Summary")
    print("=" * 80)
    print(df_cv.to_string(index=False))

    # Efficiency summary
    efficiency_cols = ['Model_View', 'Duration', 'Energy_Consumed']
    df_efficiency = df[efficiency_cols].copy()

    # Format duration and energy
    df_efficiency['Duration'] = df_efficiency['Duration'].apply(
        lambda x: f"{x:.3f}s" if isinstance(x, (int, float)) else x
    )
    df_efficiency['Energy_Consumed'] = df_efficiency['Energy_Consumed'].apply(
        lambda x: f"{x:.6f}" if isinstance(x, (int, float)) else x
    )

    print("\n\nEfficiency Summary")
    print("=" * 60)
    print(df_efficiency.to_string(index=False))

def analyze_by_view(df):
    """
    Analyze performance by view type
    """
    if df.empty:
        return

    print("\n\nPerformance Analysis by View Type")
    print("=" * 80)

    view_summary = []
    for view in ['front', 'left', 'right']:
        view_data = df[df['View'] == view]
        if not view_data.empty:
            # Calculate average metrics for this view across all models
            avg_acc = view_data['Accuracy'].apply(lambda x: x if isinstance(x, (int, float)) else None).mean()
            avg_f1 = view_data['F1'].apply(lambda x: x if isinstance(x, (int, float)) else None).mean()
            avg_roc = view_data['ROC_AUC'].apply(lambda x: x if isinstance(x, (int, float)) else None).mean()

            view_summary.append({
                'View': view,
                'Avg_Accuracy': f"{avg_acc*100:.2f}%" if avg_acc is not None else 'N/A',
                'Avg_F1': f"{avg_f1*100:.2f}%" if avg_f1 is not None else 'N/A',
                'Avg_ROC_AUC': f"{avg_roc*100:.2f}%" if avg_roc is not None else 'N/A',
                'Best_Model': view_data.loc[view_data['F1'].idxmax(), 'Model'] if not view_data['F1'].isna().all() else 'N/A'
            })

    view_df = pd.DataFrame(view_summary)
    print(view_df.to_string(index=False))

def analyze_by_model(df):
    """
    Analyze performance by model type
    """
    if df.empty:
        return

    print("\n\nPerformance Analysis by Model Type")
    print("=" * 80)

    model_summary = []
    for model in ['adaboost', 'nn', 'xgb']:
        model_data = df[df['Model'] == model]
        if not model_data.empty:
            # Calculate average metrics for this model across all views
            avg_acc = model_data['Accuracy'].apply(lambda x: x if isinstance(x, (int, float)) else None).mean()
            avg_f1 = model_data['F1'].apply(lambda x: x if isinstance(x, (int, float)) else None).mean()
            avg_roc = model_data['ROC_AUC'].apply(lambda x: x if isinstance(x, (int, float)) else None).mean()

            model_summary.append({
                'Model': model,
                'Avg_Accuracy': f"{avg_acc*100:.2f}%" if avg_acc is not None else 'N/A',
                'Avg_F1': f"{avg_f1*100:.2f}%" if avg_f1 is not None else 'N/A',
                'Avg_ROC_AUC': f"{avg_roc*100:.2f}%" if avg_roc is not None else 'N/A',
                'Best_View': model_data.loc[model_data['F1'].idxmax(), 'View'] if not model_data['F1'].isna().all() else 'N/A'
            })

    model_df = pd.DataFrame(model_summary)
    print(model_df.to_string(index=False))

def find_best_performers(df):
    """
    Find the best performing model-view combinations
    """
    if df.empty:
        return

    print("\n\nBest Performing Model-View Combinations")
    print("=" * 80)

    metrics_to_check = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']

    for metric in metrics_to_check:
        valid_data = df[df[metric] != 'N/A'].copy()
        if not valid_data.empty:
            best_idx = valid_data[metric].idxmax()
            best_combo = valid_data.loc[best_idx]
            print(f"Best {metric}: {best_combo['Model_View']} - {best_combo[metric]*100:.2f}%")

def main():
    # Read all metrics from model-view combinations
    df = read_metrics_from_dvclive()

    if df.empty:
        print("No metrics data found!")
        return

    # Sort by model and view for better readability
    df = df.sort_values(['Model', 'View'])

    # Create comprehensive summary tables
    create_summary_tables(df)

    # Analyze by view type
    analyze_by_view(df)

    # Analyze by model type
    analyze_by_model(df)

    # Find best performers
    find_best_performers(df)

    # Save detailed results to CSV
    output_file = base_path / "detailed_model_view_comparison.csv"
    df.to_csv(output_file, index=False)
    print(f"\n\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
