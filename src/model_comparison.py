import json
import os
import argparse
import pandas as pd
from pathlib import Path

base_path = Path("/teamspace/studios/this_studio/sitting-posture/dvclive")

def read_metrics_from_dvclive(combined_only=False):
    """
    Read metrics from all model-view combinations in the dvclive folder
    and return a structured DataFrame
    """
    # Define the models and view types available
    models = ["adaboost", "nn", "xgb"]

    if combined_only:
        view_types = ["combined"]
    else:
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

def create_summary_tables(df, mode="individual"):
    """
    Create various summary tables for better analysis
    """
    if df.empty:
        print("No data available for analysis")
        return

    mode_title = "Combined View" if mode == "combined" else "Individual Views"

    # Main performance metrics
    performance_cols = ['Model_View', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']
    df_performance = df[performance_cols].copy()

    # Format numerical columns to percentages
    numeric_cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']
    for col in numeric_cols:
        df_performance[col] = df_performance[col].apply(
            lambda x: f"{x*100:.2f}%" if isinstance(x, (int, float)) else x
        )

    print(f"Model Performance Summary - {mode_title} (Test Set)")
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

    print(f"\n\nCross-Validation Performance Summary - {mode_title}")
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

    print(f"\n\nEfficiency Summary - {mode_title}")
    print("=" * 60)
    print(df_efficiency.to_string(index=False))

def analyze_by_view(df):
    """
    Analyze performance by view type (only for individual views)
    """
    if df.empty:
        return

    # Skip view analysis if we're dealing with combined data
    if df['View'].iloc[0] == 'combined':
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

    if view_summary:
        view_df = pd.DataFrame(view_summary)
        print(view_df.to_string(index=False))

def analyze_by_model(df, mode="individual"):
    """
    Analyze performance by model type
    """
    if df.empty:
        return

    mode_title = "Combined View" if mode == "combined" else "Individual Views"
    print(f"\n\nPerformance Analysis by Model Type - {mode_title}")
    print("=" * 80)

    model_summary = []
    for model in ['adaboost', 'nn', 'xgb']:
        model_data = df[df['Model'] == model]
        if not model_data.empty:
            if mode == "combined":
                # For combined mode, just get the single row per model
                row = model_data.iloc[0]
                model_summary.append({
                    'Model': model,
                    'Accuracy': f"{row['Accuracy']*100:.2f}%" if isinstance(row['Accuracy'], (int, float)) else row['Accuracy'],
                    'F1': f"{row['F1']*100:.2f}%" if isinstance(row['F1'], (int, float)) else row['F1'],
                    'ROC_AUC': f"{row['ROC_AUC']*100:.2f}%" if isinstance(row['ROC_AUC'], (int, float)) else row['ROC_AUC'],
                    'Duration': f"{row['Duration']:.3f}s" if isinstance(row['Duration'], (int, float)) else row['Duration'],
                    'Energy': f"{row['Energy_Consumed']:.6f}" if isinstance(row['Energy_Consumed'], (int, float)) else row['Energy_Consumed']
                })
            else:
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

    if model_summary:
        model_df = pd.DataFrame(model_summary)
        print(model_df.to_string(index=False))

def find_best_performers(df, mode="individual"):
    """
    Find the best performing model-view combinations
    """
    if df.empty:
        return

    mode_title = "Combined View" if mode == "combined" else "Individual Views"
    print(f"\n\nBest Performing Models - {mode_title}")
    print("=" * 80)

    metrics_to_check = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']

    for metric in metrics_to_check:
        valid_data = df[df[metric] != 'N/A'].copy()
        if not valid_data.empty:
            best_idx = valid_data[metric].idxmax()
            best_combo = valid_data.loc[best_idx]
            print(f"Best {metric}: {best_combo['Model_View']} - {best_combo[metric]*100:.2f}%")

def compare_combined_vs_individual():
    """
    Compare combined models vs best individual view models
    """
    print("\n\n" + "="*80)
    print("COMBINED vs INDIVIDUAL VIEW COMPARISON")
    print("="*80)

    # Read both datasets
    df_individual = read_metrics_from_dvclive(combined_only=False)
    df_combined = read_metrics_from_dvclive(combined_only=True)

    if df_individual.empty or df_combined.empty:
        print("Cannot perform comparison - missing data")
        return

    # For each model, find best individual view and compare with combined
    comparison_data = []

    for model in ['adaboost', 'nn', 'xgb']:
        # Get combined performance
        combined_data = df_combined[df_combined['Model'] == model]
        if combined_data.empty:
            continue
        combined_row = combined_data.iloc[0]

        # Get best individual view for this model
        individual_data = df_individual[df_individual['Model'] == model]
        if individual_data.empty:
            continue

        # Find best view based on F1 score
        valid_f1_data = individual_data[individual_data['F1'] != 'N/A']
        if valid_f1_data.empty:
            continue

        best_individual_idx = valid_f1_data['F1'].idxmax()
        best_individual_row = valid_f1_data.loc[best_individual_idx]

        comparison_data.append({
            'Model': model,
            'Best_Individual_View': best_individual_row['View'],
            'Individual_F1': f"{best_individual_row['F1']*100:.2f}%" if isinstance(best_individual_row['F1'], (int, float)) else best_individual_row['F1'],
            'Combined_F1': f"{combined_row['F1']*100:.2f}%" if isinstance(combined_row['F1'], (int, float)) else combined_row['F1'],
            'Individual_Accuracy': f"{best_individual_row['Accuracy']*100:.2f}%" if isinstance(best_individual_row['Accuracy'], (int, float)) else best_individual_row['Accuracy'],
            'Combined_Accuracy': f"{combined_row['Accuracy']*100:.2f}%" if isinstance(combined_row['Accuracy'], (int, float)) else combined_row['Accuracy'],
            'Individual_ROC_AUC': f"{best_individual_row['ROC_AUC']*100:.2f}%" if isinstance(best_individual_row['ROC_AUC'], (int, float)) else best_individual_row['ROC_AUC'],
            'Combined_ROC_AUC': f"{combined_row['ROC_AUC']*100:.2f}%" if isinstance(combined_row['ROC_AUC'], (int, float)) else combined_row['ROC_AUC']
        })

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Compare model performance")
    parser.add_argument("--combined", action="store_true", help="Analyze only combined view models")
    parser.add_argument("--compare", action="store_true", help="Compare combined vs individual view models")
    args = parser.parse_args()

    if args.compare:
        compare_combined_vs_individual()
        return

    # Read metrics from model-view combinations
    df = read_metrics_from_dvclive(combined_only=args.combined)

    if df.empty:
        print("No metrics data found!")
        return

    # Sort by model and view for better readability
    df = df.sort_values(['Model', 'View'])

    mode = "combined" if args.combined else "individual"

    # Create comprehensive summary tables
    create_summary_tables(df, mode)

    # Analyze by view type (only for individual views)
    if not args.combined:
        analyze_by_view(df)

    # Analyze by model type
    analyze_by_model(df, mode)

    # Find best performers
    find_best_performers(df, mode)

    # Save detailed results to CSV
    suffix = "_combined" if args.combined else "_individual"
    output_file = base_path / f"detailed_model_view{suffix}_comparison.csv"
    df.to_csv(output_file, index=False)
    print(f"\n\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
