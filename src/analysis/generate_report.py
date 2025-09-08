import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import jinja2


def load_loso_analysis_results(analysis_dir):
    """Load LOSO analysis results."""
    results = {}

    for metric in ['accuracy', 'f1']:
        results_path = os.path.join(analysis_dir, 'loso', f'statistical_results_{metric}.json')
        rankings_path = os.path.join(analysis_dir, 'loso', f'model_rankings_{metric}.csv')

        if os.path.exists(results_path) and os.path.exists(rankings_path):
            with open(results_path, 'r') as f:
                statistical_results = json.load(f)
            rankings = pd.read_csv(rankings_path)

            results[metric] = {
                'statistical_results': statistical_results,
                'model_rankings': rankings
            }

    return results


def load_ablation_analysis_results(analysis_dir):
    """Load ablation analysis results."""
    results = {}

    experiment_types = ['data_modes', 'feature_modes', 'full']

    for exp_type in experiment_types:
        exp_dir = os.path.join(analysis_dir, 'ablation', exp_type)
        if not os.path.exists(exp_dir):
            continue

        results[exp_type] = {}

        for metric in ['accuracy', 'f1']:
            results_path = os.path.join(exp_dir, f'statistical_results_{metric}.json')
            rankings_path = os.path.join(exp_dir, f'model_rankings_{metric}.csv')

            if os.path.exists(results_path) and os.path.exists(rankings_path):
                with open(results_path, 'r') as f:
                    statistical_results = json.load(f)
                rankings = pd.read_csv(rankings_path)

                results[exp_type][metric] = {
                    'statistical_results': statistical_results,
                    'model_rankings': rankings
                }

    return results


def load_best_model_info(analysis_dir):
    """Load best model selection information."""
    best_model_path = os.path.join(analysis_dir, 'best_model.json')

    if os.path.exists(best_model_path):
        with open(best_model_path, 'r') as f:
            return json.load(f)

    return None


def load_individual_model_results(dvclive_dir):
    """Load individual model results from DVCLive."""
    models = ['adaboost', 'nn', 'xgb', 'cnn']
    results = {}

    for model in models:
        # Standard results
        standard_metrics_path = os.path.join(dvclive_dir, model, 'standard', 'metrics.json')
        if os.path.exists(standard_metrics_path):
            with open(standard_metrics_path, 'r') as f:
                standard_metrics = json.load(f)
        else:
            standard_metrics = {}

        # LOSO results
        loso_metrics_path = os.path.join(dvclive_dir, model, 'loso', 'aggregated_metrics.json')
        if os.path.exists(loso_metrics_path):
            with open(loso_metrics_path, 'r') as f:
                loso_metrics = json.load(f)
        else:
            loso_metrics = {}

        results[model] = {
            'standard': standard_metrics,
            'loso': loso_metrics
        }

    return results


def create_performance_comparison_plot(model_results, output_dir):
    """Create performance comparison plots."""
    # Prepare data for plotting
    models = list(model_results.keys())

    # Standard vs LOSO comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison: Standard vs LOSO', fontsize=16)

    metrics = ['accuracy', 'f1']
    evaluation_types = ['standard', 'loso']

    for i, metric in enumerate(metrics):
        for j, eval_type in enumerate(evaluation_types):
            ax = axes[i, j]

            metric_values = []
            model_names = []

            for model in models:
                if eval_type == 'standard':
                    value = model_results[model]['standard'].get(f'test_{metric}', 0)
                else:
                    value = model_results[model]['loso'].get(f'{metric}_mean', 0)

                if value > 0:
                    metric_values.append(value)
                    model_names.append(model)

            if metric_values:
                bars = ax.bar(model_names, metric_values, alpha=0.7)
                ax.set_title(f'{metric.capitalize()} - {eval_type.capitalize()}')
                ax.set_ylabel(metric.capitalize())
                ax.set_ylim(0, 1)

                # Add value labels on bars
                for bar, value in zip(bars, metric_values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_performance_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def create_ablation_summary_plot(ablation_results, output_dir):
    """Create ablation experiment summary plots."""
    if not ablation_results:
        return

    # Data modes comparison
    if 'data_modes' in ablation_results:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Ablation Study: Data Modes Comparison', fontsize=16)

        for i, metric in enumerate(['accuracy', 'f1']):
            if metric in ablation_results['data_modes']:
                rankings = ablation_results['data_modes'][metric]['model_rankings']

                ax = axes[i]
                bars = ax.bar(rankings['model'], rankings['mean_score'],
                             yerr=rankings['std_score'], capsize=5, alpha=0.7)
                ax.set_title(f'{metric.capitalize()} by Data Mode')
                ax.set_ylabel(f'Mean {metric.capitalize()}')
                ax.set_xticklabels(rankings['model'], rotation=45, ha='right')

                # Add value labels
                for bar, mean_val, std_val in zip(bars, rankings['mean_score'], rankings['std_score']):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01,
                           f'{mean_val:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ablation_data_modes.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # Feature modes comparison
    if 'feature_modes' in ablation_results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Ablation Study: Feature Modes Comparison', fontsize=16)

        for i, metric in enumerate(['accuracy', 'f1']):
            if metric in ablation_results['feature_modes']:
                rankings = ablation_results['feature_modes'][metric]['model_rankings']

                ax = axes[i]
                bars = ax.bar(rankings['model'], rankings['mean_score'],
                             yerr=rankings['std_score'], capsize=5, alpha=0.7)
                ax.set_title(f'{metric.capitalize()} by Feature Mode')
                ax.set_ylabel(f'Mean {metric.capitalize()}')
                ax.set_xticklabels(rankings['model'], rotation=45, ha='right')

                # Add value labels
                for bar, mean_val, std_val in zip(bars, rankings['mean_score'], rankings['std_score']):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01,
                           f'{mean_val:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ablation_feature_modes.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


def generate_summary_statistics(loso_results, ablation_results, model_results, best_model_info):
    """Generate summary statistics table."""
    summary_data = []

    # LOSO model comparison
    if 'accuracy' in loso_results:
        loso_rankings = loso_results['accuracy']['model_rankings']
        for _, row in loso_rankings.iterrows():
            summary_data.append({
                'Analysis': 'LOSO Model Comparison',
                'Configuration': row['model'],
                'Metric': 'Accuracy',
                'Mean': f"{row['mean_score']:.4f}",
                'Std': f"{row['std_score']:.4f}",
                'Rank': f"{row['average_rank']:.2f}"
            })

    # Best model information
    if best_model_info:
        summary_data.append({
            'Analysis': 'Best Model Selection',
            'Configuration': best_model_info['model_name'],
            'Metric': best_model_info['metric'],
            'Mean': f"{best_model_info['performance']['mean_score']:.4f}",
            'Std': f"{best_model_info['performance']['std_score']:.4f}",
            'Rank': f"{best_model_info['performance']['average_rank']:.2f}"
        })

    # Ablation results
    for exp_type, exp_results in ablation_results.items():
        if 'accuracy' in exp_results:
            rankings = exp_results['accuracy']['model_rankings']
            for _, row in rankings.iterrows():
                summary_data.append({
                    'Analysis': f'Ablation - {exp_type.replace("_", " ").title()}',
                    'Configuration': row['model'],
                    'Metric': 'Accuracy',
                    'Mean': f"{row['mean_score']:.4f}",
                    'Std': f"{row['std_score']:.4f}",
                    'Rank': f"{row['average_rank']:.2f}"
                })

    return pd.DataFrame(summary_data)


def generate_html_report(loso_results, ablation_results, model_results, best_model_info, summary_stats, output_dir):
    """Generate comprehensive HTML report."""

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sitting Posture Classification - Comprehensive Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
            .header { background-color: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 30px; }
            .section { margin-bottom: 40px; }
            .section h2 { color: #333; border-bottom: 2px solid #007cba; padding-bottom: 10px; }
            .section h3 { color: #666; margin-top: 25px; }
            .metrics-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            .metrics-table th, .metrics-table td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            .metrics-table th { background-color: #f2f2f2; font-weight: bold; }
            .metrics-table tr:nth-child(even) { background-color: #f9f9f9; }
            .highlight { background-color: #e7f3ff; font-weight: bold; }
            .statistical-result { background-color: #f0f8f0; padding: 15px; border-radius: 5px; margin: 10px 0; }
            .warning { background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #ffc107; }
            .success { background-color: #d4edda; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #28a745; }
            .image-container { text-align: center; margin: 20px 0; }
            .image-container img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
            .key-findings { background-color: #e3f2fd; padding: 20px; border-radius: 5px; margin: 20px 0; }
            .methodology { background-color: #f5f5f5; padding: 20px; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Sitting Posture Classification - Comprehensive Analysis Report</h1>
            <p><strong>Generated:</strong> {{ timestamp }}</p>
            <p><strong>Analysis Type:</strong> LOSO with Grouped 5-Fold Cross Validation and Ablation Studies</p>
        </div>

        <div class="section">
            <h2>Executive Summary</h2>
            <div class="key-findings">
                <h3>Key Findings</h3>
                {% if best_model_info %}
                <p><strong>Best Performing Model:</strong> {{ best_model_info.model_name }}</p>
                <p><strong>Selection Metric:</strong> {{ best_model_info.metric }}</p>
                <p><strong>Performance:</strong> {{ "%.4f"|format(best_model_info.performance.mean_score) }} ± {{ "%.4f"|format(best_model_info.performance.std_score) }}</p>
                <p><strong>Average Rank:</strong> {{ "%.2f"|format(best_model_info.performance.average_rank) }}</p>
                {% endif %}
            </div>
        </div>

        <div class="section">
            <h2>Methodology</h2>
            <div class="methodology">
                <h3>Experimental Design</h3>
                <ol>
                    <li><strong>Data Preparation:</strong> Combined view data (ignoring front/left/right distinctions)</li>
                    <li><strong>Standard Evaluation:</strong> Stratified train-test split with hyperparameter tuning</li>
                    <li><strong>LOSO Evaluation:</strong> Grouped 5-fold cross validation by participant (StratifiedGroupKFold)</li>
                    <li><strong>Statistical Analysis:</strong> Friedman test with post-hoc Nemenyi test</li>
                    <li><strong>Best Model Selection:</strong> Based on statistical significance and average ranking</li>
                    <li><strong>Ablation Studies:</strong> Testing different data modes and feature sets</li>
                </ol>
            </div>
        </div>

        <div class="section">
            <h2>Model Comparison Results (LOSO)</h2>
            {% if loso_results %}
                {% for metric, results in loso_results.items() %}
                <h3>{{ metric.title() }} Analysis</h3>

                <div class="statistical-result">
                    <h4>Statistical Test Results</h4>
                    <p><strong>Friedman Test p-value:</strong> {{ "%.6f"|format(results.statistical_results.friedman_test.p_value) }}</p>
                    <p><strong>Significant Differences:</strong> {{ "Yes" if results.statistical_results.friedman_test.significant else "No" }}</p>
                    <p><strong>Kendall's W:</strong> {{ "%.4f"|format(results.statistical_results.kendalls_w.kendalls_w) }}</p>
                </div>

                <table class="metrics-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Average Rank</th>
                            <th>Mean {{ metric.title() }}</th>
                            <th>Std {{ metric.title() }}</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for _, row in results.model_rankings.iterrows() %}
                        <tr>
                            <td>{{ row.model }}</td>
                            <td>{{ "%.3f"|format(row.average_rank) }}</td>
                            <td>{{ "%.4f"|format(row.mean_score) }}</td>
                            <td>{{ "%.4f"|format(row.std_score) }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                {% endfor %}
            {% endif %}
        </div>

        <div class="section">
            <h2>Ablation Study Results</h2>
            {% if ablation_results %}
                {% for exp_type, exp_results in ablation_results.items() %}
                <h3>{{ exp_type.replace('_', ' ').title() }}</h3>

                {% for metric, results in exp_results.items() %}
                <h4>{{ metric.title() }} Results</h4>

                <div class="statistical-result">
                    <p><strong>Friedman Test p-value:</strong> {{ "%.6f"|format(results.statistical_results.friedman_test.p_value) }}</p>
                    <p><strong>Significant Differences:</strong> {{ "Yes" if results.statistical_results.friedman_test.significant else "No" }}</p>
                    <p><strong>Kendall's W:</strong> {{ "%.4f"|format(results.statistical_results.kendalls_w.kendalls_w) }}</p>
                </div>

                <table class="metrics-table">
                    <thead>
                        <tr>
                            <th>Configuration</th>
                            <th>Average Rank</th>
                            <th>Mean {{ metric.title() }}</th>
                            <th>Std {{ metric.title() }}</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for _, row in results.model_rankings.iterrows() %}
                        <tr>
                            <td>{{ row.model }}</td>
                            <td>{{ "%.3f"|format(row.average_rank) }}</td>
                            <td>{{ "%.4f"|format(row.mean_score) }}</td>
                            <td>{{ "%.4f"|format(row.std_score) }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                {% endfor %}
                {% endfor %}
            {% endif %}
        </div>

        <div class="section">
            <h2>Individual Model Performance</h2>
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Standard Accuracy</th>
                        <th>Standard F1</th>
                        <th>LOSO Accuracy</th>
                        <th>LOSO F1</th>
                    </tr>
                </thead>
                <tbody>
                    {% for model, results in model_results.items() %}
                    <tr>
                        <td>{{ model }}</td>
                        <td>{{ "%.4f"|format(results.standard.get('test_accuracy', 0)) }}</td>
                        <td>{{ "%.4f"|format(results.standard.get('test_f1', 0)) }}</td>
                        <td>{{ "%.4f"|format(results.loso.get('accuracy_mean', 0)) }} ± {{ "%.4f"|format(results.loso.get('accuracy_std', 0)) }}</td>
                        <td>{{ "%.4f"|format(results.loso.get('f1_mean', 0)) }} ± {{ "%.4f"|format(results.loso.get('f1_std', 0)) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Summary Statistics</h2>
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Analysis</th>
                        <th>Configuration</th>
                        <th>Metric</th>
                        <th>Mean</th>
                        <th>Std</th>
                        <th>Rank</th>
                    </tr>
                </thead>
                <tbody>
                    {% for _, row in summary_stats.iterrows() %}
                    <tr>
                        <td>{{ row.Analysis }}</td>
                        <td>{{ row.Configuration }}</td>
                        <td>{{ row.Metric }}</td>
                        <td>{{ row.Mean }}</td>
                        <td>{{ row.Std }}</td>
                        <td>{{ row.Rank }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Visualizations</h2>
            <div class="image-container">
                <h3>Model Performance Comparison</h3>
                <img src="model_performance_comparison.png" alt="Model Performance Comparison">
            </div>

            <div class="image-container">
                <h3>Ablation Study - Data Modes</h3>
                <img src="ablation_data_modes.png" alt="Ablation Study Data Modes">
            </div>

            <div class="image-container">
                <h3>Ablation Study - Feature Modes</h3>
                <img src="ablation_feature_modes.png" alt="Ablation Study Feature Modes">
            </div>
        </div>

        <div class="section">
            <h2>Conclusions and Recommendations</h2>
            {% if best_model_info %}
            <div class="success">
                <h3>Recommended Model</h3>
                <p>Based on the comprehensive statistical analysis, the <strong>{{ best_model_info.model_name }}</strong> model
                is recommended for sitting posture classification with the following performance characteristics:</p>
                <ul>
                    <li>Mean {{ best_model_info.metric }}: {{ "%.4f"|format(best_model_info.performance.mean_score) }} ± {{ "%.4f"|format(best_model_info.performance.std_score) }}</li>
                    <li>Average Rank: {{ "%.2f"|format(best_model_info.performance.average_rank) }}</li>
                    <li>Selection Reason: {{ best_model_info.selection_reason }}</li>
                </ul>
            </div>
            {% endif %}

            <div class="methodology">
                <h3>Key Insights</h3>
                <ul>
                    <li>The LOSO evaluation provides robust estimates of model generalization to new subjects</li>
                    <li>Grouped 5-fold cross validation prevents subject-level data leakage</li>
                    <li>Statistical significance testing ensures reliable model comparisons</li>
                    <li>Ablation studies reveal the importance of different data sources and feature types</li>
                </ul>
            </div>
        </div>

        <div class="section">
            <h2>Technical Notes</h2>
            <div class="warning">
                <h3>Important Considerations</h3>
                <ul>
                    <li>All evaluations use grouped cross-validation to prevent subject-level leakage</li>
                    <li>Statistical tests account for multiple comparisons using appropriate corrections</li>
                    <li>Ablation experiments use the best-performing model for fair comparison</li>
                    <li>Results should be interpreted considering the specific dataset characteristics</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """

    # Render the template
    template = jinja2.Template(html_template)

    html_content = template.render(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        loso_results=loso_results,
        ablation_results=ablation_results,
        model_results=model_results,
        best_model_info=best_model_info,
        summary_stats=summary_stats
    )

    # Save HTML report
    report_path = os.path.join(output_dir, "comprehensive_analysis_report.html")
    with open(report_path, 'w') as f:
        f.write(html_content)

    logger.info(f"Generated HTML report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive analysis report')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save the report and visualizations')
    parser.add_argument('--dvclive-path', type=str, default='dvclive',
                       help='Path to DVCLive logs')
    parser.add_argument('--analysis-dir', type=str, default='dvclive/analysis',
                       help='Directory containing analysis results')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading analysis results...")

    # Load all analysis results
    loso_results = load_loso_analysis_results(args.analysis_dir)
    ablation_results = load_ablation_analysis_results(args.analysis_dir)
    best_model_info = load_best_model_info(args.analysis_dir)
    model_results = load_individual_model_results(args.dvclive_path)

    logger.info("Generating visualizations...")

    # Create visualizations
    create_performance_comparison_plot(model_results, args.output_dir)
    create_ablation_summary_plot(ablation_results, args.output_dir)

    logger.info("Generating summary statistics...")

    # Generate summary statistics
    summary_stats = generate_summary_statistics(
        loso_results, ablation_results, model_results, best_model_info
    )

    # Save summary statistics
    summary_path = os.path.join(args.output_dir, "summary_statistics.csv")
    summary_stats.to_csv(summary_path, index=False)
    logger.info(f"Saved summary statistics to {summary_path}")

    logger.info("Generating HTML report...")

    # Generate comprehensive HTML report
    generate_html_report(
        loso_results, ablation_results, model_results,
        best_model_info, summary_stats, args.output_dir
    )

    logger.success(f"Comprehensive analysis report generated successfully in {args.output_dir}")

    # Print summary
    print(f"\n=== REPORT GENERATION SUMMARY ===")
    print(f"Output directory: {args.output_dir}")
    print(f"Files generated:")
    print(f"  - comprehensive_analysis_report.html")
    print(f"  - summary_statistics.csv")
    print(f"  - model_performance_comparison.png")
    if ablation_results:
        print(f"  - ablation_data_modes.png")
        print(f"  - ablation_feature_modes.png")

    if best_model_info:
        print(f"\nBest Model: {best_model_info['model_name']}")
        print(f"Performance: {best_model_info['performance']['mean_score']:.4f} ± {best_model_info['performance']['std_score']:.4f}")


if __name__ == "__main__":
    main()
