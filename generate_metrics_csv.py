#!/usr/bin/env python3

import json
import os
import argparse
import csv
from pathlib import Path

def read_metrics_file(file_path):
    """Read the metrics JSON file and return the data."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_fold_metrics(metrics_data):
    """Extract metrics for each fold."""
    results = []
    for i in range(1, 6):  # Folds 1 through 5
        fold_key = f"fold_{i}"
        if fold_key in metrics_data:
            fold_data = metrics_data[fold_key]
            results.append({
                'fold': i,
                'accuracy': fold_data['accuracy'],
                'precision': fold_data['precision'],
                'recall': fold_data['recall'],
                'f1': fold_data['f1']
            })
    return results

def generate_csv(model_paths, output_file):
    """Generate a CSV file with metrics for each model and fold."""
    # Define the fieldnames for the CSV
    fieldnames = ['Model', 'Fold', 'Acc', 'Prec', 'Rec', 'F1']

    # Create the CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Process each model
        for model_name, file_path in model_paths.items():
            if os.path.exists(file_path):
                metrics_data = read_metrics_file(file_path)
                fold_results = extract_fold_metrics(metrics_data)

                for fold_data in fold_results:
                    writer.writerow({
                        'Model': model_name,
                        'Fold': fold_data['fold'],
                        'Acc': fold_data['accuracy'],
                        'Prec': fold_data['precision'],
                        'Rec': fold_data['recall'],
                        'F1': fold_data['f1']
                    })
            else:
                print(f"Warning: Metrics file for {model_name} not found at {file_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate metrics CSV from JSON files')
    parser.add_argument('--output', '-o', type=str, default='metrics_table.csv',
                        help='Output CSV file path (default: metrics_table.csv)')
    args = parser.parse_args()

    # Define model paths
    base_path = Path('/teamspace/studios/this_studio/sitting-posture/dvclive')
    model_paths = {
        'Adaboost': str(base_path / 'adaboost/loso/metrics.json'),
        'MLP': str(base_path / 'nn/loso/metrics.json'),
        'XGBoost': str(base_path / 'xgb/loso/metrics.json')
    }

    # Generate CSV file
    generate_csv(model_paths, args.output)
    print(f"CSV file has been generated and saved to {args.output}")

if __name__ == "__main__":
    main()
