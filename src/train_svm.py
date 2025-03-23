#!/usr/bin/env python3
"""
Training script for SVM model for Kubernetes Node Failure Prediction.
This script loads data, trains an SVM model, and saves the trained model.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from models.svm_model import SVMNodeFailurePredictor

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train SVM model for Kubernetes Node Failure Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data-path', type=str, default='src/data/generated_metrics.csv',
                        help='Path to training data CSV file')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Size of test split (0.0-1.0)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--model-dir', type=str, default='src/models/saved',
                        help='Directory to save trained model')
    parser.add_argument('--metrics-dir', type=str, default='src/metrics',
                        help='Directory to save model metrics')
    parser.add_argument('--grid-search', action='store_true',
                        help='Perform grid search for hyperparameter tuning')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations after training')
    
    return parser.parse_args()

def load_data(data_path):
    """
    Load and preprocess training data.
    
    Args:
        data_path (str): Path to training data CSV file
        
    Returns:
        DataFrame: Preprocessed data
    """
    try:
        # Load data
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # Print data info
        print(f"Loaded {len(df)} records")
        failure_col = 'failure' if 'failure' in df.columns else 'node_failure'
        print(f"Failure rate: {df[failure_col].mean():.2%}")
        
        # Verify columns
        required_columns = [
            'cpu_usage_percent', 'memory_usage_percent', 'network_latency_ms',
            'disk_io_mbps', 'error_rate', 'avg_response_time_ms',
            'pod_restarts', 'cpu_throttling_percent'
        ]
        
        # Map from alternative column names
        column_mapping = {
            'cpu_pct': 'cpu_usage_percent',
            'memory_pct': 'memory_usage_percent',
            'disk_io': 'disk_io_mbps',
            'cpu_throttling': 'cpu_throttling_percent',
            'node_failure': 'failure'
        }
        
        # Rename columns if needed
        for alt_col, std_col in column_mapping.items():
            if alt_col in df.columns and std_col not in df.columns:
                df = df.rename(columns={alt_col: std_col})
                print(f"Renamed column '{alt_col}' to '{std_col}'")
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing required columns: {missing_columns}")
            raise ValueError(f"Data is missing required columns: {missing_columns}")
        
        # Ensure failure column exists
        if 'failure' not in df.columns and 'node_failure' in df.columns:
            df = df.rename(columns={'node_failure': 'failure'})
        elif 'failure' not in df.columns:
            raise ValueError("No failure column found in dataset")
        
        # Return preprocessed data
        return df
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)

def train_model(args):
    """
    Train and evaluate SVM model.
    
    Args:
        args: Command line arguments
        
    Returns:
        SVMNodeFailurePredictor: Trained model
    """
    # Load data
    data = load_data(args.data_path)
    
    # Create model
    print("Initializing SVM model...")
    model = SVMNodeFailurePredictor()
    
    # Preprocess data
    X_train, X_test, y_train, y_test = model.preprocess_data(
        data, 
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Train model
    print(f"Training model ({'with' if args.grid_search else 'without'} grid search)...")
    best_params = model.train(
        X_train,
        y_train,
        use_grid_search=args.grid_search
    )
    print(f"Best parameters: {best_params}")
    
    # Evaluate model
    print("Evaluating model performance...")
    metrics = model.evaluate(X_test, y_test)
    
    # Save model and metrics
    print("Saving model and metrics...")
    model.save(
        model_dir=args.model_dir,
        metrics_dir=args.metrics_dir,
        metrics=metrics
    )
    
    # Print metrics summary
    print("\nModel Performance Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    if 'roc_auc' in metrics:
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    
    return model, metrics

def visualize_results(metrics, model_dir='src/models/saved'):
    """
    Generate and save visualizations.
    
    Args:
        metrics: Model evaluation metrics
        model_dir: Directory containing model files
    """
    try:
        # Create plots directory if it doesn't exist
        plots_dir = 'src/plots'
        os.makedirs(plots_dir, exist_ok=True)
        
        # Import visualization modules
        print("Generating visualizations...")
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Failure', 'Failure'],
                    yticklabels=['No Failure', 'Failure'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/confusion_matrix.png')
        plt.close()
        
        # Plot metrics as a bar chart
        plt.figure(figsize=(10, 6))
        metric_values = [metrics['accuracy'], metrics['precision'], 
                         metrics['recall'], metrics['f1_score']]
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        sns.barplot(x=metric_names, y=metric_values)
        plt.title('Model Performance Metrics')
        plt.ylim(0, 1.0)
        for i, v in enumerate(metric_values):
            plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/model_metrics.png')
        plt.close()
        
        # Create visualization metrics data
        vis_metrics = {
            'timestamp': metrics['timestamp'],
            'metrics': {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'confusion_matrix': metrics['confusion_matrix']
            }
        }
        
        # Save visualization data for later use
        with open(f'{plots_dir}/visualization_data.json', 'w') as f:
            json.dump(vis_metrics, f, indent=4)
        
        print(f"Visualizations saved to {plots_dir}/")

    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")

def main():
    """Main function to train the model."""
    # Parse arguments
    args = parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)
    
    # Train and evaluate model
    model, metrics = train_model(args)
    
    # Generate visualizations if requested
    if args.visualize:
        visualize_results(metrics, args.model_dir)
    
    print("Training completed successfully!")
    return 0

if __name__ == '__main__':
    sys.exit(main()) 