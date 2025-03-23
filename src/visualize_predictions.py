#!/usr/bin/env python3
"""
Visualization module for Kubernetes Node Failure Prediction.
Generates various plots to visualize model performance and prediction results.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve

def create_dir_if_not_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_metrics(metrics_file='src/metrics/svm_metrics.json'):
    """Load model metrics from JSON file."""
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Metrics file {metrics_file} not found.")
        return None

def load_prediction_history(history_file='src/data/predictions/prediction_history.csv'):
    """Load prediction history from CSV file."""
    try:
        return pd.read_csv(history_file)
    except FileNotFoundError:
        print(f"Prediction history file {history_file} not found.")
        return None

def plot_learning_curve(metrics):
    """Plot learning curve for the model."""
    if not metrics or 'learning_curve' not in metrics:
        print("Learning curve data not found in metrics.")
        return
    
    plt.figure(figsize=(10, 6))
    
    train_sizes = metrics['learning_curve']['train_sizes']
    train_scores = metrics['learning_curve']['train_scores']
    val_scores = metrics['learning_curve']['validation_scores']
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(val_scores, axis=1)
    test_scores_std = np.std(val_scores, axis=1)
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.title("Learning Curve (SVM)")
    plt.xlabel("Training examples")
    plt.ylabel("F1 Score")
    plt.legend(loc="best")
    plt.grid(True)
    
    create_dir_if_not_exists('src/plots')
    plt.savefig('src/plots/learning_curve.png')
    plt.close()
    print("Learning curve plot saved to src/plots/learning_curve.png")

def plot_roc_curve(metrics):
    """Plot ROC curve for the model."""
    if not metrics or 'roc' not in metrics:
        print("ROC curve data not found in metrics.")
        return
    
    plt.figure(figsize=(10, 6))
    
    fpr = metrics['roc']['fpr']
    tpr = metrics['roc']['tpr']
    roc_auc = metrics['roc']['auc']
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    create_dir_if_not_exists('src/plots')
    plt.savefig('src/plots/roc_curve.png')
    plt.close()
    print("ROC curve plot saved to src/plots/roc_curve.png")

def plot_confusion_matrix(metrics):
    """Plot confusion matrix for the model."""
    if not metrics or 'confusion_matrix' not in metrics:
        print("Confusion matrix data not found in metrics.")
        return
    
    cm = metrics['confusion_matrix']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Failure'],
                yticklabels=['Normal', 'Failure'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    create_dir_if_not_exists('src/plots')
    plt.savefig('src/plots/confusion_matrix.png')
    plt.close()
    print("Confusion matrix plot saved to src/plots/confusion_matrix.png")

def plot_feature_importance(metrics):
    """Plot feature importance for the model."""
    if not metrics or 'feature_importance' not in metrics:
        print("Feature importance data not found in metrics.")
        return
    
    feature_importance = metrics['feature_importance']
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    # Sort features by importance
    sorted_idx = np.argsort(importance)
    features = [features[i] for i in sorted_idx]
    importance = [importance[i] for i in sorted_idx]
    
    plt.figure(figsize=(10, 6))
    plt.barh(features, importance, color='skyblue')
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    
    create_dir_if_not_exists('src/plots')
    plt.savefig('src/plots/feature_importance.png')
    plt.close()
    print("Feature importance plot saved to src/plots/feature_importance.png")

def plot_prediction_history(history_df):
    """Plot prediction history over time."""
    if history_df is None or history_df.empty:
        print("No prediction history available.")
        return
    
    # Convert timestamp to datetime if not already
    if 'timestamp' in history_df.columns:
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df = history_df.sort_values('timestamp')
    
    # Plot failure probability over time
    plt.figure(figsize=(12, 6))
    plt.plot(history_df['timestamp'], history_df['failure_probability'], 
             marker='o', linestyle='-', color='tab:red')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Threshold')
    plt.title('Node Failure Probability Over Time')
    plt.xlabel('Time')
    plt.ylabel('Failure Probability')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    create_dir_if_not_exists('src/plots')
    plt.savefig('src/plots/prediction_history.png')
    plt.close()
    print("Prediction history plot saved to src/plots/prediction_history.png")

def plot_metric_trends(history_df):
    """Plot trends of different metrics over time."""
    if history_df is None or history_df.empty:
        print("No prediction history available.")
        return
    
    # Convert timestamp to datetime if not already
    if 'timestamp' in history_df.columns:
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df = history_df.sort_values('timestamp')
    
    # Select metrics columns to plot
    metric_columns = ['cpu_usage_percent', 'memory_usage_percent', 
                      'network_latency_ms', 'disk_io_mbps',
                      'error_rate', 'avg_response_time_ms', 
                      'pod_restarts', 'cpu_throttling_percent']
    
    # Check which columns exist in the dataframe
    available_metrics = [col for col in metric_columns if col in history_df.columns]
    
    if not available_metrics:
        print("No metric data available in history.")
        return
    
    # Plot trends for available metrics
    fig, axes = plt.subplots(len(available_metrics), 1, figsize=(12, 4*len(available_metrics)))
    
    if len(available_metrics) == 1:
        axes = [axes]  # Make axes iterable for single metric case
    
    for i, metric in enumerate(available_metrics):
        axes[i].plot(history_df['timestamp'], history_df[metric], 
                   marker='o', linestyle='-', label=metric)
        axes[i].set_title(f'{metric} Over Time')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel(metric)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    create_dir_if_not_exists('src/plots')
    plt.savefig('src/plots/metric_trends.png')
    plt.close()
    print("Metric trends plot saved to src/plots/metric_trends.png")

def plot_metric_correlations(history_df):
    """Plot correlation matrix between different metrics."""
    if history_df is None or history_df.empty:
        print("No prediction history available.")
        return
    
    # Select numeric columns for correlation
    numeric_columns = history_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'failure_probability' in numeric_columns:
        # Ensure failure_probability is included
        numeric_columns.remove('failure_probability')
        numeric_columns = ['failure_probability'] + numeric_columns
    
    if len(numeric_columns) < 2:
        print("Not enough numeric data for correlation analysis.")
        return
    
    # Calculate correlation matrix
    corr_matrix = history_df[numeric_columns].corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
    plt.title('Correlation Between Metrics')
    plt.tight_layout()
    
    create_dir_if_not_exists('src/plots')
    plt.savefig('src/plots/metric_correlations.png')
    plt.close()
    print("Metric correlations plot saved to src/plots/metric_correlations.png")

def plot_decision_boundary(metrics, model=None):
    """Plot decision boundary for the model using 2D PCA transformation."""
    if not metrics or 'pca_data' not in metrics:
        print("PCA data not found in metrics.")
        return
    
    pca_data = metrics['pca_data']
    
    plt.figure(figsize=(10, 8))
    
    # Extract data from metrics
    X_pca = np.array(pca_data['X_pca'])
    y = np.array(pca_data['y'])
    
    # Create mesh grid
    h = 0.02  # Step size
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Plot decision boundary
    Z = np.array(pca_data['Z']).reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    
    # Plot data points
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, 
                          cmap=plt.cm.coolwarm, edgecolors='k')
    
    plt.title('Decision Boundary (PCA 2D Projection)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter, label='Class')
    plt.tight_layout()
    
    create_dir_if_not_exists('src/plots')
    plt.savefig('src/plots/decision_boundary.png')
    plt.close()
    print("Decision boundary plot saved to src/plots/decision_boundary.png")

def generate_all_plots():
    """Generate all available plots."""
    print("Generating visualization plots...")
    
    # Load metrics and prediction history
    metrics = load_metrics()
    history_df = load_prediction_history()
    
    # Generate plots based on metrics
    if metrics:
        plot_learning_curve(metrics)
        plot_roc_curve(metrics)
        plot_confusion_matrix(metrics)
        plot_feature_importance(metrics)
        plot_decision_boundary(metrics)
    
    # Generate plots based on prediction history
    if history_df is not None:
        plot_prediction_history(history_df)
        plot_metric_trends(history_df)
        plot_metric_correlations(history_df)
    
    print("Visualization complete. Plots saved to src/plots/ directory.")

if __name__ == "__main__":
    generate_all_plots() 