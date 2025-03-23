#!/usr/bin/env python3
"""
Kubernetes Node Failure Prediction - Main Script

This is the main script for Kubernetes Node Failure Prediction, providing a
command-line interface for training models and making predictions.
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Kubernetes Node Failure Prediction System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Generate data command
    generate_parser = subparsers.add_parser('generate-data', help='Generate training data')
    generate_parser.add_argument('--num-records', type=int, default=10000,
                               help='Number of records to generate')
    generate_parser.add_argument('--failure-rate', type=float, default=0.05,
                               help='Proportion of failure records (0-1)')
    generate_parser.add_argument('--output-path', type=str, 
                               default='src/data/generated_metrics.csv',
                               help='Path to save generated data')
    generate_parser.add_argument('--skip-training', action='store_true',
                               help='Skip training after data generation')
    generate_parser.add_argument('--skip-testing', action='store_true',
                               help='Skip testing after training')
    
    # Train SVM command
    train_svm_parser = subparsers.add_parser('train-svm', help='Train SVM model')
    train_svm_parser.add_argument('--data-path', type=str, 
                               default='src/data/generated_metrics.csv',
                               help='Path to training data')
    train_svm_parser.add_argument('--test-size', type=float, default=0.2,
                               help='Proportion of data to use for testing')
    train_svm_parser.add_argument('--random-state', type=int, default=42,
                               help='Random seed for reproducibility')
    train_svm_parser.add_argument('--grid-search', action='store_true',
                               help='Perform grid search for hyperparameter tuning')
    train_svm_parser.add_argument('--visualize', action='store_true',
                               help='Generate visualizations')
    
    # Predict with SVM command
    predict_svm_parser = subparsers.add_parser('predict-svm', help='Make prediction with SVM model')
    predict_svm_parser.add_argument('--model-path', type=str, 
                                  default='src/models/saved',
                                  help='Path to trained model directory')
    predict_svm_parser.add_argument('--interactive', action='store_true',
                                  help='Interactive mode to input metrics')
    predict_svm_parser.add_argument('--cpu', type=float, 
                                  help='CPU usage percentage (0-100)')
    predict_svm_parser.add_argument('--memory', type=float, 
                                  help='Memory usage percentage (0-100)')
    predict_svm_parser.add_argument('--network', type=float, 
                                  help='Network latency in milliseconds')
    predict_svm_parser.add_argument('--disk', type=float, 
                                  help='Disk I/O in MBps')
    predict_svm_parser.add_argument('--error', type=float, 
                                  help='Error rate (0-1)')
    predict_svm_parser.add_argument('--response', type=float, 
                                  help='Average response time in milliseconds')
    predict_svm_parser.add_argument('--restarts', type=int, 
                                  help='Number of pod restarts')
    predict_svm_parser.add_argument('--throttle', type=float, 
                                  help='CPU throttling percentage (0-100)')
    predict_svm_parser.add_argument('--output-path', type=str, 
                                  default=None,
                                  help='Path to save prediction results (JSON)')
    
    # Visualize predictions command
    visualize_parser = subparsers.add_parser('visualize', help='Visualize predictions')
    visualize_parser.add_argument('--metrics-file', type=str,
                                default='src/metrics/svm_metrics.json',
                                help='Path to metrics file')
    visualize_parser.add_argument('--history-file', type=str,
                                default='src/data/predictions/prediction_history.csv',
                                help='Path to prediction history file')
    
    return parser.parse_args()

def run_generate_data(args):
    """Run the data generation script."""
    print("Generating synthetic data...")
    
    cmd = [
        'python', 'src/generate_large_dataset.py',
        '--num-records', str(args.num_records),
        '--failure-rate', str(args.failure_rate),
        '--output-path', args.output_path
    ]
    
    if args.skip_training:
        cmd.append('--skip-training')
    
    if args.skip_testing:
        cmd.append('--skip-testing')
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error generating data:")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True

def run_train_svm(args):
    """Run the SVM training script."""
    print("Training SVM model...")
    
    cmd = [
        'python', 'src/train_svm.py',
        '--data-path', args.data_path,
        '--test-size', str(args.test_size),
        '--random-state', str(args.random_state)
    ]
    
    if args.grid_search:
        cmd.append('--grid-search')
    
    if args.visualize:
        cmd.append('--visualize')
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error training model:")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True

def run_predict_svm(args):
    """Run the SVM prediction script."""
    print("Making prediction with SVM model...")
    
    cmd = [
        'python', 'src/predict_svm.py',
        '--model-path', 'src/models/saved'
    ]
    
    if args.interactive:
        cmd.append('--interactive')
    else:
        if args.cpu is not None:
            cmd.extend(['--cpu', str(args.cpu)])
        if args.memory is not None:
            cmd.extend(['--memory', str(args.memory)])
        if args.network is not None:
            cmd.extend(['--network', str(args.network)])
        if args.disk is not None:
            cmd.extend(['--disk', str(args.disk)])
        if args.error is not None:
            cmd.extend(['--error', str(args.error)])
        if args.response is not None:
            cmd.extend(['--response', str(args.response)])
        if args.restarts is not None:
            cmd.extend(['--restarts', str(args.restarts)])
        if args.throttle is not None:
            cmd.extend(['--throttle', str(args.throttle)])
    
    if args.output_path:
        cmd.extend(['--output-path', args.output_path])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error making prediction:")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True

def run_visualize(args):
    """Run the visualization script."""
    print("Generating visualizations...")
    
    # Ensure the visualization module can be imported
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from src.visualize_predictions import generate_all_plots
        generate_all_plots()
        return True
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
        return False

def main():
    """Main function to handle command-line interface."""
    args = parse_args()
    
    if args.command is None:
        print("Error: No command specified")
        print("Use --help for usage information")
        return 1
    
    # Track execution time
    start_time = datetime.now()
    
    # Execute the requested command
    success = False
    
    if args.command == 'generate-data':
        success = run_generate_data(args)
    elif args.command == 'train-svm':
        success = run_train_svm(args)
    elif args.command == 'predict-svm':
        success = run_predict_svm(args)
    elif args.command == 'visualize':
        success = run_visualize(args)
    else:
        print(f"Error: Unknown command '{args.command}'")
        return 1
    
    # Calculate execution time
    end_time = datetime.now()
    execution_time = end_time - start_time
    
    if success:
        print(f"\nCommand '{args.command}' completed successfully")
        print(f"Execution time: {execution_time.total_seconds():.2f} seconds")
        return 0
    else:
        print(f"\nCommand '{args.command}' failed")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 