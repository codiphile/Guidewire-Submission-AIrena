#!/usr/bin/env python3
"""
Script to generate a large synthetic dataset for Kubernetes node failure prediction
and train SVM model on the generated dataset.

This script will:
1. Generate a dataset with configurable number of records
2. Inject failures based on realistic patterns
3. Train the SVM model on this dataset
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import time
import argparse
import json
import subprocess

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def generate_node_metrics(failure_probability=0.05, num_records=10000):
    """
    Generate a dataset of node metrics with realistic failure patterns.
    
    Parameters:
    - failure_probability: Probability of a node failure
    - num_records: Number of records to generate
    
    Returns:
    - df: DataFrame containing generated metrics
    """
    # Lists to store generated data
    data = []
    
    # Generate timestamps
    start_time = datetime.now() - timedelta(days=30)
    timestamps = [start_time + timedelta(minutes=i*15) for i in range(num_records)]
    
    # Node IDs (simulate a cluster with multiple nodes)
    node_ids = [f"node-{i}" for i in range(1, 11)]
    
    # Define metric distributions for failure vs normal cases
    # Allow some overlap to create more realistic ambiguity
    failure_distributions = {
        'cpu_usage_percent': {'mean': 85, 'std': 8, 'min': 65, 'max': 100},
        'memory_usage_percent': {'mean': 82, 'std': 10, 'min': 60, 'max': 100},
        'network_latency_ms': {'mean': 75, 'std': 20, 'min': 30, 'max': 200},
        'disk_io_mbps': {'mean': 78, 'std': 12, 'min': 50, 'max': 100},
        'error_rate': {'mean': 0.4, 'std': 0.2, 'min': 0.1, 'max': 1.0},
        'avg_response_time_ms': {'mean': 200, 'std': 90, 'min': 50, 'max': 500},
        'pod_restarts': {'mean': 4, 'std': 2, 'min': 1, 'max': 10},
        'cpu_throttling_percent': {'mean': 70, 'std': 15, 'min': 40, 'max': 100}
    }
    
    normal_distributions = {
        'cpu_usage_percent': {'mean': 45, 'std': 20, 'min': 5, 'max': 85},
        'memory_usage_percent': {'mean': 50, 'std': 18, 'min': 10, 'max': 88},
        'network_latency_ms': {'mean': 25, 'std': 15, 'min': 2, 'max': 75},
        'disk_io_mbps': {'mean': 35, 'std': 18, 'min': 5, 'max': 80},
        'error_rate': {'mean': 0.15, 'std': 0.12, 'min': 0, 'max': 0.45},
        'avg_response_time_ms': {'mean': 80, 'std': 40, 'min': 10, 'max': 220},
        'pod_restarts': {'mean': 1, 'std': 1.2, 'min': 0, 'max': 5},
        'cpu_throttling_percent': {'mean': 30, 'std': 20, 'min': 0, 'max': 75}
    }
    
    # Function to generate a random value with boundaries
    def generate_bounded_value(mean, std, min_val, max_val):
        value = np.random.normal(mean, std)
        return max(min_val, min(max_val, value))
    
    # Create a small subset (2%) of ambiguous/mislabeled data to simulate real-world noise
    mislabel_probability = 0.02
    
    for i in range(num_records):
        # Randomly select a node
        node_id = random.choice(node_ids)
        
        # Determine if this record will be a failure
        is_failure = random.random() < failure_probability
        
        # Determine if this should be a deliberately ambiguous case
        is_ambiguous = random.random() < 0.15  # 15% ambiguous cases
        
        # Determine if this should be mislabeled (to introduce errors)
        is_mislabeled = random.random() < mislabel_probability
        
        # Apply higher noise factor for more realism
        noise_factor = random.uniform(0.1, 0.25)
        
        if is_failure:
            # Generate failure metrics
            metrics = {}
            
            # If ambiguous, mix some normal values in
            if is_ambiguous:
                # Select 2-4 metrics to be in the normal range
                num_normal_metrics = random.randint(2, 4)
                normal_metrics = random.sample(list(failure_distributions.keys()), num_normal_metrics)
                
                # For each metric, decide which distribution to use
                for metric in failure_distributions.keys():
                    if metric in normal_metrics:
                        dist = normal_distributions[metric]
                    else:
                        dist = failure_distributions[metric]
                    
                    # Generate and store value
                    if metric == 'pod_restarts':
                        value = round(generate_bounded_value(dist['mean'], dist['std'], dist['min'], dist['max']))
                    else:
                        value = generate_bounded_value(dist['mean'], dist['std'], dist['min'], dist['max'])
                    
                    # Apply noise
                    if random.random() < 0.3:  # 30% chance of adding noise
                        if metric in normal_metrics:
                            # For normal metrics in a failure case, add noise toward failure values
                            value = value * (1 + noise_factor)
                        else:
                            # For failure metrics, add noise toward normal values
                            value = value * (1 - noise_factor * 0.5)
                    
                    metrics[metric] = value
            else:
                # Regular failure case
                for metric, dist in failure_distributions.items():
                    if metric == 'pod_restarts':
                        value = round(generate_bounded_value(dist['mean'], dist['std'], dist['min'], dist['max']))
                    else:
                        value = generate_bounded_value(dist['mean'], dist['std'], dist['min'], dist['max'])
                    
                    # Apply noise to 25% of metrics
                    if random.random() < 0.25:
                        value = value * (1 - noise_factor * 0.5)
                    
                    metrics[metric] = value
            
            # If mislabeled, we'll keep the failure metrics but mark it as healthy (0)
            failure_label = 0 if is_mislabeled else 1
            
        else:
            # Generate normal/healthy metrics
            metrics = {}
            
            # If ambiguous, mix some failure values in
            if is_ambiguous:
                # Select 1-3 metrics to be in the failure range
                num_failure_metrics = random.randint(1, 3)
                failure_metrics = random.sample(list(normal_distributions.keys()), num_failure_metrics)
                
                # For each metric, decide which distribution to use
                for metric in normal_distributions.keys():
                    if metric in failure_metrics:
                        dist = failure_distributions[metric]
                    else:
                        dist = normal_distributions[metric]
                    
                    # Generate and store value
                    if metric == 'pod_restarts':
                        value = round(generate_bounded_value(dist['mean'], dist['std'], dist['min'], dist['max']))
                    else:
                        value = generate_bounded_value(dist['mean'], dist['std'], dist['min'], dist['max'])
                    
                    # Apply noise
                    if random.random() < 0.3:  # 30% chance of adding noise
                        if metric in failure_metrics:
                            # For failure metrics in a normal case, add noise toward normal values
                            value = value * (1 - noise_factor)
                        else:
                            # For normal metrics, add noise toward failure values
                            value = value * (1 + noise_factor * 0.5)
                    
                    metrics[metric] = value
            else:
                # Regular normal case
                for metric, dist in normal_distributions.items():
                    if metric == 'pod_restarts':
                        value = round(generate_bounded_value(dist['mean'], dist['std'], dist['min'], dist['max']))
                    else:
                        value = generate_bounded_value(dist['mean'], dist['std'], dist['min'], dist['max'])
                    
                    # Apply noise to 25% of metrics
                    if random.random() < 0.25:
                        value = value * (1 + noise_factor * 0.5)
                    
                    metrics[metric] = value
            
            # If mislabeled, we'll keep the normal metrics but mark it as failure (1)
            failure_label = 1 if is_mislabeled else 0
            
        # Create a record
        record = {
            'timestamp': timestamps[i],
            'node_id': node_id,
            'cpu_usage_percent': round(metrics['cpu_usage_percent'], 2),
            'memory_usage_percent': round(metrics['memory_usage_percent'], 2),
            'network_latency_ms': round(metrics['network_latency_ms'], 2),
            'disk_io_mbps': round(metrics['disk_io_mbps'], 2),
            'error_rate': round(metrics['error_rate'], 4),
            'avg_response_time_ms': round(metrics['avg_response_time_ms'], 2),
            'pod_restarts': metrics['pod_restarts'],
            'cpu_throttling_percent': round(metrics['cpu_throttling_percent'], 2),
            'failure': failure_label
        }
        
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Print summary statistics
    failures = df[df['failure'] == 1]
    non_failures = df[df['failure'] == 0]
    
    print(f"Generated dataset with {len(df)} records")
    print(f"Failure rate: {len(failures) / len(df):.2%} ({len(failures)} failures)")
    
    # Print average values for key metrics to verify separation
    print("\nAverage values for failure cases:")
    for col in ['cpu_usage_percent', 'memory_usage_percent', 'error_rate']:
        print(f"  {col}: {failures[col].mean():.2f}")
    
    print("\nAverage values for normal cases:")
    for col in ['cpu_usage_percent', 'memory_usage_percent', 'error_rate']:
        print(f"  {col}: {non_failures[col].mean():.2f}")
    
    return df

def save_dataset(df, output_path):
    """
    Save the generated dataset to a CSV file.
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    print(f"Saving dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Dataset saved successfully: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

def train_model(data_path, train_subset=None, grid_search=False, visualize=False):
    """
    Train the SVM model on the generated dataset.
    
    Parameters:
    - data_path: Path to the dataset CSV
    - train_subset: If specified, use only this many rows for training
    - grid_search: Whether to use grid search for hyperparameter optimization
    - visualize: Whether to generate visualization
    """
    # Import here to avoid circular imports
    import subprocess
    
    print("===== TRAINING SVM MODEL ON GENERATED DATASET =====")
    
    # If we're using a subset, create a temporary subset file
    subset_path = data_path
    if train_subset is not None and train_subset > 0:
        print(f"Creating training subset with {train_subset} rows...")
        df = pd.read_csv(data_path)
        if train_subset < len(df):
            # Maintain the same failure ratio in the subset
            failure_rows = df[df['failure'] == 1]
            non_failure_rows = df[df['failure'] == 0]
            
            failure_ratio = len(failure_rows) / len(df)
            subset_failure_count = int(train_subset * failure_ratio)
            subset_non_failure_count = train_subset - subset_failure_count
            
            subset_failures = failure_rows.sample(subset_failure_count, random_state=42)
            subset_non_failures = non_failure_rows.sample(subset_non_failure_count, random_state=42)
            
            subset_df = pd.concat([subset_failures, subset_non_failures])
            subset_df = subset_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            subset_path = data_path.replace('.csv', f'_subset_{train_subset}.csv')
            subset_df.to_csv(subset_path, index=False)
            print(f"Training subset created with {len(subset_df)} rows ({subset_df['failure'].sum()} failures)")
        else:
            print(f"Requested subset size ({train_subset}) >= dataset size ({len(df)}). Using full dataset.")
    
    # Train SVM model
    print("\n----- Training SVM Model -----")
    try:
        cmd = [sys.executable, "main.py", "train-svm", "--data-path", subset_path]
        
        if grid_search:
            cmd.append("--grid-search")
        
        if visualize:
            cmd.append("--visualize")
            
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        print(f"SVM model training completed with return code: {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Error training SVM model: {e}")
    
    print("\n===== MODEL TRAINING COMPLETE =====")

def test_predictions():
    """
    Test predictions with the trained SVM model on various scenarios.
    """
    print("\n===== TESTING MODEL PREDICTIONS =====")
    
    # Test with a clear failure case
    print("\n----- Testing with Failure Case -----")
    try:
        cmd = [
            sys.executable, "main.py", "predict-svm",
            "--cpu", "95.0",
            "--memory", "98.0",
            "--network", "40.0",
            "--disk", "85.0",
            "--error", "0.65",
            "--response", "50.0",
            "--restarts", "6",
            "--throttle", "75.0"
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error testing failure case: {e}")
    
    # Test with a borderline case
    print("\n----- Testing with Borderline Case -----")
    try:
        cmd = [
            sys.executable, "main.py", "predict-svm",
            "--cpu", "82.0",
            "--memory", "88.0",
            "--network", "22.0",
            "--disk", "70.0",
            "--error", "0.28",
            "--response", "30.0",
            "--restarts", "3",
            "--throttle", "35.0"
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error testing borderline case: {e}")
    
    # Test with a normal case
    print("\n----- Testing with Normal Case -----")
    try:
        cmd = [
            sys.executable, "main.py", "predict-svm",
            "--cpu", "50.0",
            "--memory", "60.0",
            "--network", "10.0",
            "--disk", "40.0",
            "--error", "0.05",
            "--response", "15.0",
            "--restarts", "1",
            "--throttle", "15.0"
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error testing normal case: {e}")
    
    print("\n===== PREDICTION TESTING COMPLETE =====")

def main():
    """
    Main function to generate data and train models.
    """
    parser = argparse.ArgumentParser(description='Generate a large dataset and train SVM model')
    parser.add_argument('--num-records', type=int, default=100000, 
                      help='Number of records to generate')
    parser.add_argument('--failure-rate', type=float, default=0.05,
                      help='Proportion of failure records (0.0-1.0)')
    parser.add_argument('--output-path', type=str, default='src/data/generated_metrics.csv',
                      help='Output path for the generated dataset')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--train-subset', type=int, default=0,
                      help='Train on a subset of this many rows (0 to use all)')
    parser.add_argument('--skip-generation', action='store_true',
                      help='Skip dataset generation, use existing file')
    parser.add_argument('--skip-training', action='store_true',
                      help='Skip model training')
    parser.add_argument('--skip-testing', action='store_true',
                      help='Skip prediction testing')
    parser.add_argument('--grid-search', action='store_true',
                      help='Use grid search for hyperparameter optimization')
    parser.add_argument('--visualize', action='store_true',
                      help='Generate and save visualization of results')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Generate dataset
    if not args.skip_generation:
        df = generate_node_metrics(args.failure_rate, args.num_records)
        save_dataset(df, args.output_path)
    else:
        print(f"Skipping dataset generation, using existing file: {args.output_path}")
    
    # Train model
    if not args.skip_training:
        train_model(
            args.output_path, 
            train_subset=args.train_subset if args.train_subset > 0 else None,
            grid_search=args.grid_search,
            visualize=args.visualize
        )
    else:
        print("Skipping model training")
    
    # Test predictions
    if not args.skip_testing:
        test_predictions()
    else:
        print("Skipping prediction testing")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

if __name__ == "__main__":
    main() 