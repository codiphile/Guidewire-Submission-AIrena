#!/usr/bin/env python3
"""
Prediction script for SVM model for Kubernetes Node Failure Prediction.
This script loads a trained model and makes predictions based on input metrics.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from models.svm_model import SVMNodeFailurePredictor

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Make predictions using trained SVM model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model-path', type=str, default='src/models/saved',
                        help='Path to saved model directory')
    parser.add_argument('--cpu', type=float, required=False,
                        help='CPU usage percentage (0-100)')
    parser.add_argument('--memory', type=float, required=False,
                        help='Memory usage percentage (0-100)')
    parser.add_argument('--network', type=float, required=False,
                        help='Network latency in ms')
    parser.add_argument('--disk', type=float, required=False,
                        help='Disk I/O in MBps')
    parser.add_argument('--error', type=float, required=False,
                        help='Error rate (0-1)')
    parser.add_argument('--response', type=float, required=False,
                        help='Average response time in ms')
    parser.add_argument('--restarts', type=int, required=False,
                        help='Number of pod restarts')
    parser.add_argument('--throttle', type=float, required=False,
                        help='CPU throttling percentage (0-100)')
    parser.add_argument('--output-path', type=str, default=None,
                        help='Path to save prediction results (JSON format)')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode and prompt for values')
    
    return parser.parse_args()

def load_model(model_path):
    """
    Load a trained model.
    
    Args:
        model_path (str): Path to saved model directory
        
    Returns:
        SVMNodeFailurePredictor: Loaded model
    """
    try:
        print(f"Loading model from {model_path}...")
        model = SVMNodeFailurePredictor.load(model_path)
        print("Model loaded successfully")
        return model
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)

def get_interactive_input():
    """
    Get metrics interactively from user input.
    
    Returns:
        dict: Input metrics
    """
    print("\n" + "=" * 50)
    print("Kubernetes Node Metrics Input")
    print("=" * 50)
    
    try:
        # Get metric values with validation
        metrics = {}
        
        metrics['cpu_usage_percent'] = float(input("CPU Usage Percentage (0-100): "))
        metrics['memory_usage_percent'] = float(input("Memory Usage Percentage (0-100): "))
        metrics['network_latency_ms'] = float(input("Network Latency (ms): "))
        metrics['disk_io_mbps'] = float(input("Disk I/O (MBps): "))
        metrics['error_rate'] = float(input("Error Rate (0-1): "))
        metrics['avg_response_time_ms'] = float(input("Average Response Time (ms): "))
        metrics['pod_restarts'] = int(input("Pod Restarts (count): "))
        metrics['cpu_throttling_percent'] = float(input("CPU Throttling Percentage (0-100): "))
        
        # Validate ranges
        if not 0 <= metrics['cpu_usage_percent'] <= 100:
            print("Warning: CPU usage should be between 0 and 100")
        
        if not 0 <= metrics['memory_usage_percent'] <= 100:
            print("Warning: Memory usage should be between 0 and 100")
        
        if not 0 <= metrics['error_rate'] <= 1:
            print("Warning: Error rate should be between 0 and 1")
        
        if not 0 <= metrics['cpu_throttling_percent'] <= 100:
            print("Warning: CPU throttling should be between 0 and 100")
        
        return metrics
    
    except ValueError as e:
        print(f"Error in input: {str(e)}")
        print("Please enter valid numeric values.")
        return get_interactive_input()

def get_command_line_input(args):
    """
    Get metrics from command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        dict: Input metrics
    """
    # Check if all required metrics are provided
    required_metrics = ['cpu', 'memory', 'network', 'disk', 
                        'error', 'response', 'restarts', 'throttle']
    
    for metric in required_metrics:
        if getattr(args, metric) is None:
            print(f"Error: Missing required metric: {metric}")
            print("Please provide all metrics or use --interactive mode")
            sys.exit(1)
    
    # Create metrics dictionary
    metrics = {
        'cpu_usage_percent': args.cpu,
        'memory_usage_percent': args.memory,
        'network_latency_ms': args.network,
        'disk_io_mbps': args.disk,
        'error_rate': args.error,
        'avg_response_time_ms': args.response,
        'pod_restarts': args.restarts,
        'cpu_throttling_percent': args.throttle
    }
    
    return metrics

def save_prediction(metrics, prediction_result, output_path=None):
    """
    Save prediction results to file.
    
    Args:
        metrics (dict): Input metrics
        prediction_result (dict): Prediction results
        output_path (str): Path to save results
    """
    # Create prediction record
    result = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'input_metrics': metrics,
        'prediction': prediction_result
    }
    
    # If no output path specified, create a default one
    if output_path is None:
        os.makedirs('src/metrics/predictions', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'src/metrics/predictions/prediction_{timestamp}.json'
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Prediction saved to {output_path}")

def append_to_history(metrics, prediction_result):
    """
    Append prediction to history CSV file.
    
    Args:
        metrics (dict): Input metrics
        prediction_result (dict): Prediction results
    """
    # Create history directory if it doesn't exist
    os.makedirs('src/data/predictions', exist_ok=True)
    history_file = 'src/data/predictions/prediction_history.csv'
    
    # Prepare record
    record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'failure_probability': prediction_result['failure_probability'],
        'prediction': prediction_result['prediction']
    }
    record.update(metrics)
    
    # Convert to DataFrame
    record_df = pd.DataFrame([record])
    
    # Check if file exists
    file_exists = os.path.isfile(history_file)
    
    # Append to file
    record_df.to_csv(history_file, mode='a', header=not file_exists, index=False)
    
    print(f"Prediction added to history: {history_file}")

def print_prediction_results(metrics, prediction_result):
    """
    Print prediction results in a formatted way.
    
    Args:
        metrics (dict): Input metrics
        prediction_result (dict): Prediction results
    """
    print("\n" + "=" * 60)
    print("KUBERNETES NODE FAILURE PREDICTION RESULTS")
    print("=" * 60)
    
    # Print input metrics
    print("\nInput Metrics:")
    print(f"  CPU Usage:          {metrics['cpu_usage_percent']:.1f}%")
    print(f"  Memory Usage:       {metrics['memory_usage_percent']:.1f}%")
    print(f"  Network Latency:    {metrics['network_latency_ms']:.1f} ms")
    print(f"  Disk I/O:           {metrics['disk_io_mbps']:.1f} MBps")
    print(f"  Error Rate:         {metrics['error_rate']:.3f}")
    print(f"  Response Time:      {metrics['avg_response_time_ms']:.1f} ms")
    print(f"  Pod Restarts:       {metrics['pod_restarts']}")
    print(f"  CPU Throttling:     {metrics['cpu_throttling_percent']:.1f}%")
    
    # Print prediction
    print("\nPrediction:")
    
    # Determine color based on probability (green for low, yellow for medium, red for high)
    probability = prediction_result['failure_probability']
    
    if probability < 0.3:
        color_code = "\033[92m"  # Green
        risk_level = "LOW RISK"
    elif probability < 0.7:
        color_code = "\033[93m"  # Yellow
        risk_level = "MEDIUM RISK"
    else:
        color_code = "\033[91m"  # Red
        risk_level = "HIGH RISK"
        
    reset_code = "\033[0m"
    
    print(f"  Failure Probability: {color_code}{probability:.4f}{reset_code}")
    print(f"  Risk Level:          {color_code}{risk_level}{reset_code}")
    print(f"  Prediction:          {color_code}{prediction_result['prediction']}{reset_code}")
    
    # Print recommendations
    print("\nRecommendations:")
    for i, rec in enumerate(prediction_result['recommendations'], 1):
        # Determine recommendation color based on content
        if "CRITICAL" in rec:
            rec_color = "\033[91m"  # Red for critical
        elif "WARNING" in rec:
            rec_color = "\033[93m"  # Yellow for warning
        else:
            rec_color = "\033[0m"   # Default
            
        print(f"  {i}. {rec_color}{rec}{reset_code}")
    
    print("\n" + "=" * 60)

def main():
    """Main function to make predictions."""
    # Parse arguments
    args = parse_args()
    
    # Load model
    model = load_model(args.model_path)
    
    # Get input metrics
    if args.interactive:
        metrics = get_interactive_input()
    else:
        metrics = get_command_line_input(args)
    
    # Make prediction
    try:
        # Make prediction using the loaded model
        prediction, probability = model.predict(metrics)
        
        # Generate recommendations
        recommendations = model.generate_recommendations(metrics, probability)
        
        # Prepare prediction result
        prediction_result = {
            'prediction': "LIKELY TO FAIL" if prediction == 1 else "NOT LIKELY TO FAIL",
            'failure_probability': probability,
            'recommendations': recommendations,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Print results
        print_prediction_results(metrics, prediction_result)
        
        # Save prediction if output path is specified
        if args.output_path:
            save_prediction(metrics, prediction_result, args.output_path)
        
        # Append to prediction history
        append_to_history(metrics, prediction_result)
        
        return 0
    
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main()) 