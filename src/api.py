#!/usr/bin/env python3
"""
Kubernetes Node Failure Prediction API

Provides a REST API for predicting Kubernetes node failures based on system metrics.
"""

import os
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from prometheus_client import Counter, Gauge, Histogram, generate_latest, REGISTRY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('prediction-api')

# Initialize Flask app
app = Flask(__name__)

# Define Prometheus metrics
PREDICTION_REQUESTS = Counter(
    'node_failure_prediction_requests_total', 
    'Total number of prediction requests',
    ['result']
)
PREDICTION_LATENCY = Histogram(
    'node_failure_prediction_latency_seconds', 
    'Prediction request latency in seconds'
)
NODE_FAILURE_PROBABILITY = Gauge(
    'node_failure_probability', 
    'Probability of node failure',
    ['node']
)
FAILURE_ALERTS = Counter(
    'node_failure_alerts_total', 
    'Total number of failure alerts generated',
    ['node']
)
CPU_USAGE = Gauge(
    'node_cpu_usage_percent', 
    'CPU usage percentage',
    ['node']
)
MEMORY_USAGE = Gauge(
    'node_memory_usage_percent', 
    'Memory usage percentage',
    ['node']
)

# Path to saved model
MODEL_PATH = os.environ.get('MODEL_PATH', 'src/models/saved/svm_model.pkl')

# Load the trained model
model = None
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Loaded model from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")

# Input metrics and their valid ranges
REQUIRED_METRICS = {
    'cpu_usage_percent': (0, 100),
    'memory_usage_percent': (0, 100),
    'network_latency_ms': (0, 1000),
    'disk_io_mbps': (0, 500),
    'error_rate': (0, 1),
    'avg_response_time_ms': (0, 10000),
    'pod_restarts': (0, 100),
    'cpu_throttling_percent': (0, 100)
}

def save_prediction_data(data, prediction, probability):
    """Save prediction data to a file for later analysis"""
    timestamp = datetime.now().isoformat()
    data_with_prediction = {
        **data,
        'prediction': prediction,
        'failure_probability': probability,
        'timestamp': timestamp
    }
    
    try:
        # Create directory if it doesn't exist
        os.makedirs('src/data/predictions', exist_ok=True)
        
        # Append to CSV
        df = pd.DataFrame([data_with_prediction])
        file_path = 'src/data/predictions/prediction_history.csv'
        
        if os.path.exists(file_path):
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(file_path, index=False)
            
        logger.info(f"Saved prediction data to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save prediction data: {e}")

def generate_recommendations(metrics, probability):
    """Generate recommendations based on prediction results"""
    recommendations = []
    
    if probability >= 0.7:
        # Critical recommendations for high probability
        if metrics.get('cpu_usage_percent', 0) > 80:
            recommendations.append("CRITICAL: Reduce CPU usage below 80% immediately")
        if metrics.get('memory_usage_percent', 0) > 85:
            recommendations.append("CRITICAL: Free up memory resources immediately")
        if metrics.get('error_rate', 0) > 0.3:
            recommendations.append("CRITICAL: Investigate and fix high error rate")
        if metrics.get('pod_restarts', 0) > 5:
            recommendations.append("CRITICAL: Check pod stability, too many restarts")
        if metrics.get('cpu_throttling_percent', 0) > 50:
            recommendations.append("CRITICAL: Reduce CPU throttling by increasing resource limits")
    elif probability >= 0.4:
        # Warning recommendations for medium probability
        if metrics.get('cpu_usage_percent', 0) > 70:
            recommendations.append("WARNING: CPU usage is high, consider scaling")
        if metrics.get('memory_usage_percent', 0) > 75:
            recommendations.append("WARNING: Memory usage is high, monitor closely")
        if metrics.get('network_latency_ms', 0) > 25:
            recommendations.append("WARNING: Network latency is increasing")
        if metrics.get('disk_io_mbps', 0) > 300:
            recommendations.append("WARNING: Disk I/O is high, check for bottlenecks")
    else:
        # General recommendations for low probability
        if metrics.get('cpu_usage_percent', 0) > 60:
            recommendations.append("Consider optimizing workloads to reduce CPU usage")
        if metrics.get('memory_usage_percent', 0) > 65:
            recommendations.append("Monitor memory usage trends")
    
    # If no specific recommendations, provide a general one
    if not recommendations:
        if probability >= 0.4:
            recommendations.append("Monitor system metrics closely for changes")
        else:
            recommendations.append("System appears to be operating normally")
    
    return recommendations

@app.route('/metrics')
def metrics():
    """Expose Prometheus metrics"""
    return generate_latest(REGISTRY)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    with PREDICTION_LATENCY.time():
        try:
            # Get JSON data from request
            data = request.json
            
            if not data:
                return jsonify({
                    'error': 'No data provided',
                    'status': 'error'
                }), 400
            
            # Log the received data
            logger.info(f"Received prediction request: {data}")
            
            # Get node name if provided
            node_name = data.pop('node_name', 'unknown')
            
            # Validate input metrics
            for metric, (min_val, max_val) in REQUIRED_METRICS.items():
                if metric not in data:
                    return jsonify({
                        'error': f"Missing required metric: {metric}",
                        'status': 'error'
                    }), 400
                
                try:
                    value = float(data[metric])
                    if value < min_val or value > max_val:
                        return jsonify({
                            'error': f"Invalid value for {metric}: {value}, must be between {min_val} and {max_val}",
                            'status': 'error'
                        }), 400
                except ValueError:
                    return jsonify({
                        'error': f"Invalid value for {metric}: {data[metric]}, must be a number",
                        'status': 'error'
                    }), 400
            
            # Export current metrics to Prometheus
            CPU_USAGE.labels(node=node_name).set(data['cpu_usage_percent'])
            MEMORY_USAGE.labels(node=node_name).set(data['memory_usage_percent'])
            
            # Check if model is loaded
            if model is None:
                return jsonify({
                    'error': 'Model not loaded',
                    'status': 'error'
                }), 503
            
            # Prepare features for prediction
            features = np.array([
                [
                    data['cpu_usage_percent'],
                    data['memory_usage_percent'],
                    data['network_latency_ms'],
                    data['disk_io_mbps'],
                    data['error_rate'],
                    data['avg_response_time_ms'],
                    data['pod_restarts'],
                    data['cpu_throttling_percent']
                ]
            ])
            
            # Make prediction
            probability = model.predict_proba(features)[0][1]
            prediction = "LIKELY TO FAIL" if probability >= 0.5 else "NOT LIKELY TO FAIL"
            
            # Export prediction to Prometheus
            NODE_FAILURE_PROBABILITY.labels(node=node_name).set(probability)
            PREDICTION_REQUESTS.labels(result=prediction.lower().replace(' ', '_')).inc()
            
            # Generate recommendations
            recommendations = generate_recommendations(data, probability)
            
            # If high probability, increment alert counter
            if probability >= 0.7:
                FAILURE_ALERTS.labels(node=node_name).inc()
            
            # Save prediction data for later analysis
            save_prediction_data(data, prediction, probability)
            
            # Return prediction results
            result = {
                'prediction': prediction,
                'failure_probability': round(float(probability), 4),
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Prediction result: {result}")
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return jsonify({
                'error': str(e),
                'status': 'error'
            }), 500

@app.route('/')
def index():
    """API root endpoint"""
    return jsonify({
        'name': 'Kubernetes Node Failure Prediction API',
        'endpoints': {
            '/predict': 'POST - Make a failure prediction',
            '/health': 'GET - Check API health',
            '/metrics': 'GET - Prometheus metrics'
        }
    })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000) 