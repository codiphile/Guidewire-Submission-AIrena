#!/usr/bin/env python3
"""
Kubernetes Node Monitor for Failure Prediction

This script monitors Kubernetes nodes, collects metrics, and uses the prediction
API to identify potential node failures. It can send alerts when the failure 
probability exceeds a certain threshold.
"""

import os
import sys
import json
import time
import logging
import requests
import pandas as pd
from datetime import datetime
from kubernetes import client, config
from kubernetes.client.rest import ApiException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('node-monitor')

# Configuration variables
PREDICTION_API_URL = os.environ.get('PREDICTION_API_URL', 'http://node-failure-prediction-svc/predict')
MONITORING_INTERVAL = int(os.environ.get('MONITORING_INTERVAL', 60))  # seconds
ALERT_THRESHOLD = float(os.environ.get('ALERT_THRESHOLD', 0.7))
CONFIG_PATH = '/app/kubernetes/config/config.json'

def load_config():
    """Load configuration from the ConfigMap."""
    try:
        config_path = os.environ.get('CONFIG_PATH', '/etc/node-monitor/config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Set up global variables from config
        global PREDICTION_API_URL, CHECK_INTERVAL, ALERT_THRESHOLD
        
        PREDICTION_API_URL = os.environ.get('PREDICTION_API_URL', 'http://node-failure-prediction-svc/predict')
        CHECK_INTERVAL = int(os.environ.get('CHECK_INTERVAL', config['monitoring']['check_interval_seconds']))
        ALERT_THRESHOLD = float(os.environ.get('ALERT_THRESHOLD', config['alert']['failure_probability_threshold']))
        MEDIUM_RISK_THRESHOLD = config['alert']['medium_risk_threshold']
        
        logging.info(f"Configuration loaded successfully: API={PREDICTION_API_URL}, interval={CHECK_INTERVAL}s, threshold={ALERT_THRESHOLD}")
        
        # Log the model performance metrics
        logging.info(f"Model performance metrics: Precision={config['model_performance']['precision']*100:.1f}%, " 
                    f"Recall={config['model_performance']['recall']*100:.1f}%, "
                    f"F1={config['model_performance']['f1_score']*100:.1f}%")
        
        # Export model metrics to Prometheus if available
        try:
            from prometheus_client import Gauge
            precision_gauge = Gauge('k8s_node_prediction_precision', 'Model precision metric')
            recall_gauge = Gauge('k8s_node_prediction_recall', 'Model recall metric')
            f1_gauge = Gauge('k8s_node_prediction_f1', 'Model F1 score metric')
            
            precision_gauge.set(config['model_performance']['precision'])
            recall_gauge.set(config['model_performance']['recall'])
            f1_gauge.set(config['model_performance']['f1_score'])
            logging.info("Prometheus metrics for model performance exported")
        except (ImportError, Exception) as e:
            logging.warning(f"Could not export model metrics to Prometheus: {str(e)}")
        
        return config
    except Exception as e:
        logging.error(f"Error loading configuration: {str(e)}")
        # Use default configuration
        return {
            'thresholds': {
                'cpu_percent': 80,
                'memory_percent': 85,
                'network_latency_ms': 40,
                'disk_io_mbps': 75,
                'error_rate': 0.3,
                'response_time_ms': 100,
                'pod_restarts': 3,
                'cpu_throttling': 60
            },
            'model_performance': {
                'precision': 0.89,
                'recall': 0.88,
                'f1_score': 0.89,
                'accuracy': 0.96
            },
            'alert': {
                'failure_probability_threshold': 0.65,
                'medium_risk_threshold': 0.30
            },
            'monitoring': {
                'check_interval_seconds': 300
            }
        }

def setup_kubernetes_client():
    """Set up the Kubernetes client based on environment."""
    try:
        # Try to load in-cluster config first
        config.load_incluster_config()
        logger.info("Using in-cluster configuration")
    except config.ConfigException:
        # Fall back to kubeconfig
        config.load_kube_config()
        logger.info("Using kubeconfig configuration")
    
    return client.CoreV1Api(), client.CustomObjectsApi()

def get_node_metrics(custom_api, node_name):
    """Get metrics for a specific node using the metrics API."""
    try:
        metrics = custom_api.get_cluster_custom_object(
            "metrics.k8s.io", "v1beta1", "nodes", node_name
        )
        
        cpu_usage = metrics.get('usage', {}).get('cpu', '0')
        cpu_usage = int(cpu_usage.rstrip('n')) / 1000000000  # Convert to cores
        
        memory_usage = metrics.get('usage', {}).get('memory', '0')
        memory_usage = int(memory_usage.rstrip('Ki')) / 1024  # Convert to MB
        
        return {
            'cpu_cores': cpu_usage,
            'memory_mb': memory_usage
        }
    except ApiException as e:
        logger.error(f"Error getting metrics for node {node_name}: {str(e)}")
        return {'cpu_cores': 0, 'memory_mb': 0}

def get_node_capacity(core_api, node_name):
    """Get node capacity information."""
    try:
        node = core_api.read_node(name=node_name)
        
        cpu_capacity = node.status.capacity.get('cpu', '0')
        cpu_capacity = int(cpu_capacity)
        
        memory_capacity = node.status.capacity.get('memory', '0')
        memory_capacity = int(memory_capacity.rstrip('Ki')) / 1024  # Convert to MB
        
        return {
            'cpu_cores': cpu_capacity,
            'memory_mb': memory_capacity
        }
    except ApiException as e:
        logger.error(f"Error getting capacity for node {node_name}: {str(e)}")
        return {'cpu_cores': 1, 'memory_mb': 1024}

def get_pods_on_node(core_api, node_name):
    """Get all pods running on a specific node."""
    try:
        pod_list = core_api.list_pod_for_all_namespaces(field_selector=f'spec.nodeName={node_name}')
        return pod_list.items
    except ApiException as e:
        logger.error(f"Error getting pods for node {node_name}: {str(e)}")
        return []

def calculate_pod_metrics(pods):
    """Calculate pod-based metrics."""
    total_restarts = 0
    error_pods = 0
    total_pods = len(pods)
    
    for pod in pods:
        # Count container restarts
        for container_status in pod.status.container_statuses or []:
            total_restarts += container_status.restart_count
            
            # Check for errors in container statuses
            if (container_status.ready is False and 
                container_status.state and 
                hasattr(container_status.state, 'terminated') and
                container_status.state.terminated and
                container_status.state.terminated.exit_code != 0):
                error_pods += 1
    
    # Calculate error rate
    error_rate = error_pods / max(total_pods, 1)
    
    return {
        'pod_restarts': total_restarts,
        'error_rate': error_rate,
        'total_pods': total_pods
    }

def collect_node_metrics(core_api, custom_api, node_name):
    """Collect all metrics for a specific node."""
    # Get node capacity
    capacity = get_node_capacity(core_api, node_name)
    
    # Get current node metrics
    metrics = get_node_metrics(custom_api, node_name)
    
    # Calculate usage percentages
    cpu_usage_percent = (metrics['cpu_cores'] / max(capacity['cpu_cores'], 1)) * 100
    memory_usage_percent = (metrics['memory_mb'] / max(capacity['memory_mb'], 1)) * 100
    
    # Get pods on this node
    pods = get_pods_on_node(core_api, node_name)
    pod_metrics = calculate_pod_metrics(pods)
    
    # Estimate other metrics (these would ideally come from proper monitoring systems)
    # For a real system, you'd integrate with Prometheus or other monitoring tools
    network_latency_ms = 10 + (cpu_usage_percent / 10)  # Simplified estimate
    disk_io_mbps = 30 + (memory_usage_percent / 5)  # Simplified estimate
    avg_response_time_ms = 20 + (cpu_usage_percent / 5)  # Simplified estimate
    cpu_throttling_percent = max(0, cpu_usage_percent - 80) * 2 if cpu_usage_percent > 80 else 0
    
    # Combine all metrics
    combined_metrics = {
        'cpu_usage_percent': cpu_usage_percent,
        'memory_usage_percent': memory_usage_percent,
        'network_latency_ms': network_latency_ms,
        'disk_io_mbps': disk_io_mbps,
        'error_rate': pod_metrics['error_rate'],
        'avg_response_time_ms': avg_response_time_ms,
        'pod_restarts': pod_metrics['pod_restarts'],
        'cpu_throttling_percent': cpu_throttling_percent,
        'node_name': node_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_pods': pod_metrics['total_pods']
    }
    
    return combined_metrics

def predict_node_failure(metrics):
    """Send metrics to prediction API and get failure probability."""
    try:
        # Prepare request data
        payload = {
            'cpu_usage_percent': metrics['cpu_usage_percent'],
            'memory_usage_percent': metrics['memory_usage_percent'],
            'network_latency_ms': metrics['network_latency_ms'],
            'disk_io_mbps': metrics['disk_io_mbps'],
            'error_rate': metrics['error_rate'],
            'avg_response_time_ms': metrics['avg_response_time_ms'],
            'pod_restarts': metrics['pod_restarts'],
            'cpu_throttling_percent': metrics['cpu_throttling_percent']
        }
        
        # Send request to prediction API
        response = requests.post(
            PREDICTION_API_URL,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                'failure_probability': result.get('failure_probability', 0),
                'prediction_label': result.get('prediction', 'UNKNOWN'),
                'recommendations': result.get('recommendations', [])
            }
        else:
            logger.error(f"API request failed with status {response.status_code}: {response.text}")
            return {'failure_probability': 0, 'prediction_label': 'ERROR', 'recommendations': []}
            
    except Exception as e:
        logger.error(f"Error making prediction request: {str(e)}")
        return {'failure_probability': 0, 'prediction_label': 'ERROR', 'recommendations': []}

def send_alert(node_name, metrics, prediction, config):
    """Send an alert based on prediction result."""
    alert_channels = config.get('alert_channels', [])
    
    # Format alert message
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    alert_message = f"""
    ⚠️ *Kubernetes Node Failure Alert* ⚠️
    
    *Node:* {node_name}
    *Time:* {timestamp}
    *Failure Probability:* {prediction['failure_probability']:.2%}
    *Status:* {prediction['prediction_label']}
    
    *Current Metrics:*
    - CPU Usage: {metrics['cpu_usage_percent']:.1f}%
    - Memory Usage: {metrics['memory_usage_percent']:.1f}%
    - Network Latency: {metrics['network_latency_ms']:.1f} ms
    - Disk I/O: {metrics['disk_io_mbps']:.1f} MBps
    - Error Rate: {metrics['error_rate']:.3f}
    - Response Time: {metrics['avg_response_time_ms']:.1f} ms
    - Pod Restarts: {metrics['pod_restarts']}
    - CPU Throttling: {metrics['cpu_throttling_percent']:.1f}%
    
    *Recommendations:*
    """
    
    # Add recommendations
    for i, rec in enumerate(prediction['recommendations'], 1):
        alert_message += f"{i}. {rec}\n"
    
    logger.info(f"ALERT: Node {node_name} has failure probability of {prediction['failure_probability']:.2%}")
    
    # Send to each configured channel
    for channel in alert_channels:
        channel_type = channel.get('type', '').lower()
        
        if channel_type == 'slack':
            try:
                webhook_url = channel.get('webhook_url')
                if webhook_url:
                    requests.post(
                        webhook_url,
                        json={'text': alert_message},
                        timeout=10
                    )
                    logger.info(f"Slack alert sent for node {node_name}")
            except Exception as e:
                logger.error(f"Error sending Slack alert: {str(e)}")
        
        elif channel_type == 'pagerduty':
            # Implement PagerDuty integration
            pass
        
        elif channel_type == 'email':
            # Implement email integration
            pass

def save_metrics_history(metrics, prediction):
    """Save metrics and prediction to history file."""
    history_file = '/app/kubernetes/data/node_metrics_history.csv'
    
    # Combine metrics and prediction
    data = {**metrics, **{
        'failure_probability': prediction['failure_probability'],
        'prediction_label': prediction['prediction_label']
    }}
    
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Append to CSV
    try:
        df.to_csv(history_file, mode='a', header=not os.path.exists(history_file), index=False)
    except Exception as e:
        logger.error(f"Error saving metrics history: {str(e)}")

def monitor_nodes():
    """Main function to monitor all nodes."""
    # Load configuration
    config = load_config()
    settings = config.get('monitoring_settings', {})
    
    # Override with environment variables if provided
    interval = int(os.environ.get('MONITORING_INTERVAL', settings.get('interval_seconds', MONITORING_INTERVAL)))
    threshold = float(os.environ.get('ALERT_THRESHOLD', settings.get('prediction_threshold', ALERT_THRESHOLD)))
    
    # Set up Kubernetes clients
    core_api, custom_api = setup_kubernetes_client()
    
    logger.info(f"Starting node monitoring (interval: {interval}s, threshold: {threshold})")
    
    while True:
        try:
            # Get all nodes
            nodes = core_api.list_node().items
            logger.info(f"Monitoring {len(nodes)} nodes")
            
            for node in nodes:
                node_name = node.metadata.name
                
                # Skip nodes that are not Ready
                node_ready = True
                for condition in node.status.conditions:
                    if condition.type == 'Ready' and condition.status != 'True':
                        node_ready = False
                        break
                
                if not node_ready:
                    logger.info(f"Skipping node {node_name} as it's not in Ready state")
                    continue
                
                # Collect metrics
                logger.info(f"Collecting metrics for node {node_name}")
                metrics = collect_node_metrics(core_api, custom_api, node_name)
                
                # Make prediction
                prediction = predict_node_failure(metrics)
                
                # Log prediction
                logger.info(f"Node {node_name} - Failure probability: {prediction['failure_probability']:.2%}")
                
                # Save metrics to history
                save_metrics_history(metrics, prediction)
                
                # Send alert if probability exceeds threshold
                if prediction['failure_probability'] >= threshold:
                    send_alert(node_name, metrics, prediction, config)
            
            # Sleep until next monitoring interval
            time.sleep(interval)
            
        except ApiException as e:
            logger.error(f"Kubernetes API error: {str(e)}")
            time.sleep(interval)
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            time.sleep(interval)

if __name__ == '__main__':
    try:
        # Ensure data directory exists
        os.makedirs('/app/kubernetes/data', exist_ok=True)
        
        # Start monitoring
        monitor_nodes()
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1) 