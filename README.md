# Kubernetes Node Failure Prediction

A machine learning system that predicts Kubernetes node failures based on system metrics using Support Vector Machine (SVM). This system helps detect potential node failures before they occur, allowing for proactive interventions to prevent service disruptions and downtime in Kubernetes clusters.

## Problem Statement

Kubernetes clusters can experience node failures due to various factors such as resource exhaustion, hardware issues, or software bugs. Predicting these failures in advance can help prevent service disruptions and improve cluster reliability.

This project uses Support Vector Machine (SVM) to build a model that can predict the probability of node failure based on various system metrics. By continuously monitoring node health metrics, the system provides early warnings and actionable recommendations to prevent potential failures.

## Key Features

- **Advanced Prediction System**: Uses SVM to predict node failures with high accuracy
- **Real-time Monitoring**: Continuously monitors cluster nodes for potential issues
- **Early Warning System**: Provides proactive alerts before failures occur
- **Actionable Recommendations**: Suggests specific actions to prevent failures
- **Seamless Kubernetes Integration**: Deploys easily into any Kubernetes cluster
- **Prometheus Metrics Integration**: Exposes metrics for monitoring and alerting
- **Interactive Dashboard**: Visualizes node health and failure probabilities
- **REST API**: Enables integration with existing monitoring solutions
- **Comprehensive Visualization**: Includes tools for understanding model performance

## Project Structure

```
.
├── main.py                     # Main entry point for the application
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration for containerization
├── kubernetes/                 # Kubernetes deployment files
│   ├── model_manifest/         # Model deployment manifests
│   │   ├── deployment.yaml     # Original test deployment
│   │   └── api_deployment.yaml # API service deployment
│   ├── services/               # Service configurations
│   │   ├── node_monitor.py     # Node monitoring script
│   │   ├── node_monitor_deployment.yaml # Monitor deployment
│   │   └── dashboard.yaml      # Grafana dashboard config
│   └── test_app_manifests/     # Test application manifests
├── src/                        # Source code directory
│   ├── data/                   # Data directory
│   │   ├── generated_metrics.csv  # Generated dataset
│   │   ├── predictions/        # Prediction history data
│   │   │   └── prediction_history.csv # History of predictions
│   │   └── archive/            # Archived data files
│   ├── metrics/                # Metrics and evaluation results
│   │   ├── svm_metrics.json    # SVM model metrics
│   │   └── predictions/        # Individual prediction results
│   ├── models/                 # Model definitions
│   │   ├── __init__.py
│   │   ├── svm_model.py        # SVM model implementation
│   │   └── saved/              # Saved model files
│   ├── plots/                  # Visualization outputs
│   │   ├── confusion_matrix.png # Confusion matrix visualization
│   │   ├── prediction_history.png # Historical predictions
│   │   ├── metric_trends.png   # Metric trends over time
│   │   ├── metric_correlations.png # Correlation between metrics
│   │   └── archive/            # Archived visualization files
│   ├── __init__.py
│   ├── api.py                  # Flask API for model serving
│   ├── train_svm.py            # SVM training script
│   ├── predict_svm.py          # SVM prediction script
│   ├── visualize_predictions.py # Visualization script
│   └── generate_large_dataset.py # Dataset generation script
```

## Quick Start Guide

### Local Development Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd kubernetes-node-failure-prediction
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Generate training data:

   ```bash
   python main.py generate-data --num-records 10000 --failure-rate 0.05
   ```

5. Train the model:

   ```bash
   python main.py train-svm --grid-search --visualize
   ```

6. Test a prediction:

   ```bash
   python main.py predict-svm --interactive
   ```

7. Start the API server:
   ```bash
   python -m src.api
   ```

### Kubernetes Deployment in 5 Minutes

1. Build the Docker image:

   ```bash
   docker build -t k8s-failure-prediction:latest .
   ```

2. Deploy the prediction API:

   ```bash
   kubectl apply -f kubernetes/model_manifest/api_deployment.yaml
   ```

3. Deploy the node monitor:

   ```bash
   kubectl apply -f kubernetes/services/node_monitor_deployment.yaml
   ```

4. Install the monitoring dashboard:
   ```bash
   kubectl apply -f kubernetes/services/dashboard.yaml
   ```

## Detailed Workflow

### 1. Data Generation

The project can generate synthetic data with realistic failure patterns to train the model:

```bash
python main.py generate-data --num-records 100000 --failure-rate 0.05 --output-path src/data/generated_metrics.csv
```

Parameters:

- `--num-records`: Number of records to generate (default: 10000)
- `--failure-rate`: Proportion of failure records (default: 0.05)
- `--output-path`: Path to save generated data
- `--skip-training`: Skip training after data generation
- `--skip-testing`: Skip testing after training

The generated data includes the following metrics:

- CPU usage percentage
- Memory usage percentage
- Network latency (ms)
- Disk I/O (MBps)
- Error rate
- Average response time (ms)
- Pod restart count
- CPU throttling percentage

### 2. Training the SVM Model

To train an optimized SVM model on the generated data:

```bash
python main.py train-svm --data-path src/data/generated_metrics.csv --grid-search --visualize
```

Parameters:

- `--data-path`: Path to training data CSV file
- `--test-size`: Size of test split (default: 0.2)
- `--random-state`: Random seed for reproducibility (default: 42)
- `--grid-search`: Enable grid search for hyperparameter tuning
- `--visualize`: Generate visualizations after training

The training process includes:

1. Data loading and preprocessing
2. Feature standardization
3. Training/testing data split
4. Grid search for hyperparameter optimization (if enabled)
5. Model training and evaluation
6. Model persistence to disk
7. Performance visualization (if enabled)

The trained model and metrics are saved to:

- Model: `src/models/saved/svm_model.pkl`
- Scaler: `src/models/saved/svm_scaler.pkl`
- Metrics: `src/metrics/svm_metrics.json`
- Visualizations: `src/plots/` directory

### 3. Making Predictions

You can use the trained model to make predictions in several ways:

#### Command Line Predictions

Provide specific metric values to get a prediction:

```bash
python main.py predict-svm --cpu 85.3 --memory 92.1 --network 75.6 --disk 45.2 --error 0.5 --response 180.3 --restarts 1 --throttle 35.7
```

Parameters:

- `--model-path`: Path to trained model (default: src/models/saved/svm_model.pkl)
- `--cpu`: CPU usage percentage (0-100)
- `--memory`: Memory usage percentage (0-100)
- `--network`: Network latency in ms
- `--disk`: Disk I/O in MBps
- `--error`: Error rate (0-1)
- `--response`: Average response time in ms
- `--restarts`: Number of pod restarts
- `--throttle`: CPU throttling percentage (0-100)
- `--output-path`: Path to save prediction results (JSON)

#### Interactive Mode

Run in interactive mode to input values through a user-friendly interface:

```bash
python main.py predict-svm --interactive
```

The output includes:

- Failure probability (0-1)
- Binary prediction (LIKELY TO FAIL / NOT LIKELY TO FAIL)
- Actionable recommendations based on the specific metrics

### 4. Visualizing Model Performance

To generate visualizations of model performance and prediction history:

```bash
python main.py visualize
```

This generates the following visualizations in the `src/plots/` directory:

- Learning curve: Shows how model performance improves with more training data
- ROC curve: Displays the trade-off between true positive and false positive rates
- Confusion matrix: Shows prediction accuracy for failure and non-failure cases
- Feature importance: Identifies which metrics most strongly influence predictions
- Decision boundary: Visualizes how the model separates failure and non-failure cases
- Prediction history: Tracks failure probability over time
- Metric trends: Shows how different metrics change over time
- Metric correlations: Identifies relationships between metrics

## API Integration

The project includes a Flask API for integrating with Kubernetes monitoring systems:

### Starting the API Server

For development:

```bash
python -m src.api
```

For production with Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 src.api:app
```

### API Endpoints

1. **Root Endpoint**

   - URL: `/`
   - Method: GET
   - Description: Provides API information and available endpoints
   - Response Example:
     ```json
     {
       "name": "Kubernetes Node Failure Prediction API",
       "endpoints": {
         "/predict": "POST - Make a failure prediction",
         "/health": "GET - Check API health",
         "/metrics": "GET - Prometheus metrics"
       }
     }
     ```

2. **Health Check**

   - URL: `/health`
   - Method: GET
   - Description: Verifies API and model health
   - Response Example:
     ```json
     {
       "status": "healthy",
       "model_loaded": true
     }
     ```

3. **Metrics**

   - URL: `/metrics`
   - Method: GET
   - Description: Exposes Prometheus metrics
   - Response Format: Prometheus text-based format

4. **Prediction**
   - URL: `/predict`
   - Method: POST
   - Description: Makes a node failure prediction based on provided metrics
   - Request Body Example:
     ```json
     {
       "node_name": "worker-node-1",
       "cpu_usage_percent": 85.3,
       "memory_usage_percent": 92.1,
       "network_latency_ms": 75.6,
       "disk_io_mbps": 45.2,
       "error_rate": 0.5,
       "avg_response_time_ms": 180.3,
       "pod_restarts": 1,
       "cpu_throttling_percent": 35.7
     }
     ```
   - Response Example:
     ```json
     {
       "prediction": "LIKELY TO FAIL",
       "failure_probability": 0.89,
       "recommendations": [
         "CRITICAL: Reduce CPU usage below 80% immediately",
         "CRITICAL: Free up memory resources immediately",
         "WARNING: Network latency is increasing"
       ],
       "timestamp": "2023-07-25T14:30:45Z"
     }
     ```

### Testing the API

You can test the API using curl:

```bash
curl -X POST \
  http://localhost:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "node_name": "worker-node-1",
    "cpu_usage_percent": 85.3,
    "memory_usage_percent": 92.1,
    "network_latency_ms": 75.6,
    "disk_io_mbps": 45.2,
    "error_rate": 0.5,
    "avg_response_time_ms": 180.3,
    "pod_restarts": 1,
    "cpu_throttling_percent": 35.7
  }'
```

## Kubernetes Integration Guide

This section provides a comprehensive guide to deploying and integrating the failure prediction system in a Kubernetes cluster.

### Prerequisites

- A running Kubernetes cluster (v1.19+)
- kubectl configured to communicate with your cluster
- Docker installed for building images
- (Optional) Prometheus and Grafana installed for monitoring

### 1. Building the Docker Image

First, build the Docker image:

```bash
docker build -t k8s-failure-prediction:latest .
```

For production environments, push the image to a container registry:

```bash
docker tag k8s-failure-prediction:latest your-registry/k8s-failure-prediction:latest
docker push your-registry/k8s-failure-prediction:latest
```

Remember to update the image reference in the deployment manifests if using a custom registry.

### 2. Setting Up Persistent Storage

The system uses PersistentVolumeClaims for storing:

- Trained models
- Prediction history

These claims are defined in `kubernetes/model_manifest/api_deployment.yaml`. Ensure your cluster has a StorageClass that supports dynamic provisioning, or edit the manifests to use your specific storage solution.

### 3. Deploying the API Service

Deploy the prediction API service to your cluster:

```bash
kubectl apply -f kubernetes/model_manifest/api_deployment.yaml
```

This creates:

- A Deployment with 2 replicas of the API service
- A Service that exposes the API within the cluster
- PersistentVolumeClaims for model storage and prediction history
- An Ingress to expose the API externally (if needed)

Verify the deployment:

```bash
kubectl get pods -l app=node-failure-prediction
kubectl get svc node-failure-prediction-svc
```

### 4. Setting Up RBAC for Node Monitoring

The node monitor requires permissions to access node metrics. The required RBAC resources are included in `kubernetes/services/node_monitor_deployment.yaml`:

- ServiceAccount: `node-monitor-sa`
- ClusterRole: `node-monitor-role` with permissions to access node and pod metrics
- ClusterRoleBinding: Binds the ServiceAccount to the ClusterRole

### 5. Configuring the Node Monitor

The node monitor is configured through a ConfigMap defined in `kubernetes/services/node_monitor_deployment.yaml`. Key configuration options include:

- Monitoring thresholds for each metric
- Alert channels (e.g., Slack webhook URL)
- Monitoring interval and prediction threshold
- Namespaces to monitor

Edit the ConfigMap section in `kubernetes/services/node_monitor_deployment.yaml` before deploying:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: node-monitor-config
data:
  config.json: |
    {
      "thresholds": {
        "cpu_usage_percent": 80,
        "memory_usage_percent": 85,
        # ...other thresholds
      },
      "alert_channels": [
        {
          "type": "slack",
          "webhook_url": "https://hooks.slack.com/services/YOUR_WEBHOOK_URL"
        }
      ],
      "monitoring_settings": {
        "interval_seconds": 60,
        "prediction_threshold": 0.7,
        "include_system_pods": false,
        "watch_namespaces": ["default", "kube-system", "monitoring"]
      }
    }
```

### 6. Deploying the Node Monitor

Deploy the node monitoring service:

```bash
kubectl apply -f kubernetes/services/node_monitor_deployment.yaml
```

Verify the deployment:

```bash
kubectl get pods -l app=node-monitor
```

Check the logs to ensure it's running correctly:

```bash
kubectl logs -l app=node-monitor
```

### 7. Setting Up Prometheus Monitoring

The API service exposes Prometheus metrics at the `/metrics` endpoint. The deployment manifest includes annotations for Prometheus service discovery:

```yaml
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/path: "/metrics"
  prometheus.io/port: "5000"
```

If you're using the Prometheus Operator, deploy the included ServiceMonitor:

```bash
kubectl apply -f kubernetes/services/dashboard.yaml
```

Available metrics include:

- `node_failure_prediction_requests_total{result="likely_to_fail|not_likely_to_fail"}` - Counter of prediction requests
- `node_failure_prediction_latency_seconds` - Histogram of prediction latency
- `node_failure_probability{node="<node_name>"}` - Gauge of failure probability by node
- `node_failure_alerts_total{node="<node_name>"}` - Counter of alerts generated
- `node_cpu_usage_percent{node="<node_name>"}` - CPU usage by node
- `node_memory_usage_percent{node="<node_name>"}` - Memory usage by node

### 8. Installing the Grafana Dashboard

If you have Grafana installed in your cluster:

```bash
kubectl apply -f kubernetes/services/dashboard.yaml
```

This installs a Grafana dashboard that displays:

- Node CPU and memory usage trends
- Failure probability gauges for each node
- Alert history and count
- Node status table with health indicators
- Prediction request metrics

To access the dashboard:

1. Open your Grafana instance
2. Navigate to Dashboards > Manage
3. Look for "Kubernetes Node Failure Prediction"

### 9. Testing the Complete System

To verify the entire system is working:

1. Check that the API service is running:

   ```bash
   kubectl port-forward svc/node-failure-prediction-svc 5000:80
   ```

   Then in another terminal:

   ```bash
   curl http://localhost:5000/health
   ```

2. Verify the node monitor is collecting metrics and making predictions:

   ```bash
   kubectl logs -l app=node-monitor -f
   ```

3. Access the Grafana dashboard to see the visualized metrics

4. Test a manual prediction through the API:
   ```bash
   curl -X POST \
     http://localhost:5000/predict \
     -H 'Content-Type: application/json' \
     -d '{
       "node_name": "test-node",
       "cpu_usage_percent": 95.0,
       "memory_usage_percent": 98.0,
       "network_latency_ms": 40.0,
       "disk_io_mbps": 85.0,
       "error_rate": 0.65,
       "avg_response_time_ms": 50.0,
       "pod_restarts": 6,
       "cpu_throttling_percent": 75.0
     }'
   ```

### 10. Configuring Alerts

For critical alerts, configure Prometheus AlertManager rules:

```yaml
groups:
  - name: node-failure-prediction
    rules:
      - alert: NodeFailurePredicted
        expr: node_failure_probability > 0.7
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Node failure predicted for {{ $labels.node }}"
          description: "Node {{ $labels.node }} has {{ $value }} failure probability"
```

## Example Prediction Cases

### High Risk Node (Likely to Fail)

```bash
python main.py predict-svm --cpu 95.0 --memory 98.0 --network 40.0 --disk 85.0 --error 0.65 --response 50.0 --restarts 6 --throttle 75.0
```

Expected result: HIGH probability of failure (>0.8) with critical action recommendations.

### Moderate Risk Node (Warning Zone)

```bash
python main.py predict-svm --cpu 82.0 --memory 88.0 --network 22.0 --disk 70.0 --error 0.28 --response 30.0 --restarts 3 --throttle 35.0
```

Expected result: MODERATE probability of failure (0.4-0.7) with warning recommendations.

### Healthy Node (Low Risk)

```bash
python main.py predict-svm --cpu 50.0 --memory 60.0 --network 10.0 --disk 40.0 --error 0.05 --response 15.0 --restarts 1 --throttle 15.0
```

Expected result: LOW probability of failure (<0.3) with normal operation recommendations.

## Troubleshooting

### Common Issues

1. **Model not loaded error**:

   ```
   Error: Model not loaded
   ```

   Solution: Ensure the model file exists at the expected path. If using PVCs, check if the volume is mounted correctly.

   ```bash
   kubectl exec -it $(kubectl get pod -l app=node-failure-prediction -o name | head -1) -- ls -la /app/src/models/saved
   ```

2. **Permission issues with node monitor**:

   ```
   Error: Forbidden: pods is forbidden: User "system:serviceaccount:default:node-monitor-sa" cannot list resource "pods"
   ```

   Solution: Verify that the RBAC resources are correctly applied:

   ```bash
   kubectl get clusterrolebinding node-monitor-binding
   kubectl get clusterrole node-monitor-role
   ```

3. **API service not reachable**:
   Solution: Check the service and endpoints:

   ```bash
   kubectl get svc node-failure-prediction-svc
   kubectl get endpoints node-failure-prediction-svc
   ```

4. **Prometheus metrics not appearing**:
   Solution: Verify that Prometheus can access the metrics endpoint:
   ```bash
   kubectl port-forward svc/node-failure-prediction-svc 5000:80
   curl http://localhost:5000/metrics
   ```

## Model Performance

The SVM model for Kubernetes node failure prediction achieves realistic performance metrics:

- **Precision**: 89.48% (proportion of correctly predicted failures)
- **Recall**: 88.59% (proportion of actual failures detected)
- **F1 Score**: 89.03% (harmonic mean of precision and recall)
- **Accuracy**: 96.43% (overall correct predictions)
- **ROC AUC**: 85.12% (area under ROC curve)

These performance metrics reflect a well-balanced model that provides good predictive capability while maintaining realistic expectations for real-world scenarios. The model avoids false alarms while still catching most potential failure cases.
