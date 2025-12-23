#!/usr/bin/env python3
"""
Prometheus Custom Metrics Exporter
Untuk Dicoding Submission - Kriteria 4 (Advanced)

Export custom metrics untuk monitoring model performance
"""

import time
import random
import numpy as np
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Info
from datetime import datetime
import psutil
import os

# ============================================================
# CUSTOM METRICS (Additional metrics untuk Advanced level)
# ============================================================

# Model performance metrics
model_training_time = Gauge(
    'model_training_time_seconds',
    'Time taken for model training'
)

model_size_mb = Gauge(
    'model_size_mb',
    'Size of the model in megabytes'
)

dataset_size = Gauge(
    'dataset_size_rows',
    'Number of rows in the dataset'
)

feature_count = Gauge(
    'feature_count',
    'Number of features in the dataset'
)

# Data quality metrics
missing_values_count = Gauge(
    'missing_values_count',
    'Number of missing values in the dataset'
)

data_drift_score = Gauge(
    'data_drift_score',
    'Data drift score (0-1, higher is worse)'
)

# Prediction distribution
churn_rate = Gauge(
    'churn_rate_percent',
    'Percentage of churn predictions'
)

no_churn_rate = Gauge(
    'no_churn_rate_percent',
    'Percentage of no-churn predictions'
)

# System resource metrics
system_cpu_percent = Gauge(
    'system_cpu_percent',
    'System CPU usage percentage'
)

system_memory_percent = Gauge(
    'system_memory_percent',
    'System memory usage percentage'
)

system_disk_percent = Gauge(
    'system_disk_percent',
    'System disk usage percentage'
)

# Service health metrics
service_health_status = Gauge(
    'service_health_status',
    'Service health status (1=healthy, 0=unhealthy)'
)

model_version = Info(
    'model_version',
    'Information about the current model version'
)

# Business metrics
daily_predictions = Counter(
    'daily_predictions_total',
    'Total predictions made today'
)

high_risk_customers = Gauge(
    'high_risk_customers_count',
    'Number of customers with high churn risk (>70% probability)'
)

# ============================================================
# METRIC COLLECTION FUNCTIONS
# ============================================================

def collect_system_metrics():
    """Collect system resource metrics"""
    try:
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent

        system_cpu_percent.set(cpu)
        system_memory_percent.set(memory)
        system_disk_percent.set(disk)

        print(f"[System Metrics] CPU: {cpu:.1f}%, Memory: {memory:.1f}%, Disk: {disk:.1f}%")
    except Exception as e:
        print(f"Error collecting system metrics: {e}")

def collect_model_metrics():
    """Collect model-related metrics"""
    try:
        # Simulate model metrics (in production, get from actual model)
        model_training_time.set(np.random.uniform(180, 300))  # 3-5 minutes
        model_size_mb.set(np.random.uniform(5, 15))
        dataset_size.set(7043)  # Actual dataset size
        feature_count.set(45)

        print("[Model Metrics] Training time, size, dataset updated")
    except Exception as e:
        print(f"Error collecting model metrics: {e}")

def collect_data_quality_metrics():
    """Collect data quality metrics"""
    try:
        missing_values_count.set(np.random.randint(0, 50))
        data_drift_score.set(np.random.uniform(0.01, 0.15))  # Low drift

        print("[Data Quality] Missing values and drift score updated")
    except Exception as e:
        print(f"Error collecting data quality metrics: {e}")

def collect_prediction_metrics():
    """Collect prediction distribution metrics"""
    try:
        # Simulate prediction distribution
        churn_pct = np.random.uniform(20, 35)  # 20-35% churn rate
        churn_rate.set(churn_pct)
        no_churn_rate.set(100 - churn_pct)

        # High risk customers
        high_risk = np.random.randint(50, 200)
        high_risk_customers.set(high_risk)

        print(f"[Predictions] Churn rate: {churn_pct:.1f}%, High risk: {high_risk}")
    except Exception as e:
        print(f"Error collecting prediction metrics: {e}")

def update_service_health():
    """Update service health status"""
    try:
        # Check if service is healthy
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent

        # Service is healthy if CPU < 90% and Memory < 85%
        is_healthy = cpu < 90 and memory < 85
        service_health_status.set(1 if is_healthy else 0)

        status = "HEALTHY" if is_healthy else "UNHEALTHY"
        print(f"[Health] Service status: {status}")
    except Exception as e:
        print(f"Error updating service health: {e}")
        service_health_status.set(0)

def set_model_info():
    """Set model version information"""
    try:
        model_version.info({
            'version': '1.0.0',
            'algorithm': 'Random Forest',
            'framework': 'scikit-learn',
            'trained_date': '2025-12-22',
            'accuracy': '0.8091',
            'f1_score': '0.6027'
        })
        print("[Info] Model version information set")
    except Exception as e:
        print(f"Error setting model info: {e}")

# ============================================================
# MAIN LOOP
# ============================================================

def collect_all_metrics():
    """Collect all metrics"""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Collecting metrics...")

    collect_system_metrics()
    collect_model_metrics()
    collect_data_quality_metrics()
    collect_prediction_metrics()
    update_service_health()

    # Increment daily predictions counter (simulate)
    daily_predictions.inc(np.random.randint(5, 20))

    print("=" * 60)

def main():
    """Main function"""
    print("=" * 60)
    print("Prometheus Custom Metrics Exporter")
    print("Dicoding Submission - Kriteria 4 (Advanced)")
    print("=" * 60)

    # Start HTTP server for metrics
    port = 8000
    start_http_server(port)
    print(f"\nMetrics server started on port {port}")
    print(f"Metrics available at: http://localhost:{port}/metrics")
    print("\nCollecting metrics every 10 seconds...")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    # Set model info (once)
    set_model_info()

    # Collect metrics periodically
    try:
        while True:
            collect_all_metrics()
            time.sleep(10)  # Collect every 10 seconds
    except KeyboardInterrupt:
        print("\n\nStopping metrics exporter...")
        print("Goodbye!")

if __name__ == '__main__':
    main()
