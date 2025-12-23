#!/usr/bin/env python3
"""
Model Training Script untuk MLProject
Dicoding Submission - Kriteria 3 (Advanced)

Script ini didesain untuk:
- Training otomatis via MLflow Project
- CI/CD dengan GitHub Actions
- Docker image deployment
"""

import os
import sys
import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def load_preprocessed_data(data_dir: Path):
    """Load data yang sudah di-preprocess"""
    print(f"Loading data from: {data_dir}")

    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()
    y_test = pd.read_csv(data_dir / "y_test.csv").values.ravel()

    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape:  {X_test.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  y_test shape:  {y_test.shape}\n")

    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_true, y_pred, save_path: str = "confusion_matrix.png"):
    """Plot dan simpan confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_roc_curve(y_true, y_pred_proba, save_path: str = "roc_curve.png"):
    """Plot dan simpan ROC curve"""
    # Convert string labels to binary if needed
    if isinstance(y_true[0], str):
        y_true_binary = (y_true == 'Yes').astype(int)
    else:
        y_true_binary = y_true

    fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_proba)
    auc = roc_auc_score(y_true_binary, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_feature_importance(model, feature_names, save_path: str = "feature_importance.png"):
    """Plot dan simpan feature importance (untuk tree-based models)"""
    if not hasattr(model, 'feature_importances_'):
        return None

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.title('Top 20 Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return save_path


def _train_and_log(model, model_name, X_train, X_test, y_train, y_test, params, run=None):
    """Helper function untuk training dan logging"""

    # Use the provided run context if available
    active_run = run or mlflow.active_run()

    # Log parameters dengan prefix
    if params:
        for key, value in params.items():
            mlflow.log_param(f"{model_name}_{key}", value)

    # Log model class name
    mlflow.log_param(f"{model_name}_type", model.__class__.__name__)

    # Train model
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='Yes', zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label='Yes', zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label='Yes', zero_division=0)

    # Convert string labels to binary for ROC-AUC
    y_test_binary = (y_test == 'Yes').astype(int)
    roc_auc = roc_auc_score(y_test_binary, y_pred_proba)

    # Log metrics dengan prefix
    mlflow.log_metric(f"{model_name}_accuracy", accuracy)
    mlflow.log_metric(f"{model_name}_precision", precision)
    mlflow.log_metric(f"{model_name}_recall", recall)
    mlflow.log_metric(f"{model_name}_f1_score", f1)
    mlflow.log_metric(f"{model_name}_roc_auc", roc_auc)

    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

    # ===== ARTIFACTS (4+) untuk Advanced Criteria =====

    # 1. Confusion Matrix
    cm_path = plot_confusion_matrix(y_test, y_pred, f"{model_name}_cm.png")
    mlflow.log_artifact(cm_path)
    print(f"  [ARTIFACT] {model_name}_cm.png")

    # 2. ROC Curve
    roc_path = plot_roc_curve(y_test, y_pred_proba, f"{model_name}_roc.png")
    mlflow.log_artifact(roc_path)
    print(f"  [ARTIFACT] {model_name}_roc.png")

    # 3. Classification Report (text file)
    report = classification_report(y_test, y_pred,
                                  target_names=['No Churn', 'Churn'])
    report_path = f"{model_name}_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    mlflow.log_artifact(report_path)
    print(f"  [ARTIFACT] {model_name}_report.txt")

    # 4. Feature Importance (untuk tree-based models)
    if hasattr(model, 'feature_importances_'):
        feature_names = X_train.columns.tolist()
        fi_path = plot_feature_importance(model, feature_names, f"{model_name}_fi.png")
        if fi_path:
            mlflow.log_artifact(fi_path)
            print(f"  [ARTIFACT] {model_name}_fi.png")

    # Log model dengan prefix
    mlflow.sklearn.log_model(model, f"{model_name}_model")
    print(f"  [MODEL] Logged to MLflow")

    # Cleanup temporary files
    for temp_file in [f"{model_name}_cm.png", f"{model_name}_roc.png",
                     f"{model_name}_report.txt", f"{model_name}_fi.png"]:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    print(f"\n[DONE] {model_name} training completed")
    print(f"{'='*60}\n")


def train_model_with_mlflow(
    model,
    model_name: str,
    X_train,
    X_test,
    y_train,
    y_test,
    params: dict = None
):
    """
    Train model dengan manual logging MLflow (Advanced Criteria)

    KRITERIA ADVANCED:
    - Manual logging (bukan autolog)
    - 4+ artifacts: confusion matrix, ROC curve, classification report, feature importance
    - Metrics: accuracy, precision, recall, f1_score, roc_auc
    """

    # Log directly to the active run created by mlflow run
    # All 3 models log to same parent run with prefixed names
    _train_and_log(model, model_name, X_train, X_test, y_train, y_test, params)

    return model


def main(data_path: str = "telco_preprocessing",
         model_type: str = "all",
         experiment_name: str = "Telco_Churn_MLProject"):
    """
    Main training function

    Args:
        data_path: Path to preprocessed data folder
        model_type: Type of model to train (lr/rf/gb/all)
        experiment_name: MLflow experiment name
    """

    # Check if running inside mlflow run (MLflow sets MLFLOW_RUN_ID env var)
    inside_mlflow_run = os.environ.get("MLFLOW_RUN_ID") is not None
    run_id = os.environ.get("MLFLOW_RUN_ID")

    if not inside_mlflow_run:
        # Only set experiment when running directly (python modelling.py)
        mlflow.set_experiment(experiment_name)

    # Enable autolog (BASIC requirement untuk Kriteria 2)
    # Will be used by sklearn models automatically
    mlflow.autolog(log_models=False, exclusive=False)

    print(f"\n{'='*60}")
    print(f"MLflow Project - Model Training")
    print(f"Experiment: {experiment_name}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Inside MLflow Run: {inside_mlflow_run}")
    if run_id:
        print(f"Parent Run ID: {run_id}")
    print(f"MLflow Autolog: ENABLED (non-exclusive mode)")
    print(f"{'='*60}\n")

    # Load data
    data_dir = Path(data_path)

    if not data_dir.exists():
        print(f"Error: Data directory tidak ditemukan: {data_dir}")
        print("   Pastikan folder telco_preprocessing ada di MLProject/")
        sys.exit(1)

    X_train, X_test, y_train, y_test = load_preprocessed_data(data_dir)

    # Training models berdasarkan argument
    models_to_train = []

    if model_type in ["lr", "all"]:
        lr_params = {
            "max_iter": 1000,
            "random_state": 42,
            "solver": "lbfgs"
        }
        models_to_train.append(("Logistic_Regression", LogisticRegression(**lr_params), lr_params))

    if model_type in ["rf", "all"]:
        rf_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1
        }
        models_to_train.append(("Random_Forest", RandomForestClassifier(**rf_params), rf_params))

    if model_type in ["gb", "all"]:
        gb_params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42
        }
        models_to_train.append(("Gradient_Boosting", GradientBoostingClassifier(**gb_params), gb_params))

    # Train all selected models with manual logging
    # Use mlflow.start_run() to properly attach to the parent run context
    with mlflow.start_run() as run:
        print(f"[INFO] Active Run ID: {run.info.run_id}")

        for model_name, model, params in models_to_train:
            train_model_with_mlflow(
                model,
                model_name,
                X_train, X_test, y_train, y_test,
                params=params
            )

    print("\n" + "="*60)
    print(f"[SUCCESS] Training completed!")
    print(f"Total models trained: {len(models_to_train)}")
    print(f"View results at: {mlflow.get_tracking_uri()}")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLflow Project - Model Training")
    parser.add_argument(
        "--data-path",
        type=str,
        default="telco_preprocessing",
        help="Path to preprocessed data folder"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="all",
        choices=["lr", "rf", "gb", "all"],
        help="Model type to train (lr=Logistic, rf=RandomForest, gb=GradientBoosting, all=All models)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="Telco_Churn_MLProject",
        help="MLflow experiment name"
    )

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        model_type=args.model_type,
        experiment_name=args.experiment_name
    )
