#!/usr/bin/env python3
"""
MLOps-compatible training script.

- Reads environment variables (MLOPS_*)
- Handles headerless CSVs with default column names
- Timestamped artifact directories
- Supports classification and regression tasks
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR


# Default columns if CSV has no header
DEFAULT_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]


# -----------------------------
# Helpers
# -----------------------------
def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(f"MLOPS_{name}", default)


# -----------------------------
# Load data
# -----------------------------
def load_data(csv_path: Path, features: list[str], target: str):
    """Load dataset from CSV and handle headerless CSVs if necessary."""
    try:
        df = pd.read_csv(csv_path)
        if set(DEFAULT_COLUMNS).issubset(range(df.shape[1])) or df.columns.tolist() == list(range(df.shape[1])):
            df = pd.read_csv(csv_path, header=None, names=DEFAULT_COLUMNS)

        missing = [f for f in features + [target] if f not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in data: {missing}")

        return df[features], df[target]

    except Exception as e:
        raise RuntimeError(f"Failed to load data from {csv_path}: {e}") from e


# -----------------------------
# Model selection
# -----------------------------
def make_estimator(task: str, model_type: str, params: dict):
    task = task.lower()
    model_type = model_type.lower() if model_type else None
    if task == "classification":
        if model_type == "rf":
            return RandomForestClassifier(random_state=params.pop("random_state", None), **params)
        if model_type in {"svc", "svm"}:
            return SVC(probability=True, **params)
        return LogisticRegression(max_iter=1000, **params)
    else:
        if model_type == "rf":
            return RandomForestRegressor(random_state=params.pop("random_state", None), **params)
        if model_type in {"svr", "svm"}:
            return SVR(**params)
        return LinearRegression(**params)


# -----------------------------
# Train and evaluate
# -----------------------------
def train_and_evaluate(X, y, estimator, test_size: float, random_state: int, task: str):
    stratify = y if task.lower() == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)

    metrics = {}
    if task.lower() == "classification":
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
    else:
        metrics["r2"] = float(r2_score(y_test, y_pred))
        metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
        metrics["rmse"] = float(mean_squared_error(y_test, y_pred, squared=False))

    return estimator, metrics


# -----------------------------
# Save artifacts
# -----------------------------
def save_artifacts(model, metrics, model_file_env: str, metrics_file_env: str):
    """Save model and metrics in timestamped subdirectories under the specified base paths."""
    timestamp = datetime.now().strftime("%d%m%y_%H%M%S")

    # Model path
    model_base = Path(model_file_env)
    model_dir = model_base.parent / datetime.now().strftime("%d%m%y") if model_base.suffix else model_base / datetime.now().strftime("%d%m%y")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / f"model_{timestamp}.joblib"
    joblib.dump(model, model_file)

    # Metrics path
    metrics_base = Path(metrics_file_env)
    metrics_dir = metrics_base.parent / datetime.now().strftime("%d%m%y") if metrics_base.suffix else metrics_base / datetime.now().strftime("%d%m%y")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_dir / f"metrics_{timestamp}.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return model_file, metrics_file


# -----------------------------
# Main
# -----------------------------
def main():
    try:
        # Environment variables
        data_path = Path(_env("DATA_PATH", "iris.csv"))
        target = _env("TARGET", "species")
        features_env = _env("FEATURES")
        features_list = [f.strip() for f in features_env.split(",")] if features_env else DEFAULT_COLUMNS[:-1]
        task = _env("TASK", "classification")
        model_type = _env("MODEL_TYPE", None)
        params = json.loads(_env("PARAMS", "{}"))
        test_size = float(_env("TEST_SIZE", 0.2))
        random_state = int(_env("RANDOM_STATE", 42))
        model_file_env = _env("MODEL_FILE", "./artifacts/model.joblib")
        metrics_file_env = _env("METRICS_FILE", "./artifacts/metrics.json")

        # Load data
        X, y = load_data(data_path, features_list, target)

        # Train model
        estimator = make_estimator(task, model_type, params)
        estimator, metrics = train_and_evaluate(X, y, estimator, test_size, random_state, task)

        # Save artifacts
        model_file, metrics_file = save_artifacts(estimator, metrics, model_file_env, metrics_file_env)

        # Output JSON for CLI capture
        print(json.dumps({"metrics": metrics, "model_file": str(model_file), "metrics_file": str(metrics_file)}))

    except Exception as e:
        print(json.dumps({"error": str(e), "type": type(e).__name__}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
