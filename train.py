#!/usr/bin/env python3
"""
train.py — a drop‑in training script for any ML repo.

Features
- Data source, target, model, and outputs are driven by environment variables (see below)
- Handles both classification and regression (auto‑detect or force via env)
- Robust preprocessing with sklearn Pipelines (impute + scale + one‑hot)
- Metrics computed and written to JSON; also printed to STDOUT as a single JSON line
- Saves trained pipeline (preprocess + estimator) to a single joblib artifact
- Minimal dependencies: pandas, numpy, scikit‑learn, joblib

Environment variables (all prefixed MLOPS_)
- DATA_PATH: path to CSV file. If omitted, uses sklearn toy datasets.
- TARGET: target column name (required if DATA_PATH provided).
- FEATURES: comma‑separated feature column names to select (optional).
- ID_COLUMN: column to drop (id/index) if present (optional).
- TASK: one of {auto, classification, regression}. Default: auto.
- MODEL_TYPE: estimator to use; classification: {logreg, rf, svm}; regression: {linreg, rf, svr}. Default depends on TASK.
- PARAMS: JSON string of model hyperparameters, e.g. '{"n_estimators":200}'.
- TEST_SIZE: float in (0,1). Default: 0.2
- RANDOM_STATE: int. Default: 42
- STANDARDIZE: bool ("1"/"true"/"yes"). Default: true for linear/svm, false for rf.
- MAX_SAMPLES: int to downsample rows (useful for quick runs). Optional.
- METRICS_FILE: path to write metrics JSON. Default: ./artifacts/metrics.json
- MODEL_FILE: path to write trained pipeline. Default: ./artifacts/model.joblib
- TAGS: optional comma‑separated run tags; stored in metrics JSON only.
- RUN_ID: optional run identifier; stored in metrics JSON only.

Example
$ export MLOPS_DATA_PATH=data/iris.csv
$ export MLOPS_TARGET=species
$ python train.py

This prints a single JSON line to STDOUT with key metrics and writes artifacts/metrics.json and artifacts/model.joblib by default.
"""

from __future__ import annotations
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR


# -----------------------------
# Utilities
# -----------------------------

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(f"MLOPS_{name}", default)


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _json_or_none(value: Optional[str]) -> Optional[Dict[str, Any]]:
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        print(f"Warning: could not parse JSON from PARAMS: {value}", file=sys.stderr)
        return None


@dataclass
class Config:
    data_path: Optional[str]
    target: Optional[str]
    features: Optional[List[str]]
    id_column: Optional[str]
    task: str
    model_type: Optional[str]
    params: Dict[str, Any]
    test_size: float
    random_state: int
    standardize: Optional[bool]
    max_samples: Optional[int]
    metrics_file: Path
    model_file: Path
    tags: List[str]
    run_id: Optional[str]

    @staticmethod
    def from_env() -> "Config":
        data_path = _env("DATA_PATH")
        target = _env("TARGET")
        features = _env("FEATURES")
        features_list = [c.strip() for c in features.split(",")] if features else None
        id_column = _env("ID_COLUMN")
        task = (_env("TASK", "auto") or "auto").lower()
        model_type = _env("MODEL_TYPE")
        params = _json_or_none(_env("PARAMS")) or {}
        test_size = float(_env("TEST_SIZE", "0.2"))
        random_state = int(_env("RANDOM_STATE", "42"))
        standardize_env = _env("STANDARDIZE")
        standardize = None if standardize_env is None else _as_bool(standardize_env)
        max_samples_env = _env("MAX_SAMPLES")
        max_samples = int(max_samples_env) if max_samples_env else None
        metrics_file = Path(_env("METRICS_FILE", "./artifacts/metrics.json"))
        model_file = Path(_env("MODEL_FILE", "./artifacts/model.joblib"))
        tags = [t.strip() for t in (_env("TAGS") or "").split(",") if t.strip()]
        run_id = _env("RUN_ID")
        return Config(
            data_path,
            target,
            features_list,
            id_column,
            task,
            model_type,
            params,
            test_size,
            random_state,
            standardize,
            max_samples,
            metrics_file,
            model_file,
            tags,
            run_id,
        )


# -----------------------------
# Data loading
# -----------------------------

def load_data(cfg: Config) -> Tuple[pd.DataFrame, pd.Series]:
    if cfg.data_path:
        if not cfg.target:
            raise ValueError("MLOPS_TARGET must be set when using MLOPS_DATA_PATH")
        df = pd.read_csv(cfg.data_path)
        if cfg.id_column and cfg.id_column in df.columns:
            df = df.drop(columns=[cfg.id_column])
        if cfg.features:
            missing = [c for c in cfg.features if c not in df.columns]
            if missing:
                raise ValueError(f"Requested FEATURES not in data columns: {missing}")
            X = df[cfg.features].copy()
        else:
            X = df.drop(columns=[cfg.target]).copy()
        y = df[cfg.target].copy()
        return X, y

    # Fallback toy datasets for quick smoke tests
    from sklearn.datasets import load_iris, load_diabetes
    if cfg.task in ("auto", "classification"):
        iris = load_iris(as_frame=True)
        X = iris.data
        y = iris.target
        return X, y
    else:
        diab = load_diabetes(as_frame=True)
        X = diab.data
        y = diab.target
        return X, y


# -----------------------------
# Task detection & model selection
# -----------------------------

def detect_task(y: pd.Series, requested: str) -> str:
    if requested != "auto":
        return requested
    # Heuristic: numeric with many unique values -> regression; else classification
    if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 20:
        return "regression"
    return "classification"


def make_estimator(task: str, model_type: Optional[str], params: Dict[str, Any]) -> Any:
    if task == "classification":
        kind = (model_type or "logreg").lower()
        if kind == "rf":
            return RandomForestClassifier(random_state=params.pop("random_state", None), **params)
        if kind == "svm" or kind == "svc":
            return SVC(probability=True, **params)
        # default
        return LogisticRegression(max_iter=1000, **params)
    else:
        kind = (model_type or "linreg").lower()
        if kind == "rf":
            return RandomForestRegressor(random_state=params.pop("random_state", None), **params)
        if kind == "svr" or kind == "svm":
            return SVR(**params)
        # default
        return LinearRegression(**params)


# -----------------------------
# Preprocessing pipeline
# -----------------------------

def build_pipeline(X: pd.DataFrame, estimator: Any, standardize: Optional[bool]) -> Pipeline:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Decide default standardization behavior based on estimator type if None
    if standardize is None:
        needs_scale = any(name in estimator.__class__.__name__.lower() for name in ["logistic", "svc", "svr", "linear"])
        standardize = bool(needs_scale)

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if standardize:
        num_steps.append(("scaler", StandardScaler()))

    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=num_steps), numeric_cols),
            ("cat", Pipeline(steps=cat_steps), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator),
    ])
    return pipe


# -----------------------------
# Metrics
# -----------------------------

def compute_metrics(task: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
    if task == "classification":
        out: Dict[str, Any] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
        # ROC-AUC (binary only)
        try:
            if y_proba is not None and (y_proba.shape[1] == 2 or y_proba.ndim == 1):
                proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                out["roc_auc"] = float(roc_auc_score(y_true, proba))
        except Exception:
            pass
        return out
    else:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": rmse,
        }


# -----------------------------
# I/O helpers
# -----------------------------

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    cfg = Config.from_env()

    # Load data
    X, y = load_data(cfg)

    # Optional subsample for quick runs
    if cfg.max_samples and len(X) > cfg.max_samples:
        X = X.sample(cfg.max_samples, random_state=cfg.random_state)
        y = y.loc[X.index]

    # Detect task and choose estimator
    task = detect_task(y, cfg.task)
    estimator = make_estimator(task, cfg.model_type, cfg.params.copy())

    # Build pipeline
    pipe = build_pipeline(X, estimator, cfg.standardize)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y if task == "classification" else None
    )

    # Fit
    pipe.fit(X_train, y_train)

    # Predict
    y_pred = pipe.predict(X_test)
    y_proba = None
    if task == "classification":
        # Try to get probabilities if available
        try:
            if hasattr(pipe.named_steps["model"], "predict_proba"):
                y_proba = pipe.predict_proba(X_test)
        except Exception:
            y_proba = None

    # Metrics
    metrics = compute_metrics(task, y_test.to_numpy(), y_pred, y_proba)

    # Assemble run report
    report = {
        "run_id": cfg.run_id,
        "tags": cfg.tags,
        "task": task,
        "model_type": (cfg.model_type or ("logreg" if task == "classification" else "linreg")),
        "params": cfg.params,
        "random_state": cfg.random_state,
        "test_size": cfg.test_size,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "metrics": metrics,
        "feature_columns": list(X.columns),
        "artifacts": {
            "model_file": str(cfg.model_file),
            "metrics_file": str(cfg.metrics_file),
        },
    }

    # Persist artifacts
    ensure_parent(cfg.model_file)
    dump(pipe, cfg.model_file)

    write_json(cfg.metrics_file, report)

    # Emit a single JSON line to STDOUT for easy CLI scraping
    print(json.dumps(report, separators=(",", ":")))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        # Print a structured error so CLIs can parse it
        err = {"error": str(e), "type": e.__class__.__name__}
        print(json.dumps(err, separators=(",", ":")), file=sys.stderr)
        raise
