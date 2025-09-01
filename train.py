import json
import pandas as pd
import joblib
import logging
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
import typer
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

app = typer.Typer(help="Train an Iris classifier with versioned artifacts.")

@app.command()
def main(config: Path = typer.Option(..., help="Path to JSON config")):
    # Load params from config
    with open(config, "r") as f:
        params = json.load(f)

    # Dynamically resolve project_path from config or fallback to current directory
    project_path_str = params.get("project_path", config.parent)
    if project_path_str == "{{cwd}}":
        project_path = Path(os.getcwd()).resolve()
    else:
        project_path = Path(project_path_str).resolve()

    logging.debug(f"Project path: {project_path}")

    # Create artifacts dirs
    artifacts_dir = project_path / "artifacts"
    models_dir = artifacts_dir / "models"
    metrics_dir = artifacts_dir / "metrics"
    preprocessors_dir = artifacts_dir / "preprocessors"
    
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    preprocessors_dir.mkdir(parents=True, exist_ok=True)

    # Resolve dataset path relative to config file
    data_path = (config.parent / params["data_path"]).resolve()
    logging.debug(f"Data path: {data_path}")

    # Load dataset
    df = pd.read_csv(data_path)

    # Drop empty rows if any
    df = df.dropna()
    row_count = len(df)
    logging.debug(f"Loaded dataset with {row_count} rows.")

    if row_count != 150:
        warnings.warn(f"Expected 150 rows, but got {row_count}. Proceeding anyway.")

    # Unique run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Features & labels
    X = df.drop(columns=["species"])

    # Preprocessor (LabelEncoder)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["species"])

    # Save preprocessor
    preprocessor_path = preprocessors_dir / f"preprocessor_{run_id}.pkl"
    logging.debug(f"Saving preprocessor at: {preprocessor_path}")
    joblib.dump(label_encoder, preprocessor_path)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params.get("test_size", 0.2),
        random_state=params.get("random_state", 42),
        stratify=y
    )

    # Model
    model = LogisticRegression(
        max_iter=params.get("max_iter", 200),
        C=params.get("C", 1.0),
        solver=params.get("solver", "lbfgs"),
        multi_class="auto"
    )
    logging.debug("Fitting the model.")
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save model
    model_path = models_dir / f"model_{run_id}.pkl"
    logging.debug(f"Saving model at: {model_path}")
    joblib.dump(model, model_path)
    logging.debug(f"Model saved at: {model_path}")

    # Save metrics
    metrics = {
        "run_id": run_id,
        "config": params,
        "accuracy": acc,
        "report": report,
    }
    metrics_path = metrics_dir / f"metrics_{run_id}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logging.debug(f"Run {run_id} completed.")
    logging.debug(f"Model saved to {model_path}.")
    logging.debug(f"Metrics saved to {metrics_path}.")
    logging.debug(f"Preprocessor saved to {preprocessor_path}.")

if __name__ == "__main__":
    app()
