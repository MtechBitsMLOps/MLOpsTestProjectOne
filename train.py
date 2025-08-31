import json
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
import typer

app = typer.Typer(help="Train an Iris classifier with versioned artifacts.")


@app.command()
def main(config: Path = typer.Option(..., help="Path to JSON config")):
    # Load params
    with open(config, "r") as f:
        params = json.load(f)

    # Create artifacts dirs
    artifacts_dir = Path("artifacts")
    models_dir = artifacts_dir / "models"
    metrics_dir = artifacts_dir / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_csv(params["data_path"])

    # Drop empty rows if any
    df = df.dropna()
    row_count = len(df)

    if row_count != 150:
        warnings.warn(f"Expected 150 rows, but got {row_count}. Proceeding anyway.")

    # Features & labels
    X = df.drop(columns=["species"])
    y = LabelEncoder().fit_transform(df["species"])

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
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Unique run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model
    model_path = models_dir / f"model_{run_id}.pkl"
    joblib.dump(model, model_path)

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

    typer.echo(f"✅ Run {run_id} completed")
    typer.echo(f"   Model saved:   {model_path}")
    typer.echo(f"   Metrics saved: {metrics_path}")


if __name__ == "__main__":
    app()
