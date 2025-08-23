This is a test project to test the MLOps Pipeline Tool
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