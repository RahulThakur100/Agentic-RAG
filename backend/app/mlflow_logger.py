import mlflow
from pathlib import Path
from .config import MLFLOW_TRACKING_URI


# Prefer explicit tracking URI from env/.env, otherwise fall back to local sqlite DB
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
else:
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _MLFLOW_DB_PATH = _PROJECT_ROOT / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{_MLFLOW_DB_PATH.as_posix()}")


def start_run(name: str, params: dict):
    mlflow.set_experiment(name)
    mlflow.start_run()
    mlflow.log_params(params)


def end_run(metrics: dict, artifacts: dict):
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
    for fname, path in artifacts.items():
        mlflow.log_artifact(path, artifact_path=fname)
    mlflow.end_run()
