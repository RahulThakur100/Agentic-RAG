import mlflow


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
