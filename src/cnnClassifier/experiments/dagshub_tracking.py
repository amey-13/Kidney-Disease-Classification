import dagshub
import mlflow

def run_dagshub_experiment():
    # Initialize DagsHub and MLFlow tracking
    dagshub.init(repo_owner='amey-13', repo_name='Kidney-Disease-Classification', mlflow=True)

    # Start an MLFlow run and log parameters/metrics
    with mlflow.start_run():
        mlflow.log_param('parameter name', 'value')
        mlflow.log_metric('metric name', 1)