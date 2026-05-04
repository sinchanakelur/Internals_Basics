import mlflow
from mlflow.tracking import MlflowClient
import json

client = MlflowClient()

experiment = client.get_experiment_by_name("genomeflow-sequencing-hours")
runs = client.search_runs(experiment.experiment_id)

# Filter runs that actually have mae
valid_runs = [r for r in runs if "mae" in r.data.metrics]

if len(valid_runs) == 0:
    raise Exception("No runs with MAE found. Check training step.")

# Get best run (lowest MAE)
best_run = sorted(valid_runs, key=lambda x: x.data.metrics["mae"])[0]

model_uri = f"runs:/{best_run.info.run_id}/Ridge"

registered_model = mlflow.register_model(
    model_uri,
    "genomeflow-sequencing-hours-predictor"
)

output = {
    "registered_model_name": "genomeflow-sequencing-hours-predictor",
    "version": registered_model.version,
    "run_id": best_run.info.run_id,
    "source_metric": "mae",
    "source_metric_value": best_run.data.metrics["mae"]
}

with open("results/step3_s6.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 3 DONE")