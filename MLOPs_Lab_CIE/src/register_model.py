import mlflow
import mlflow.sklearn
import json
import os

# -----------------------------
# Model name (as per question)
# -----------------------------
MODEL_NAME = "freshbasket-delivery-time-min-predictor"

# -----------------------------
# Get latest run
# -----------------------------
client = mlflow.tracking.MlflowClient()

experiment = client.get_experiment_by_name("freshbasket-delivery-time-min")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.rmse ASC"],
    max_results=1
)

best_run = runs[0]

run_id = best_run.info.run_id

# -----------------------------
# Register model
# -----------------------------
model_uri = f"runs:/{run_id}/model"

result = mlflow.register_model(model_uri, MODEL_NAME)

# -----------------------------
# Save JSON output
# -----------------------------
output = {
    "registered_model_name": MODEL_NAME,
    "version": result.version,
    "run_id": run_id,
    "source_metric": "rmse",
    "source_metric_value": best_run.data.metrics["rmse"]
}

os.makedirs("results", exist_ok=True)

with open("results/step3_s6.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ Task 3 completed. Model registered.")