import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import os

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("sequencing_hours", axis=1)
y = df["sequencing_hours"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("genomeflow-sequencing-hours")

results = []

models = {
    "Ridge": Ridge(),
    "RandomForest": RandomForestRegressor(random_state=42)
}

best_model_name = None
best_mae = float("inf")
best_model = None

for name, model in models.items():
    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        mlflow.log_param("model_name", name)
        mlflow.log_metrics({"mae": mae, "rmse": rmse})
        mlflow.set_tag("project_phase", "model_selection")

        mlflow.sklearn.log_model(model, name)

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse
        })

        if mae < best_mae:
            best_mae = mae
            best_model_name = name
            best_model = model

# Save best model
os.makedirs("models", exist_ok=True)
import joblib
joblib.dump(best_model, "models/best_model.pkl")

# Save JSON
output = {
    "experiment_name": "genomeflow-sequencing-hours",
    "models": results,
    "best_model": best_model_name,
    "best_metric_name": "mae",
    "best_metric_value": best_mae
}

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 1 completed!")