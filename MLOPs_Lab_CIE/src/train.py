import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import os

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("data/training_data.csv")

X = df.drop("delivery_time_min", axis=1)
y = df["delivery_time_min"]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# MLflow setup
# -----------------------------
mlflow.set_experiment("freshbasket-delivery-time-min")

results = []

models = {
    "Lasso": Lasso(),
    "RandomForest": RandomForestRegressor(random_state=42)
}

# -----------------------------
# Train models
# -----------------------------
for name, model in models.items():
    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        preds = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, preds)

        # ✅ FIXED RMSE (works for all sklearn versions)
        rmse = mean_squared_error(y_test, preds) ** 0.5

        # Log to MLflow
        mlflow.log_param("model", name)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        # Save results
        results.append({
            "name": name,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4)
        })

# -----------------------------
# Select best model (lowest RMSE)
# -----------------------------
best_model = min(results, key=lambda x: x["rmse"])

final_output = {
    "experiment_name": "freshbasket-delivery-time-min",
    "models": results,
    "best_model": best_model["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best_model["rmse"]
}

# -----------------------------
# Save JSON result
# -----------------------------
os.makedirs("results", exist_ok=True)

with open("results/step1_s1.json", "w") as f:
    json.dump(final_output, f, indent=4)

print("✅ Task 1 completed. Results saved to results/step1_s1.json")