import pandas as pd
import mlflow
import mlflow.sklearn
import json
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("data/training_data.csv")

X = df.drop("delivery_time_min", axis=1)
y = df["delivery_time_min"]

# -----------------------------
# Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# MLflow setup
# -----------------------------
mlflow.set_experiment("freshbasket-delivery-time-min")

# -----------------------------
# Parameter grid (for Lasso)
# -----------------------------
param_grid = {
    "alpha": [0.01, 0.1, 1, 10]
}

# -----------------------------
# Grid Search
# -----------------------------
model = Lasso()

grid = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring="neg_mean_squared_error"
)

with mlflow.start_run(run_name="tuning-freshbasket"):

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    preds = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    # Log best params
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)

# -----------------------------
# Save JSON output
# -----------------------------
output = {
    "search_type": "grid",
    "n_folds": 3,
    "total_trials": len(grid.cv_results_["params"]),
    "best_params": grid.best_params_,
    "best_mae": round(mae, 4),
    "best_rmse": round(rmse, 4),
    "parent_run_name": "tuning-freshbasket"
}

os.makedirs("results", exist_ok=True)

with open("results/step2_s2.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ Task 2 completed. Results saved to results/step2_s2.json")