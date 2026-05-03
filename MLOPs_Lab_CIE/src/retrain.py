import pandas as pd
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# -----------------------------
# Load datasets
# -----------------------------
train_df = pd.read_csv("data/training_data.csv")
new_df = pd.read_csv("data/new_data.csv")

combined_df = pd.concat([train_df, new_df], ignore_index=True)

# -----------------------------
# Prepare data
# -----------------------------
X = train_df.drop("delivery_time_min", axis=1)
y = train_df["delivery_time_min"]

X_new = combined_df.drop("delivery_time_min", axis=1)
y_new = combined_df["delivery_time_min"]

# -----------------------------
# SAME TEST SPLIT (important)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_new, _, y_train_new, _ = train_test_split(
    X_new, y_new, test_size=0.2, random_state=42
)

# -----------------------------
# Train original model
# -----------------------------
model_old = Lasso(alpha=0.01)
model_old.fit(X_train, y_train)

preds_old = model_old.predict(X_test)
rmse_old = mean_squared_error(y_test, preds_old) ** 0.5

# -----------------------------
# Retrain model
# -----------------------------
model_new = Lasso(alpha=0.01)
model_new.fit(X_train_new, y_train_new)

preds_new = model_new.predict(X_test)
rmse_new = mean_squared_error(y_test, preds_new) ** 0.5

# -----------------------------
# Compare
# -----------------------------
improvement = rmse_old - rmse_new

if improvement >= 0.5:
    action = "promoted"
else:
    action = "kept_champion"

# -----------------------------
# Save JSON
# -----------------------------
output = {
    "original_data_rows": len(train_df),
    "new_data_rows": len(new_df),
    "combined_data_rows": len(combined_df),
    "champion_rmse": round(rmse_old, 4),
    "retrained_rmse": round(rmse_new, 4),
    "improvement": round(improvement, 4),
    "min_improvement_threshold": 0.5,
    "action": action,
    "comparison_metric": "rmse"
}

os.makedirs("results", exist_ok=True)

with open("results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ Task 4 completed.")