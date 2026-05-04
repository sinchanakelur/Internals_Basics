import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# Load data
train_df = pd.read_csv("data/training_data.csv")
new_df = pd.read_csv("data/new_data.csv")

combined = pd.concat([train_df, new_df])

X = combined.drop("sequencing_hours", axis=1)
y = combined["sequencing_hours"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load existing model
champion = joblib.load("models/best_model.pkl")

champ_preds = champion.predict(X_test)
champ_rmse = np.sqrt(mean_squared_error(y_test, champ_preds))

# Retrain same model type
if isinstance(champion, Ridge):
    new_model = Ridge()
else:
    new_model = RandomForestRegressor(random_state=42)

new_model.fit(X_train, y_train)

new_preds = new_model.predict(X_test)
new_rmse = np.sqrt(mean_squared_error(y_test, new_preds))

improvement = champ_rmse - new_rmse

if improvement >= 0.5:
    action = "promoted"
    joblib.dump(new_model, "models/best_model.pkl")
else:
    action = "kept_champion"

output = {
    "original_data_rows": len(train_df),
    "new_data_rows": len(new_df),
    "combined_data_rows": len(combined),
    "champion_rmse": champ_rmse,
    "retrained_rmse": new_rmse,
    "improvement": improvement,
    "min_improvement_threshold": 0.5,
    "action": action,
    "comparison_metric": "rmse"
}

with open("results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 4 DONE")