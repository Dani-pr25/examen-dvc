
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import json

# Load scaled data
X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()

# Define model and parameter grid
model = RandomForestRegressor(random_state=42)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}

# GridSearch
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring="r2", n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Save best model parameters
#best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Save to .pkl
os.makedirs("models", exist_ok=True)
with open("models/best_params.pkl", "wb") as f:
    pickle.dump(best_params, f)



with open("metrics/best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)

print("GridSearch completed. Best parameters  saved to models/best_params.pkl")
