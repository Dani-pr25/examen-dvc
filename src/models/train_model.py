import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor  # or your model

# Load training data
X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv")  # 
y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()  # ensure it's a 1D array

# Load best parameters
with open("models/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

# Train the model
model = RandomForestRegressor(**best_params)
model.fit(X_train_scaled, y_train)

from sklearn.metrics import mean_squared_error, r2_score

# === Evaluate on training data ===
y_pred = model.predict(X_train_scaled)

r2 = r2_score(y_train, y_pred)
mse = mean_squared_error(y_train, y_pred)

print(f"Training R² score: {r2:.4f}")
print(f"Training MSE: {mse:.4f}")

# Save trained model
with open("models/trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Trained model saved to models/trained_model.pkl")