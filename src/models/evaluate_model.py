import pandas as pd
import pickle
import json
from sklearn.metrics import mean_squared_error, r2_score

# Load test data 
X_test_scaled = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv").values.ravel()

# Load trained model 
with open("models/trained_model.pkl", "rb") as f:
    model = pickle.load(f)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R² score on test set: {r2:.4f}")
print(f"MSE on test set: {mse:.4f}")

metrics = {"r2": round(r2, 4), "mse": round(mse, 4)}
with open("metrics/scores.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Metrics saved to metrics/scores.json")


# Create new dataset with predictions
X_test = pd.read_csv("data/processed_data/X_test.csv")  # original, unscaled
test_dataset_with_predictions = X_test.copy()
test_dataset_with_predictions["real_silica_concentrate"] = y_test
test_dataset_with_predictions["pred_silica_concentrate"] = y_pred

test_dataset_with_predictions.to_csv("data/test_dataset_with_predictions.csv", index=False)

print("Predictions saved to data/test_dataset_with_predictions.csv")
#print(test_dataset_with_predictions.columns)