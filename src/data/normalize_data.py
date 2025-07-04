import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Paths
processed_path = "data/processed_data"
x_train_path = os.path.join(processed_path, "X_train.csv")
x_test_path = os.path.join(processed_path, "X_test.csv")

# Load data
X_train = pd.read_csv(x_train_path)
X_test = pd.read_csv(x_test_path)

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Save normalized data
X_train_scaled.to_csv(os.path.join(processed_path, "X_train_scaled.csv"), index=False)
X_test_scaled.to_csv(os.path.join(processed_path, "X_test_scaled.csv"), index=False)

print("Normalization completed and files saved.")