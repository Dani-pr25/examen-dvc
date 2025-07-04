import pandas as pd
df = pd.read_csv("data/raw_data/raw.csv")
# Drop the 'date' column
df = df.drop(columns=["date"])
print("First 5 rows:\n", df.head(2))
print("Shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
#print("\nSummary statistics:\n", df.describe())
print("\nMinimum values:\n", df.min())
print("\nMaximum values:\n", df.max())