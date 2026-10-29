from pathlib import Path
import json

import joblib
import pandas as pd

from src.preprocessing import transform_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load original data
df = pd.read_csv(
    PROJECT_ROOT / "data" / "raw" / "Country-data.csv"
)

# Load saved artifacts
scaler = joblib.load(
    PROJECT_ROOT / "models" / "scaler.joblib"
)

model = joblib.load(
    PROJECT_ROOT / "models" / "kmeans_model.joblib"
)

with open(
    PROJECT_ROOT / "models" / "model_metadata.json",
    "r"
) as file:
    metadata = json.load(file)


# Apply production preprocessing
X_transformed = transform_features(df)

# IMPORTANT: transform, not fit_transform
X_scaled = scaler.transform(X_transformed)

# Predict using loaded K-Means model
predictions = model.predict(X_scaled)


print("Loaded model successfully")

print("\nCluster counts:")
print(
    pd.Series(predictions)
    .value_counts()
    .sort_index()
)

print("\nCluster mapping:")
print(metadata["cluster_mapping"])