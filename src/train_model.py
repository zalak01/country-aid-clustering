from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.preprocessing import (
    transform_features,
    MODEL_FEATURES,
    INFLATION_YJ_LAMBDA,
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Country-data.csv"
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load raw training data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Raw data shape:", df.shape)


# --------------------------------------------------
# Apply the same feature transformations
# used during notebook experimentation
# --------------------------------------------------

X_transformed = transform_features(df)

print("Transformed data shape:", X_transformed.shape)


# --------------------------------------------------
# Fit scaler
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_transformed)


# --------------------------------------------------
# Train final K-Means model
# --------------------------------------------------

kmeans_model = KMeans(
    n_clusters=3,
    init="k-means++",
    n_init=10,
    random_state=42,
)

kmeans_model.fit(X_scaled)


# --------------------------------------------------
# Display cluster sizes for verification
# --------------------------------------------------

cluster_counts = (
    pd.Series(kmeans_model.labels_)
    .value_counts()
    .sort_index()
)

print("\nCluster counts:")
print(cluster_counts)


# --------------------------------------------------
# Save fitted objects
# --------------------------------------------------

joblib.dump(
    scaler,
    MODELS_DIR / "scaler.joblib",
)

joblib.dump(
    kmeans_model,
    MODELS_DIR / "kmeans_model.joblib",
)


# --------------------------------------------------
# Save model/preprocessing metadata
# --------------------------------------------------

metadata = {
    "model_type": "KMeans",
    "n_clusters": 3,
    "model_features": MODEL_FEATURES,
    "inflation_yeojohnson_lambda": float(INFLATION_YJ_LAMBDA),

    "transformations": {
        "child_mort": "log1p",
        "exports": "sqrt",
        "health": "none",
        "imports": "sqrt",
        "inflation": "yeo-johnson",
        "life_expec": "none",
        "total_fer": "none",
        "gdpp": "log1p",
    },

    "cluster_mapping": {
        "0": "Intermediate Vulnerability",
        "1": "High Vulnerability",
        "2": "Lower Vulnerability",
    },
}

with open(
    MODELS_DIR / "model_metadata.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metadata, file, indent=4)


print("\nArtifacts saved successfully:")
print(MODELS_DIR / "scaler.joblib")
print(MODELS_DIR / "kmeans_model.joblib")
print(MODELS_DIR / "model_metadata.json")