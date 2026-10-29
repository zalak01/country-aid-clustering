from flask import Flask, request, jsonify
from pathlib import Path
import json
import joblib
from src.preprocessing import transform_features, MODEL_FEATURES
import pandas as pd


app = Flask(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

print("Model artifacts loaded successfully")


# @app.route("/")
# def home():
#     return "Country Aid API is running"

# @app.route("/health")
# def health():
#     return "API is healthy"

# @app.route("/echo",methods = ["POST"])
# def echo():
#     data = request.get_json()

#     return jsonify({
#         "received_data": data
#     })


@app.route("/predict",methods=["POST"])
def predict():
    data = request.get_json()

    missing_features = [
    feature for feature in MODEL_FEATURES
    if feature not in data
    ]

    if missing_features:
        return jsonify({
        "error": "Missing required features",
        "missing_features": missing_features
    }), 400

    for feature in MODEL_FEATURES:
        try:
            data[feature] = float(data[feature])
        except (TypeError, ValueError):
            return jsonify({
                "error": f"{feature} must be numeric"
            }), 400
        
    
    non_negative_features = [
    "child_mort",
    "exports",
    "health",
    "imports",
    "life_expec",
    "total_fer",
    "gdpp"
    ]

    for feature in non_negative_features:
        if data[feature] < 0:
            return jsonify({
            "error": f"{feature} cannot be negative"
        }), 400

    input_df = pd.DataFrame([data])

    transformed_df = transform_features(input_df)

    scaled_values = scaler.transform(transformed_df)

    cluster = int(model.predict(scaled_values)[0])

    vulnerability = metadata["cluster_mapping"][str(cluster)]

    return jsonify({
        "cluster": cluster,
        "vulnerability": vulnerability
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port =5001)