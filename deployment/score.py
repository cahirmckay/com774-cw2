import json
import joblib
import pandas as pd
import os
from training.preprocessing import select_features


def init():
    """
    Called once when the Azure ML endpoint starts.
    Loads the model into memory.
    """

    global model

    # Azure ML places model files in /var/azureml-app/ or model directory inside the container
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", ""), "model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = joblib.load(model_path)
    print("Model loaded successfully from:", model_path)


def run(raw_data):
    """
    Called for each request to the endpoint.
    """

    try:
        request = json.loads(raw_data)

        if "data" not in request:
            return {"error": "Missing 'data' field in request"}

        feature_version = request.get("feature_version", "raw")

        # Convert incoming records to DataFrame
        df = pd.DataFrame(request["data"])

        # Apply feature selection using the SAME logic as training
        X = select_features(df, feature_version=feature_version)

        # Predict
        predictions = model.predict(X)

        # Convert to Python types for JSON
        predictions = predictions.tolist()

        return {"predictions": predictions}

    except Exception as e:
        return {"error": str(e)}
