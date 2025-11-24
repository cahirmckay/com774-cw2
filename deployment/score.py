import json
import joblib
import numpy as np
import os



def init():
    """
    Called once when the Azure ML deployment container starts.
    Loads the trained model from the 'model' directory.
    """
    global model

    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", "model"), "model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")


def run(raw_data):
    """
    Handles prediction requests.
    """

    try:
        # Parse JSON string input
        data = json.loads(raw_data)
        input_rows = data.get("data", [])

        if not input_rows:
            return json.dumps({"error": "No input data received"})

        # Convert list of dicts to feature matrix
        import pandas as pd
        df = pd.DataFrame(input_rows)

        # Make predictions
        predictions = model.predict(df)

        # Convert numpy values to native Python types
        predictions = predictions.tolist()

        # Classification model → return predicted class labels
        # Regression model → return numeric values
        return json.dumps({"predictions": predictions})

    except Exception as e:
        return json.dumps({"error": str(e)})
