import json
import numpy as np
import joblib
import os

def init():
    global model, class_labels
    
    # Path to the model inside the Azure ML endpoint
    model_path = os.path.join(
    os.getenv("AZUREML_MODEL_DIR"),
    "outputs",
    "classification_model_vzscore.pkl"
)

    
    # Load the model
    model = joblib.load(model_path)

    # The classes in the order used during training
    class_labels = ["0–24h", "1–3d", "3–7d", "7+d"]


def run(raw_data):
    try:
        # Parse input JSON
        data = json.loads(raw_data)["data"]

        # Convert to numpy array
        X = np.array(data)

        # Get predictions
        preds = model.predict(X)

        # Convert numpy array -> Python list
        preds = preds.tolist()

        # Return predictions as JSON
        return json.dumps({"predictions": preds})

    except Exception as e:
        return json.dumps({"error": str(e)})

