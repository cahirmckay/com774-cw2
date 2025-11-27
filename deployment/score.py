import json
import numpy as np
import joblib
import os

def init():
    global model, class_labels
    
    # Path to the model inside the Azure ML endpoint
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "classification_model_vzscore.pkl")
    
    # Load the model
    model = joblib.load(model_path)

    # The classes in the order used during training
    class_labels = ["0–24h", "1–3d", "3–7d", "7+d"]


def run(raw_data):
    try:
        # Parse input JSON
        data = json.loads(raw_data)

        # Expecting JSON: {"data": [[...], [...]]}
        input_array = np.array(data["data"])

        # Make predictions
        predictions = model.predict(input_array)

        # Convert class index → label
        predicted_labels = [class_labels[int(p)] for p in predictions]

        # Return response
        return {"predictions": predicted_labels}

    except Exception as e:
        return {"error": str(e)}
