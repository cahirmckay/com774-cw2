import argparse
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from azureml.core import Run

from training.preprocessing import load_parquet, prepare_classification_data

__version__ = "1.0.0"


def main(args):
    run = Run.get_context()

    # Load dataset
    df = load_parquet(args.data_path)

    # Preprocess (X, y)
    X, y = prepare_classification_data(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Metrics
    weighted_f1_score = f1_score(y_test, predictions, average="weighted")

    print("Weighted F1 Score:", weighted_f1_score)
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    # Log to Azure ML
    run.log("weighted_f1_score", weighted_f1_score)

    # Save model
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "outputs/classification_model.pkl")
    print("Model saved to outputs/classification_model.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/com_774_dataset.parquet",
        help="Path to the parquet dataset",
    )

    args = parser.parse_args()
    main(args)
