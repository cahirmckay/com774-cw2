import argparse
import sys
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from azureml.core import Run

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.preprocessing import load_parquet, prepare_classification_data


def main(args):
    print("Starting classification training script.")
    print(f"Data path provided: {args.data_path}")
    print(f"Selected feature version: {args.feature_version}")

    run = Run.get_context()

    # Load the dataset
    print("Loading dataset...")
    df = load_parquet(args.data_path)
    print(f"Dataset successfully loaded with {len(df)} rows.")

    # Preprocess the data
    print("Beginning preprocessing steps...")
    X, y = prepare_classification_data(df, feature_version=args.feature_version)
    print(f"Preprocessing completed. Feature matrix shape: {X.shape}. Target length: {len(y)}")

    # Train-test split
    print("Performing train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Train-test split completed.")

    # Model definition
    print("Initialising the RandomForestClassifier model...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42
    )

    # Training
    print("Training the classification model...")
    model.fit(X_train, y_train)
    print("Model training completed.")

    # Predictions
    print("Generating predictions on the test set...")
    predictions = model.predict(X_test)

    # Metrics
    print("Calculating performance metrics...")
    weighted_f1 = f1_score(y_test, predictions, average="weighted")
    report = classification_report(y_test, predictions)

    print(f"Weighted F1 Score: {weighted_f1}")
    print("Classification Report:")
    print(report)

    # Log to Azure ML
    print("Logging performance metrics to Azure ML...")
    run.log("weighted_f1_score", weighted_f1)
    run.log("feature_version_used", args.feature_version)

    # Save the model
    print("Saving trained model to the outputs folder...")
    os.makedirs("outputs", exist_ok=True)
    model_path = f"outputs/classification_model_v{args.feature_version}.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved successfully to: {model_path}")

    print("Classification training script completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a RandomForest classifier.")

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/com_774_dataset.parquet",
        help="Path to the parquet dataset used for training.",
    )

    parser.add_argument(
        "--feature_version",
        type=str,
        default="raw",
        help="Feature version to use. Options: raw, minmax, zscore.",
    )

    args = parser.parse_args()
    main(args)
