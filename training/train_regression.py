import argparse
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from azureml.core import Run
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.preprocessing import load_parquet, prepare_regression_data


def main(args):
    print("Starting regression training script.")
    print(f"Data path provided: {args.data_path}")
    print(f"Selected feature version: {args.feature_version}")

    run = Run.get_context()

    # Load the dataset
    print("Loading dataset...")
    df = load_parquet(args.data_path)
    print(f"Dataset successfully loaded with {len(df)} rows.")

    # Preprocess the data
    print("Beginning preprocessing steps...")
    X, y = prepare_regression_data(df, feature_version=args.feature_version)
    print(f"Preprocessing completed. Feature matrix shape: {X.shape}. Target length: {len(y)}")

    # Split the data
    print("Performing train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Train-test split completed.")

    # Define the regression model
    print("Initialising the RandomForestRegressor model...")
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42
    )

    # Train the model
    print("Training the regression model...")
    model.fit(X_train, y_train)
    print("Model training completed.")

    # Make predictions
    print("Generating predictions on the test set...")
    predictions = model.predict(X_test)

    # Calculate metrics
    print("Calculating performance metrics...")
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")

    # Log to Azure ML
    print("Logging performance metrics to Azure ML...")
    run.log("mean_absolute_error", mae)
    run.log("mean_squared_error", mse)
    run.log("r2_score", r2)
    run.log("feature_version_used", args.feature_version)

    # Save the model
    print("Saving trained model to the outputs folder...")
    os.makedirs("outputs", exist_ok=True)

    model_filename = f"regression_model_v_{args.feature_version}.pkl"
    model_path = os.path.join("outputs", model_filename)

    joblib.dump(model, model_path)
    print(f"Model saved successfully to: {model_path}")

    print("Regression training script completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a RandomForest regression model.")

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
