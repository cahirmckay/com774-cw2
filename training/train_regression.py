import argparse
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from azureml.core import Run

from training.preprocessing import load_parquet, prepare_regression_data

__version__ = "1.0.0"


def main(args):
    run = Run.get_context()

    # Load dataset
    df = load_parquet(args.data_path)

    # Preprocess (X, y)
    X, y = prepare_regression_data(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Metrics
    mean_absolute_error_value = mean_absolute_error(y_test, predictions)
    mean_squared_error_value = mean_squared_error(y_test, predictions)
    root_mean_squared_error_value = mean_squared_error_value ** 0.5
    r2_score_value = r2_score(y_test, predictions)

    print("Mean Absolute Error:", mean_absolute_error_value)
    print("Root Mean Squared Error:", root_mean_squared_error_value)
    print("R2 Score:", r2_score_value)

    # Log to Azure ML
    run.log("mean_absolute_error", mean_absolute_error_value)
    run.log("root_mean_squared_error", root_mean_squared_error_value)
    run.log("r2_score", r2_score_value)

    # Save model
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "outputs/regression_model.pkl")
    print("Model saved to outputs/regression_model.pkl")


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
