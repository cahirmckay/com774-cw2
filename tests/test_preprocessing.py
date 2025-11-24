import pandas as pd
from training.preprocessing import (
    prepare_regression_data,
    prepare_classification_data,
)


def test_preprocessing_regression():
    df = pd.read_parquet("data/com_774_dataset.parquet")
    X, y = prepare_regression_data(df)

    assert y.notna().all(), "Regression target contains missing values"
    assert len(X) == len(y), "Feature matrix and target length mismatch"


def test_preprocessing_classification():
    df = pd.read_parquet("data/com_774_dataset.parquet")
    X, y = prepare_classification_data(df)

    assert y.notna().all(), "Classification target contains missing values"
    assert len(X) == len(y), "Feature matrix and target length mismatch"
