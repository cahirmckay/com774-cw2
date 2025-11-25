import pandas as pd
import pytest
from training.preprocessing import (
    prepare_regression_data,
    prepare_classification_data,
)

@pytest.mark.parametrize("version", ["raw", "minmax", "zscore"])
def test_preprocessing_regression(version):
    df = pd.read_parquet("data/com_774_dataset.parquet")

    # Use feature version 1 (baseline encoded features)
    X, y = prepare_regression_data(df, feature_version=version)

    assert y.notna().all(), "Regression target contains missing values"
    assert len(X) == len(y), "Feature matrix and target length mismatch"

@pytest.mark.parametrize("version", ["raw", "minmax", "zscore"])
def test_preprocessing_classification(version):
    df = pd.read_parquet("data/com_774_dataset.parquet")

    # Use feature version 1
    X, y = prepare_classification_data(df, feature_version=version)

    assert y.notna().all(), "Classification target contains missing values"
    assert len(X) == len(y), "Feature matrix and target length mismatch"
