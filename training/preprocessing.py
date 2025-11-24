import pandas as pd


def load_parquet(path: str) -> pd.DataFrame:
    """
    Loads a parquet dataset from disk.
    """
    try:
        df = pd.read_parquet(path)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load parquet file: {e}")


def select_features(df: pd.DataFrame):
    """
    Selects the feature columns used for both training and inference.

    We use encoded categorical features (from CW1) and simple numeric features.
    This avoids scaling and keeps preprocessing simple.
    """

    feature_cols = [
        "reassignment_count",
        "reopen_count",
        "sys_mod_count",
        "impact_encoded",
        "urgency_encoded",
        "priority_encoded",
        "contact_type_encoded",
        "opened_hour",
        "opened_month",
        "opened_weekday_encoded",
    ]

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    return df[feature_cols]


def prepare_classification_data(df: pd.DataFrame):
    """
    Prepares X and y for classification.
    Target: time_to_resolve_grouped
    """

    target_col = "time_to_resolve_grouped"

    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    # Remove rows with missing target
    df = df.dropna(subset=[target_col])

    X = select_features(df)
    y = df[target_col]

    return X, y


def prepare_regression_data(df: pd.DataFrame):
    """
    Prepares X and y for regression.
    Target: time_to_resolve (numeric)
    """

    target_col = "time_to_resolve"

    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    df = df.dropna(subset=[target_col])

    X = select_features(df)
    y = df[target_col]

    return X, y
