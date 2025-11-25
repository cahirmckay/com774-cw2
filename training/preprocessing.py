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


RAW_NUMERIC = [
    "reassignment_count",
    "reopen_count",
    "sys_mod_count",
]

MINMAX_NUMERIC = [
    "reassignment_count_minmax",
    "reopen_count_minmax",
    "sys_mod_count_minmax",
]

ZSCORE_NUMERIC = [
    "reassignment_count_zscore",
    "reopen_count_zscore",
    "sys_mod_count_zscore",
]

ENCODED_CATEGORICAL = [
    "impact_encoded",
    "urgency_encoded",
    "priority_encoded",
    "contact_type_encoded",
    "opened_hour",
    "opened_month",
    "opened_weekday_encoded",
]



def select_features(df: pd.DataFrame, feature_version: str):
    """
    Selects the feature columns based on the requested feature version.

    feature_version can be:
      - 'raw'
      - 'minmax'
      - 'zscore'
    """

    if feature_version == "raw":
        numeric_cols = RAW_NUMERIC
    elif feature_version == "minmax":
        numeric_cols = MINMAX_NUMERIC
    elif feature_version == "zscore":
        numeric_cols = ZSCORE_NUMERIC
    else:
        raise ValueError(
            f"Invalid feature_version '{feature_version}'. "
            f"Must be one of: raw, minmax, zscore."
        )

    feature_cols = numeric_cols + ENCODED_CATEGORICAL

    # Check all cols exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns for version '{feature_version}': {missing}")

    return df[feature_cols]



def prepare_classification_data(df: pd.DataFrame, feature_version: str):
    """
    Prepares X and y for classification.
    Target: time_to_resolve_grouped
    """

    target_col = "time_to_resolve_grouped"

    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    df = df.dropna(subset=[target_col])

    X = select_features(df, feature_version)
    y = df[target_col]

    return X, y


def prepare_regression_data(df: pd.DataFrame, feature_version: str):
    """
    Prepares X and y for regression.
    Target: time_to_resolve (numeric)

    NOTE:
    Target is ALWAYS raw time_to_resolve.
    Only input features vary by version.
    """

    target_col = "time_to_resolve"

    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    df = df.dropna(subset=[target_col])

    X = select_features(df, feature_version)
    y = df[target_col]

    return X, y
