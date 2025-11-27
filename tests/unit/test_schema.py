import pandas as pd

def test_dataset_schema():
    """
    Checks that the dataset contains the required columns.
    This prevents training from failing later.
    """

    df = pd.read_parquet("data/com_774_dataset.parquet")

    required_columns = [
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
        "time_to_resolve",
        "time_to_resolve_grouped",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    assert not missing, f"Missing required columns: {missing}"
