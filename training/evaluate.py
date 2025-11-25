# training/evaluate.py

"""
Shared evaluation functions for both regression and classification models.
"""

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    f1_score,
    classification_report,
)


def evaluate_regression(y_true, y_pred):
    """
    Returns regression metrics as a dictionary.
    """
    return {
        "mean_absolute_error": mean_absolute_error(y_true, y_pred),
        "mean_squared_error": mean_squared_error(y_true, y_pred),
        "r2_score": r2_score(y_true, y_pred),
    }


def evaluate_classification(y_true, y_pred):
    """
    Returns F1 + text report.
    """
    return {
        "weighted_f1_score": f1_score(y_true, y_pred, average="weighted"),
        "classification_report": classification_report(y_true, y_pred),
    }
