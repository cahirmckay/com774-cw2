from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def test_regression_metrics_simple():
    y_true = [1, 2, 3]
    y_pred = [1, 2, 3]

    # Perfect predictions
    assert mean_absolute_error(y_true, y_pred) == 0
    assert mean_squared_error(y_true, y_pred) == 0
    assert r2_score(y_true, y_pred) == 1
