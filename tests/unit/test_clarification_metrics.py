from sklearn.metrics import f1_score

def test_classification_metrics_simple():
    y_true = ["a", "b", "c"]
    y_pred = ["a", "b", "c"]

    assert f1_score(y_true, y_pred, average="weighted") == 1
