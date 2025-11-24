import importlib


def test_regression_script_imports():
    """
    Ensures the regression training script can be imported
    without execution errors.
    """
    module = importlib.import_module("training.train_regression")
    assert hasattr(module, "main"), "train_regression.py must define a main() function"


def test_regression_version_variable():
    """
    Ensures the regression training script defines a version string.
    """
    module = importlib.import_module("training.train_regression")

    version = getattr(module, "__version__", None)
    assert version is not None, "train_regression.py should define __version__"
    assert isinstance(version, str), "__version__ must be a string"
