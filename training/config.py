# training/config.py

"""
Central configuration for CW2 ML project.
This file stores constants used across training scripts.
"""

# Supported feature versions
FEATURE_VERSIONS = ["raw", "minmax", "zscore"]

# Default test_size for train/test split
TEST_SIZE = 0.2

# Random seed for reproducibility
RANDOM_STATE = 42

# Name of the folder where trained models will be saved
MODEL_OUTPUT_DIR = "outputs"

# Classification target column
CLASSIFICATION_TARGET = "time_to_resolve_grouped"

# Regression target column
REGRESSION_TARGET = "time_to_resolve"
