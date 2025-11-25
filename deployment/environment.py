# deployment/environment.py

"""
Defines the Python environment for the deployed endpoint.

This file is used only by Azure ML to understand requirements
when creating the managed online deployment. It is NOT executed locally.
"""

import os

# The scoring environment simply needs to expose a function that Azure ML can read.
# We define a simple helper so the file is not empty.
def get_environment_info():
    return {
        "python_version": "3.10",
        "dependencies_source": "pip",
        "note": "See pipelines/environment.yml for full dependency list."
    }
