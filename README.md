# COM774 Coursework 2 — ML Model Training & Deployment

This repository contains the full machine-learning workflow for training and deploying
a regression and classification models using Azure Machine Learning (Azure ML).


## Features

- Regression model predicting continuous **time_to_resolve**
- Classification model predicting **time_to_resolve_grouped**
- Three feature-version variants: `raw`, `minmax`, `zscore`
- Azure ML Command Jobs with environment reproducibility
- Managed Online Endpoint with scoring script
- Full CI using pytest + ruff

## Setup

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

## Running tests
pytest
ruff check .

## Training
az ml job create -f pipelines/regression_job.yml
az ml job create -f pipelines/classification_job.yml

