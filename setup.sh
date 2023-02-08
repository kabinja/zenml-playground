#!/bin/bash

python -m pip install --upgrade pip

pip install "zenml[server]"
zenml integration install airflow mlflow
pip install apache-airflow-providers-docker
pip install scikit-learn

zenml experiment-tracker register mlflow_tracker  --flavor=mlflow
zenml orchestrator register local_airflow  --flavor=airflow --local=True

zenml stack register local_airflow_stack \
      -a default \
      -o local_airflow \
      -e mlflow_tracker

zenml stack register local_stack \
      -a default \
      -o default \
      -e mlflow_tracker