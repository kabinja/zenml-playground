#!/bin/bash

rm -rf ~/.config/zenml

python -m pip install --upgrade pip

pip install "zenml[server]"
zenml integration install airflow mlflow kubeflow -y
pip install apache-airflow-providers-docker
pip install scikit-learn

zenml init
zenml up --docker

zenml experiment-tracker register mlflow_tracker  --flavor=mlflow
zenml orchestrator register local_airflow  --flavor=airflow --local=True
zenml orchestrator register local_kubeflow --flavor=kubeflow --kubernetes_context=mlops
zenml image-builder register local_builder --flavor=local

zenml stack register local_airflow_stack \
      -a default \
      -o local_airflow \
      -e mlflow_tracker \
      -i local_builder

zenml stack register local_kubeflow_stack \
      -a default \
      -o local_kubeflow \
      -e mlflow_tracker \
      -i local_builder

zenml stack register local_stack \
      -a default \
      -o default \
      -e mlflow_tracker \
      -i local_builder

