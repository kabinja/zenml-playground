#!/bin/bash

ARTIFACT_STORE=$(zenml artifact-store describe | grep "with id" | cut -d\' -f6)
echo "mlflow points to the artifact store with id '$ARTIFACT_STORE'"
mlflow ui --backend-store-uri /home/renaud/.config/zenml/local_stores/$ARTIFACT_STORE/mlruns