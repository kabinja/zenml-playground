#!/bin/bash

zenml stack down
zenml stack set $1
zenml stack up

cd src
python run.py