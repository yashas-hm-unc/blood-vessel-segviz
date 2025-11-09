#!/bin/bash

echo "Creating virtual Environment"
python -m venv venv

echo "Activating Environment"
source ./venv/bin/activate

echo "Installing dependencies"
pip install -r requirements.txt