#!/bin/bash

echo "Installing Pip"
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
rm get-pip.py filename

PIP_VERSION=$(pip3 --version)
echo "The pip version installed is: ${PIP_VERSION}" 

echo "Installing Virtual Environment Command"
pip install virtualenv

echo "Creating Virtual Environment"
virtualenv glucose_venv
source glucose_venv/bin/activate

echo "Installing requirements.txt into Virtual Environment"
pip install -r requirements.txt

# Run this command in case you want to go back to base environment after activation
# deactivate 