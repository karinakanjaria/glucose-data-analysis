#!/bin/bash

echo "Installing Virtual Environment Command"
pip install virtualenv

echo "Creating Virtual Environment"
virtualenv glucose_venv
source glucose_venv/bin/activate

echo "Installing requirements.txt into Virtual Environment"
pip install -r requirements.txt

sudo apt update
sudo apt install openjdk-17-jre-headless -y
ipython kernel install --name "glucose-venv" --user

# Run this command in case you want to go back to base environment after activation
# deactivate 15555