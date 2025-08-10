#!/bin/bash
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update && sudo apt upgrade
sudo apt install python3.10-full
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
deactivate
