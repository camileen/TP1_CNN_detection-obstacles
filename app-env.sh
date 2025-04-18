#!/bin/bash

# Create log directory
mkdir -p /mnt/c/Users/byoub/Code/INSA-Lyon/IAT/TP1_CNN_detection-obstacles/logs/fit
mkdir -p /mnt/c/Users/byoub/Code/INSA-Lyon/IAT/TP1_CNN_detection-obstacles/results

# Create .venv directory
mkdir -p /mnt/c/Users/byoub/Code/INSA-Lyon/IAT/TP1_CNN_detection-obstacles/.venv
cd /mnt/c/Users/byoub/Code/INSA-Lyon/IAT/TP1_CNN_detection-obstacles/.venv
python3 -m venv iat-tp1 # create a virtual environment named iat-tp1
