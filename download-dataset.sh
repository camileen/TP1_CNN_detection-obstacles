#!/bin/bash

# Download the dataset zip from Kaggle
if [ ! -d /mnt/c/Users/byoub/Code/INSA-Lyon/IAT/TP1_CNN_detection-obstacles/dataset/ ]; then
  curl -L \
  -o /mnt/c/Users/byoub/Downloads/obstacles-dataset.zip \
    https://www.kaggle.com/api/v1/datasets/download/idrisskh/obstacles-dataset

  # Unzip the dataset
  unzip /mnt/c/Users/byoub/Downloads/obstacles-dataset.zip \
    -d /mnt/c/Users/byoub/Code/INSA-Lyon/IAT/TP1_CNN_detection-obstacles

  # Rename dataset folder in project directory
  mv /mnt/c/Users/byoub/Code/INSA-Lyon/IAT/TP1_CNN_detection-obstacles/obstacles\ dataset \
    /mnt/c/Users/byoub/Code/INSA-Lyon/IAT/TP1_CNN_detection-obstacles/dataset/
fi


