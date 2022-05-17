# FedCHAR

This is the repo for Ubicomp 2022 paper under review: " Hierarchical Clustering-based Personalized Federated Learning for Robust and Fair Human Activity Recognition ".

## About

Our code is based on [here](https://github.com/litian96/ditto) with improvements.

### Datasets

We used five datasets for human activity recognition collected by devices of different sensing types.

For IMU, UWB, Depth, HARBox datasets, the full datasets are available for download  [here](https://github.com/xmouyang/FL-Datasets-for-HAR).

For the FMCW dataset, the full dataset can be downloaded [here](https://github.com/DI-HGR/cross_domain_gesture_dataset).

## Data Preparation

1. Under "data/IMU", create the "data/" folder. Then, under "data/IMU/data", create "train/" and "test/" folders respectively (e.g., data/IMU/data/train).
2. Please run "data_pre.py" under "data/IMU/" to generate the "train(test).json" file, the two json files will be stored in "data/IMU/data/train(test)/" respectively.

## Run

In the "FedCHAR/sh/IMU/" folder, run the sh file we provided with the command run IMU_XXX.sh.



Our code will be updated iteratively in the future.

