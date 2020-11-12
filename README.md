# spoof-detector
Helper functions module for face spoof detection algorithm  
Module is still in development  
To run example type in terminal `python Predict.py extracted_data/`, path to pretrained model is simple `pretrained.model`

Dependencies:
* numpy
* scipy
* sklearn
* joblib
* OpenCV
* xgboost

File `model_training.py` contains all of necessary steps (functions) for model training.  
Folder `transformer` contains scripts for picture normalization and feature exraction.
