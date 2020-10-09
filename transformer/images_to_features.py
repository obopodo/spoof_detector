from .read_data import read_train, read_test
from  .feature_extractor import extract_features
import numpy as np

def dataset_to_features(folder, is_train = True, features_already_extracted = False):
    if not features_already_extracted:
        if is_train:
            imgs, y = read_train(folder) # read images and their classes
            np.savetxt(trainfolder + 'X_y.txt', np.hstack((X, y.reshape(-1, 1)))) # save extracted features
        else:
            imgs, y = read_test(folder)

        X = np.array([extract_features(img) for img in imgs]) # extract features from images

    else:
        X_y = np.fromfile(trainfolder + 'X_y.txt')
        X, y = X_y[:, :-1], X_y[:, -1]

    return X, y
