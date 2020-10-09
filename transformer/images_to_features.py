from .read_data import read_train, read_test
from  .feature_extractor import extract_features
import numpy as np

def dataset_to_features(folder, is_train = True, return_images=False, save_features = True, features_already_extracted = False):
    '''
    Extract features from images
    '''
    if not features_already_extracted:
        if is_train:
            imgs, y = read_train(folder) # read images and their classes
        else:
            imgs, y = read_test(folder)

        X = np.array([extract_features(img) for img in imgs]) # extract features from images
        if save_features:
            np.savetxt(folder + 'X_y.txt', np.hstack((X, y.reshape(-1, 1)))) # save extracted features
    else:
        X_y = np.fromfile(folder + 'X_y.txt')
        X, y = X_y[:, :-1], X_y[:, -1]

    if return_images:
        return X, y, imgs
    else:
        return X, y
