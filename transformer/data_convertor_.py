from .read_data import read_train, read_test
from  .feature_extractor import extract_features
import numpy as np

def data_convertor(folder, is_train = True, save_features = True):
    '''
    Extract features from images, and save them if needed.
    returns X, y as np.arrays. If return_images==True also returns list of images.
    '''

    if is_train:
        imgs, y = read_train(folder) # read images and their classes
    else:
        imgs, y = read_test(folder)

    X = []
    for i, img in enumerate(imgs):
        X.append(extract_features(img)) # extract features from images
        print(f'\rFeatures of {i} image is extracted', end='')
    print('\n')

    X = np.array(X)

    if save_features:
        np.savetxt(folder + 'X_y.txt', np.hstack((X, y.reshape(-1, 1)))) # save extracted features

    if return_images:
        return X, y, imgs
    else:
        return X, y
