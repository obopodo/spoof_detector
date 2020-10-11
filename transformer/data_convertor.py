import os
import numpy as np
import cv2
from .feature_extractor import extract_features

def img_generator(filenames):
    for i, fname in enumerate(filenames):
        yield cv2.imread(fname)

def read_data(filenames, amount_of_files):
    X = []
    for i, img in enumerate(img_generator(filenames)):
        print(f'\rProcessing {i+1} of {amount_of_files} images ', end='')
        X.append(extract_features(img))
    print('\n')

    return np.array(X)

def convert_train(trainfolder, save_features=True):
    '''
    Read train images set, extract features.
    '''
    real_filenames = [trainfolder + 'real/' + fname for fname in os.listdir(trainfolder + 'real/') if (not fname.startswith('.')) and fname.endswith('.png')]
    spoof_filenames = [trainfolder + 'spoof/' + fname for fname in os.listdir(trainfolder + 'spoof') if (not fname.startswith('.')) and fname.endswith('.png')]
    real_amount = len(real_filenames)
    spoof_amount = len(spoof_filenames)

    X_real = read_data(real_filenames, real_amount)
    X_spoof = read_data(spoof_filenames, spoof_amount)

    print('Total number of files ', real_amount + spoof_amount)
    y_real = np.ones(real_amount, dtype='float')
    y_spoof = np.zeros(spoof_amount, dtype='float')

    X = np.vstack((X_real, X_spoof))
    y = np.append(y_real, y_spoof)

    if save_features:
        np.savetxt('X_y_train.txt', np.hstack((X, y.reshape(-1, 1)))) # save extracted features

    return X, y


def convert_test(testfolder, save_features=True):
    '''
    Read test images set.
    Returns: X - list of images, names - list of filenames
    '''
    names = [fname for fname in os.listdir(testfolder) if (not fname.startswith('.')) and fname.endswith('.png')]
    test_filenames = [testfolder + fname for fname in names]
    test_amount = len(names)

    X_test = read_data(test_filenames, test_amount)

    if save_features:
        np.savetxt('X_test.txt', X_test) # save extracted features

    return X_test, names
