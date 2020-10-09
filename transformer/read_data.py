import os
import numpy as np
import cv2

def read_multiple(filenames):
    number_of_files = len(filenames)
    X = []
    for i, fname in enumerate(filenames):
        X.append(cv2.imread(fname))
        print(f'\rReading {i} of {number_of_files} images', end='')
    print('\n')
    return X, number_of_files

def read_train(trainfolder):
    '''
    Read train images set.
    Returns: X - list of images, y - np.array of images classes
    '''
    real_filenames = [trainfolder + 'real/' + fname for fname in os.listdir(trainfolder + 'real/') if (not fname.startswith('.')) and fname.endswith('.png')]
    spoof_filenames = [trainfolder + 'spoof/' + fname for fname in os.listdir(trainfolder + 'spoof') if (not fname.startswith('.')) and fname.endswith('.png')]

    X_real, real_amount = read_multiple(real_filenames)
    X_spoof, spoof_amount = read_multiple(spoof_filenames)

    print('Total number of files ', real_amount + spoof_amount)
    y_real = np.ones(real_amount, dtype='float')
    y_spoof = np.zeros(spoof_amount, dtype='float')

    X = X_real + X_spoof
    y = np.append(y_real, y_spoof)

    return X, y


def read_test(testfolder):
    '''
    Read test images set.
    Returns: X - list of images, names - list of filenames
    '''
    names = [fname for fname in os.listdir(testfolder) if (not fname.startswith('.')) and fname.endswith('.png')]

    test_filenames = [testfolder + fname for fname in names]
    X_test, _ = read_multiple(test_filenames)

    return X_test, names
