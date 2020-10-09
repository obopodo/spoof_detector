import os
import numpy as np
import cv2

def read_train(trainfolder):
    real_filenames = [trainfolder + 'real/' + fname for fname in os.listdir(trainfolder + 'real/') if (not fname.startswith('.')) and fname.endswith('.png')]
    spoof_filenames = [trainfolder + 'spoof/' + fname for fname in os.listdir(trainfolder + 'spoof') if (not fname.startswith('.')) and fname.endswith('.png')]

    X_real = [cv2.imread(fname) for fname in real_filenames]
    X_spoof = [cv2.imread(fname) for fname in spoof_filenames]

    y_real = np.ones(len(real_filenames), dtype='float')
    y_spoof = np.zeros(len(spoof_filenames), dtype='float')

    X = X_real + X_spoof
    y = np.append(y_real, y_spoof)

    return X, y


def read_test(testfolder):
    names = [fname for fname in os.listdir(testfolder) if (not fname.startswith('.')) and fname.endswith('.png')]

    test_filenames = [testfolder + fname for fname in names]
    X_test = [cv2.imread(fname) for fname in real_filenames]

    return X_test, names
