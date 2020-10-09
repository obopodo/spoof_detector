import cv2
import numpy as np
import scipy.stats as stats

def chromatic_moments(img):
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = []
    s = []
    skew = []
    max_bin = []
    min_bin = []
    for i in range(3):
        chanel = img_HSV[:,:,i]
        m.append(np.mean(chanel))
        s.append(np.std(chanel))
        skew.append(stats.skew(chanel.flatten()))

        hist = np.histogram(chanel.flatten(), bins=32)
        max_bin.append(hist[1][np.argmax(hist[0])])
        min_bin.append(hist[1][np.argmin(hist[0])])

    return np.array(m + s + skew + min_bin + max_bin)
