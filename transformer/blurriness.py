import cv2
import numpy as np

def Laplacian_blur_feature(img):
    '''
    Convolves grayscale image with Laplacian kernel and takes the variance
    '''
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()


def Low_pass_blur_feature(img):
    '''
    Blur Metric from Frédérique Crété-Roffet article
    '''
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel_factor = 9
    hh = (1/kernel_factor) * np.array([[1]*kernel_factor]) # horizontal blur kernel
    hv = hh.T # vertical blur kernel
    Bh = cv2.filter2D(img_gray, -1, hh) # img blurred horizontally
    Bv = cv2.filter2D(img_gray, -1, hv) # img blurred vertically

    D_Fh = np.abs(img_gray[:, 1:] - img_gray[:, :-1])
    D_Fv = np.abs(img_gray[1:, :] - img_gray[:-1, :])
    D_Bh = np.abs(Bh[:, 1:] - Bh[:, :-1])
    D_Bv = np.abs(Bv[1:, :] - Bv[:-1, :])

    zeros_h = np.zeros_like(D_Fh)
    zeros_v = np.zeros_like(D_Fv)

    D_Vh = np.maximum(D_Fh - D_Bh, zeros_h)
    D_Vv = np.maximum(D_Fv - D_Bv, zeros_v)

    b_Fv = 1. - np.sum(D_Vv) / np.sum(D_Fv)
    b_Fh = 1. - np.sum(D_Vh) / np.sum(D_Fh)

    return max(b_Fh, b_Fv)
