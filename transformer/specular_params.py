import cv2
import numpy as np

def specular_params(img):
    '''
    function for specular reflection features extraction
    '''
    img_HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    Lchannel = img_HLS[:,:,1]
    mask = cv2.inRange(Lchannel, 220, 255) / 255
    mask = mask.astype('uint8')
    mask_3 = np.stack((mask, mask, mask), axis=2)
    specular = mask_3 * img
    specular = specular[np.nonzero(specular)]
    r = 100 * specular.size / img.size # specular pixels percentage
    if r == 0:
        m = 0.
        s = 0.
    else:
        m = np.mean(specular) # mean intensity
        s = np.std(specular) # variance

    return np.array((r, m, s))
