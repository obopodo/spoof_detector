import cv2
import numpy as np
import scipy.stats as stats

def specular_features(img):
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

def blurriness_feature_1(img):
    '''
    Convolves grayscale image with Laplacian kernel and takes the variance
    (Laplacian blur)
    '''
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()


def blurriness_feature_2(img):
    '''
    blurriness Metric from Frédérique Crété-Roffet article.
    Low pass filter applied to rows/columns of pixels.
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

### STILL DON'T QUITE UNDERSTAND WHICH COLOR DIVERSITY METRIC IS BETTER
### HIST OR JUST TOP 100 DISTINCT COLORS. NOW DISTINCT COLORS ARE USED

def color_diversity(img):
    quantized = (img // 8)
    quantized = quantized.reshape(-1, 3)

    colors_distribution = np.unique(quantized, axis=0, return_counts=True)
    n_distinct_colors = colors_distribution[0].shape[0]
    top_100_colors_counts = np.sort(colors_distribution[1])[-100:]
    result = np.append(top_100_colors_counts, n_distinct_colors)
    return result

# def color_diversity_hist(img):
#     quantized = (img // 8)
#     channels = [quantized[:,:,i].flatten() for i in range(3)]
#     img_flatten = channels[0] + 32*channels[1] + 32*32*channels[2]
#     n_distinct_colors = np.unique(img_flatten).shape[0]
#     bins_counts = np.sort(np.histogram(img_flatten, bins=100)[0])
#     result = np.append(bins_counts, n_distinct_colors)
#     return result
