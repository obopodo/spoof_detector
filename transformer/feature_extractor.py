from .normalize_img import normalize
from .features import *

def extract_features(img):
    img_norm = normalize(img)

    specular_f = specular_features(img_norm)
    blurriness_f = np.array([blurriness_feature_1(img_norm), blurriness_feature_2(img_norm)])
    chromatic_f = chromatic_moments(img_norm)
    color_f = color_diversity(img_norm)

    return np.concatenate((specular_f, blurriness_f, chromatic_f, color_f))
