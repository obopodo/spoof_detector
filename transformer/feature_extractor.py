from .img_prerprocessor import img_preprocessor
# from .specular_params import specular_params
# from .blurriness import *
# from .chromatic_moments import chromatic_moments
# from .color_diversity import color_diversity

from .features import *

def extract_features(img):
    img_norm = img_preprocessor(img)

    specular_features = specular_features(img_norm)
    blurriness_features = np.array([blurriness_feature_1(img_norm), blurriness_feature_2(img_norm)])
    chromatic_features = chromatic_moments(img_norm)
    color_features = color_diversity(img_norm)

    return np.concatenate((specular_features, blurriness_features, chromatic_features, color_features))
#
# def extract_features(img):
#     img_norm = img_preprocessor(img)
#
#     specular_features = specular_params(img_norm)
#     blurriness_features = np.array([Laplacian_blur_feature(img_norm), Low_pass_blur_feature(img_norm)])
#     chromatic_features = chromatic_moments(img_norm)
#     color_features = color_diversity(img_norm)
#
#     return np.concatenate((specular_features, blurriness_features, chromatic_features, color_features))
