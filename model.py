import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from transformer import feature_extractor, read_data

from sklearn.preprocessing import MinMaxScaler , StandardScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import classification_report

from sklearn.pipeline import make_pipeline, Pipeline
from xgboost import XGBClassifier

import dill

trainfolder = '../Data/train/'

features_already_extracted = False
if not features_already_extracted:
    imgs, y = read_data.read_train(trainfolder) # read images and their classes
    X = np.array([feature_extractor.extract_features(img) for img in imgs]) # extract features from images
    np.savetxt(trainfolder + 'X_y.txt', np.hstack((X, y.reshape(-1, 1)))) # save extracted features
else:
    X_y = np.fromfile(trainfolder + 'X_y.txt')
    X, y = X_y[:, :-1], X_y[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2)

cv_strat = StratifiedKFold(5, shuffle=True)
scaler = MinMaxScaler()
classifier = XGBClassifier(max_depth=5, learning_rate=0.01, n_estimators=100, gamma=0, \
                           min_child_weight=1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005)

def create_pipeline(scaler, classifier):
    return Pipeline([('scaler', scaler), ('classifier', classifier)])

pipeline = create_pipeline(scaler, classifier)

tuninig_params = {'n_estimators': [50, 100, 300, 500],
                  'max_depth': [3, 5, 10],
                  'learning_rate': [0.005, 0.01, 0.05]
                 }

grid_search = RandomizedSearchCV(pipeline, tuninig_params, scoring=...,
                                 cv=cv_strat, verbose=1, n_iter=8, n_jobs=3)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(best_params)

# Retrain estimator:

classifier_tuned = XGBClassifier(**best_params)
pipeline_tuned = create_pipeline(scaler, classifier_tuned)

pipeline_tuned.fit()

with open('pretrained_model.model', 'wb') as ouf:
    dill.dump(pipeline_tuned, ouf)
