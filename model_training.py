import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from .transformer.data_convertor import convert_train

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, auc, precision_recall_curve, make_scorer
from sklearn.metrics import classification_report

from sklearn.pipeline import make_pipeline, Pipeline
from xgboost import XGBClassifier

import dill

trainfolder = '../Data/train/'

def create_pipeline(scaler, classifier):
    return Pipeline([('scaler', scaler), ('classifier', classifier)])

def PR_AUC_score(y_true, y_pred):
    '''
    precision-recall AUC metric
    '''
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)

def prepare_data(trainfolder, save_features=True):
    X, y = convert_train(trainfolder, save_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2)
    return X_train, y_train, X_test, y_test

def tune_parameters(X_train, y_train):
    cv_strat = StratifiedKFold(5, shuffle=True)
    scaler = MinMaxScaler()
    classifier = XGBClassifier(max_depth=5, learning_rate=0.01, n_estimators=100, gamma=0, \
                               min_child_weight=1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005)

    pipeline = create_pipeline(scaler, classifier)

    tuninig_params = {'classifier__n_estimators': [50, 100, 300, 500],
                      'classifier__max_depth': [3, 5, 10],
                      'classifier__learning_rate': [0.005, 0.01, 0.05]
                     }

    score_metrics = {'F1': make_scorer(f1_score),
                     'PR_AUC': make_scorer(PR_AUC_score)}

    grid_search = RandomizedSearchCV(pipeline, tuninig_params,
                                     scoring=score_metrics,
                                     refit='PR_AUC',
                                     cv=cv_strat, verbose=1, n_iter=8, n_jobs=3)

    grid_search.fit(X_train, y_train)
    return grid_search

def retrain_best_model(grid_search_result, X_train, y_train, X_valid, y_valid, save_model=True):
    best_params = grid_search_result.best_params_
    print('BEST PARAMETERS')
    print(best_params)

    # Retrain estimator:
    classifier_tuned = XGBClassifier(**best_params)
    model_tuned = create_pipeline(scaler, classifier_tuned)

    X = np.vstack((X_train, X_valid))
    y = np.append(y_train, y_valid)

    model_tuned.fit(X, y)

    if save_model:
        with open('pretrained.model', 'wb') as ouf:
            dill.dump(model_tuned, ouf)

    return model_tuned
