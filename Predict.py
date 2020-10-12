import numpy as np
from transformer.data_convertor import convert_test
import sys

def predict_test(model=None, data_path='', extracted=False):
    '''
    Make predictions with existing model.
    PARAMETERS:
        model: sklearn classifier or pipeline.
        data_path: path to images dataset or to file with extracted features.
        extracted: if True searches for 'X_test.txt' file with extracted features from images dataset
                   and 'X_test_filenames.txt' with files names list.
    '''

    if not model:
        import joblib
        print('Please input full relative path to the model: <dir>/<filename>')
        model_path = input()
        model = joblib.load(model_path) # load model from file

    if not extracted:
        X_test, filenames = convert_test(data_path)
    else:
        X_test = np.fromfile(data_path + 'X_test.txt', sep='\t').reshape(-1, 121)
        with open(data_path + 'X_test_filenames.txt', 'r') as inf:
            filenames = inf.read().splitlines()

    pred_classes = model.predict(X_test)
    pred_probs = model.predict_proba(X_test)

    with open('results_on_test.txt', 'wt') as ouf:
        for name, prob in zip(filenames, pred_probs[:, 1]):
            ouf.write(name + ',' + str(prob) + '\n')


if __name__ == '__main__':
    path = sys.argv[1]
    print(path)
    predict_test(model=None, data_path=path, extracted=True)
