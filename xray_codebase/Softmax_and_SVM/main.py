import time
import itertools
from sklearn import cross_validation
from multiprocessing import Pool

from models import ModelFactory
import utils

MODEL = 'LOG'
PARAMS = {
    # 'SVM': {'C': [.01, .05, .1, .5, 1.0, 5.0, 10.0, 50.0, 100.0], 'gamma': [5.0, 10.0, 50.0, 100.0, 1000.0, 10000.0], 'degree': [2, 3, 4, 5, 6]},
    # 'SVM': {'C': [5.0], 'gamma': [50]},  # RBF best params
    # 'SVM': {'C': [5.0], 'gamma': [10000.0], 'degree': [4]},  # polySVM best params, accuracy ~54%
    'SVM': {'C': [5.0, 10.0, 50.0, 100.0, 500.0], 'gamma': [10.0, 50.0, 100.0, 500.0, float(1e4)], 'degree': [3, 4]},
    'LOG': {'reg': [1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10, 3e10]},
    'LOGHOG': {'reg': [1, 5, 10, 20, 30]},
}

TRAIN_TOTAL = 30000  # max 30000
TEST_TOTAL = 300000  # max 300000
HOG = True
VALSIZE = 0.20
PPC = 4
NUM_PROCESSES = 12


def _choose_slave(args):
    # prob not threadsafe
    X, y, X_val, y_val, model_type, m, params = args
    model = ModelFactory.create_model(model_type, m, **params)
    model.fit(X, y)
    return (model.score(X_val, y_val), model, params)


def choose_best_model(X, X_val, y, y_val, model_type, params):
    p = Pool(processes=NUM_PROCESSES)
    scores = p.map(_choose_slave, [(X, y, X_val, y_val, model_type, X_train.shape[0], dict(zip(params.keys(), ps))) for ps in itertools.product(*params.values())])
    best_acc, best_model, best_params = max(scores)
    return best_params, best_acc, best_model


if __name__ == '__main__':
    t = time.time()
    X, y = utils.load_hog(PPC, TRAIN_TOTAL) if HOG else utils.load_images(TRAIN_TOTAL)
    print 'Done loading', str(time.time() - t)
    t = time.time()

    # split data
    X_train, X_val, y_train, y_val = cross_validation.train_test_split(X, y, test_size=VALSIZE)
    print 'Done splitting train', str(time.time() - t)
    t = time.time()

    # sweep hyperparameters field and pick best model
    params, acc, model = choose_best_model(X_train, X_val, y_train, y_val, MODEL, PARAMS[MODEL])
    # if not SVM:
    #     reg, acc, model = find_regularization(X_train, X_val, y_train, y_val, REGS_HOG if HOG else REGS)
    #     print "Best regularization strength is: " + str(reg)
    # else:
    #     params, acc, model = choose_best_model(X_train, X_val, y_train, y_val, MODEL, PARAMS[MODEL])
    print "Best params are " + str(params)
    print "Accuracy was: " + str(acc)
    print 'Done training model', str(time.time() - t)
    t = time.time()

    X_test, _ = utils.load_hog(PPC, TEST_TOTAL, False) if HOG else utils.load_images(TEST_TOTAL)
    print 'Done loading test', str(time.time() - t)
    t = time.time()
    y_test = model.predict(X_test, chunk=2)
    print 'Done predicting test', str(time.time() - t)
    t = time.time()
    utils.format_output(y_test)
    print 'Done writing test', str(time.time() - t)
    t = time.time()
