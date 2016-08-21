from sklearn import linear_model
from sklearn import svm
import numpy as np
from multiprocessing import Pool

NUM_PROCESSES = 12


def _predict_slave(modelX):
    model, X = modelX
    return model.predict(X)


class ModelFactory(object):
    @staticmethod
    def create_model(model_type, m, **params):
        if model_type == 'SVM':
            params['gamma'] /= m
            return SVM(**params)
        elif model_type in ['LOG', 'LOGHOG']:
            return LogReg(**params)
        else:
            raise Exception('Unrecognized model')


class Model(object):
    def __init__(self, **params):
        raise NotImplementedError('Model needs to be specified')

    def fit(self, X, y):
        self.model.fit(X, y)

    def score(self, X, y):
        return self.model.score(X, y)

    def predict(self, X, chunk=1):
        return self.model.predict(X)
        p = Pool(processes=NUM_PROCESSES)
        print zip([self.model] * (X.shape[0] / chunk), np.array_split(X, chunk))
        y_pred = p.map(_predict_slave, zip([self.model] * (X.shape[0] / chunk), np.array_split(X, chunk)))
        print y_pred
        return y_pred


class SVM(Model):
    def __init__(self, **params):
        self.model = svm.SVC(kernel='poly', **params)


class LogReg(Model):
    def __init__(self, **params):
        self.model = linear_model.LogisticRegression(C=1.0/params['reg'], fit_intercept=False, solver='lbfgs')
