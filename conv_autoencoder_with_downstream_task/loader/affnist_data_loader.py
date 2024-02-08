import numpy as np
import pandas as pd
from utils import pixel_scaling
import scipy.io as spio

def loadmat(filename):
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _todict(matobj):
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def _check_keys(dict):
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict      

class DataLoader:
    def load(self):
        dataset = loadmat('1.mat')

        y = dataset['affNISTdata']['label_int']
        X = dataset['affNISTdata']['image']
        
        return pd.DataFrame(X.T), pd.DataFrame(y)
    
    def preprocess(self):
        X, y = self.load()
        na_indices = X[X.isna().any(axis=1)].index
        X = X.drop(na_indices)
        y = y.drop(na_indices)
        X = X.apply(pixel_scaling)

        X = X.to_numpy()
        y = y.to_numpy()

        indices = np.arange(X.shape[0])
        indices = indices[::20]
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        return self.split_data(X, y)

    def split_data(self, X, y):
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))

        X_train = X[:train_size]
        y_train = y[:train_size]

        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]

        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]

        return X_train, y_train, X_val, y_val, X_test, y_test
