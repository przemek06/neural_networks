import numpy as np
import pandas as pd
from utils import pixel_scaling

class DataLoader:
    def load(self):
        # fetch dataset 
        df = pd.read_csv('fashion-mnist_train.csv')

        # data (as pandas dataframes) 
        X = df.iloc[:, 1:] 
        y = df['label'] 
        
        return X, y
    
    def preprocess(self):
        X, y = self.load()
        na_indices = X[X.isna().any(axis=1)].index
        X = X.drop(na_indices)
        y = y.drop(na_indices)
        X = X.apply(pixel_scaling)

        X = X.to_numpy()
        y = y.to_numpy()

        indices = np.arange(X.shape[0])
        indices = indices[::5]
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
