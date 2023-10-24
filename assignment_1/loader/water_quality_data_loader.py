from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd

class DataLoader:
    def load(self):
        # fetch dataset 
        df = pd.read_csv('water_potability.csv')

        # data (as pandas dataframes) 
        X = df.iloc[:, 0:9] 
        y = df['Potability'] 
        
        return X, y
    
    def min_max_scaling(self, column):
        min_val = column.min()
        max_val = column.max()
        return (column - min_val) / (max_val - min_val)

    def preprocess(self):
        X, y = self.load()
        na_indices = X[X.isna().any(axis=1)].index
        X = X.drop(na_indices)
        y = y.drop(na_indices)
        X = X.apply(self.min_max_scaling)

        X = X.to_numpy()
        y = y.to_numpy()

        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        y = (y > 0).astype(int)
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
