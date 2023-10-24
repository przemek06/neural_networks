import cv2
import numpy as np
import os
import re 
import pandas as pd

class DataLoader:
    def load(self):
        photo_folder = 'FGNET/images'
        target_width, target_height = 64, 48

        X = []
        y = []

        for filename in os.listdir(photo_folder):
            if filename.endswith(('.JPG')):
                age = int(re.findall("A(\d+)\D?.JPG", filename)[0])
                image = cv2.imread(os.path.join(photo_folder, filename))

                if image is not None:
                    image = cv2.resize(image, (target_width, target_height))

                    if len(image.shape) == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    image_vector = image.flatten()
                    X.append(image_vector)
                    y.append(age)

        X = pd.DataFrame(np.array(X))
        y = pd.DataFrame(np.array(y))
        return X,y

    
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
