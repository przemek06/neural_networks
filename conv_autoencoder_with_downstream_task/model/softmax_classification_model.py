from model.model import Model
import numpy as np
from utils import binary_accuracy
from layer.softmax_layer import SoftmaxLayer

class SoftmaxClassificationModel(Model):
    
    def predict(self, X):
        X_matrix = X.T
        probabilities = self.forward_propagation(X_matrix)
        return np.argmax(probabilities.T, axis=1)

    def calculate_loss(self, y_pred, y):
        y_true = np.eye(y_pred.shape[0])[y]
        y_pred = y_pred.T
        return -1/len(y_true) * np.sum(np.sum(y_true * np.log(y_pred + 10**-100)))
    
    def calculate_dA(self, y_pred, y):
        y_true = np.eye(y_pred.shape[0])[y]
        return y_pred - y_true.T

    def evaluation(self, y_true, y_pred):
        return binary_accuracy(y_true, y_pred)
    
    def last_layer_class(self):
        return SoftmaxLayer