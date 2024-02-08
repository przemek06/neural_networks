from model.model import Model
import numpy as np
from utils import binary_accuracy
from layer.sigmoid_layer import SigmoidLayer

class BinaryClassificationModel(Model):
    
    def predict(self, X):
        X_matrix = X.T
        probabilities = self.forward_propagation(X_matrix)
        return np.round(probabilities).astype(int)

    def calculate_loss(self, y_pred, y):
        n = 1/y_pred.shape[1]
        return -n * (np.matmul(y, np.log(y_pred).T) + np.matmul((1 - y), np.log(1 - y_pred).T))[0][0]
    
    def calculate_dA(self, y_pred, y):
        return - (np.divide(y, y_pred) - np.divide(1 - y, 1 - y_pred))

    def evaluation(self, y_true, y_pred):
        return binary_accuracy(y_true, y_pred)
    
    def last_layer_class(self):
        return SigmoidLayer