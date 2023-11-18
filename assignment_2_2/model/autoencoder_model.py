from model.model import Model
from utils import mean_absolute_error
from layer.regression_layer import RegressionLayer
from layer.sigmoid_layer import SigmoidLayer
import numpy as np

class AutoencoderModel(Model):
    
    def sum_weights(self):
        weight_sum = 0

        for layer in self._layers:
            weight_sum += np.sum(np.square(layer._W))
        
        return weight_sum

    def predict(self, X):
        X_matrix = X.T
        return self.forward_propagation(X_matrix, True) 

    def calculate_loss(self, y_pred, y):
        n = 1/y_pred.shape[1]
        return n/2 * ((y - y_pred) ** 2).sum() + self._weight_decay *  self.sum_weights()
    
    def calculate_dA(self, y_pred, y):
        return y_pred - y

    def last_layer_class(self):
        return SigmoidLayer
    
    def evaluation(self, y_true, y_pred):
        return -mean_absolute_error(y_true, y_pred)
    
    def encode(self, X):
        A = X
        for i in range(int(len(self._layers)/2)):
            A = self._layers[i].forward_propagation(A, True)
        return A