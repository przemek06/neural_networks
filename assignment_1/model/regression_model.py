from model.model import Model
from utils import mean_absolute_error
from layer.regression_layer import RegressionLayer

class RegressionModel(Model):
    
    def predict(self, X):
        X_matrix = X.T
        return self.forward_propagation(X_matrix)

    def calculate_loss(self, y_pred, y):
        n = 1/y_pred.shape[1]
        return n/2 * ((y - y_pred) ** 2).sum()
    
    def calculate_dA(self, y_pred, y):
        return y_pred - y

    def last_layer_class(self):
        return RegressionLayer
    
    def evaluation(self, y_true, y_pred):
        return -mean_absolute_error(y_true, y_pred)