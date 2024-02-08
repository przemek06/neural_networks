from .layer import Layer
from utils import xavier_initialization

class RegressionLayer(Layer):   
    def activation(self, Z):
        self._A = Z
        return self._A
        
    def backward_activation(self, dA):
        dZ = dA
        return dZ

    def initialization(self):
        return xavier_initialization