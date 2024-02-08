from .layer import Layer
import numpy as np
from utils import xavier_initialization

class SigmoidLayer(Layer):

    def __init__(self, in_size, out_size) -> None:
        super().__init__(in_size, out_size)

    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))
    
    def activation(self, Z):
        self._A = self.sigmoid(Z)
        return self._A
        
    def backward_activation(self, dA):
        dZ = dA * self._A * (1 - self._A)
        return dZ
    
    def initialization(self):
        return xavier_initialization