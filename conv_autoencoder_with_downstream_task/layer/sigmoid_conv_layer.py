from .layer import Layer
import numpy as np
from utils import xavier_initialization

class SigmoidLayer():

    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))
    
    def forward_propagation(self, Z):
        self._A = self.sigmoid(Z)
        return self._A
        
    def backward_propagation(self, dA):
        dZ = dA * self._A * (1 - self._A)
        return dZ

    def update(self, learning_rate):
        pass