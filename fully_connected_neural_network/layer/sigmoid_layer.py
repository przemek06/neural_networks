from .layer import Layer
import numpy as np
import math

class SigmoidLayer(Layer):
    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))
    
    def activation(self, Z):
        self._A = self.sigmoid(Z)
        return self._A
        
    def backward_activation(self, dA):
        dZ = dA * self._A * (1 - self._A)
        return dZ