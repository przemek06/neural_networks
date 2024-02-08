from .layer import Layer
import numpy as np
import math

class TanhLayer(Layer):
    def tanh(self, Z):
        return np.tanh(Z)
    
    def activation(self, Z):
        self._A = self.tanh(Z)
        return self._A
        
    def backward_activation(self, dA):
        
        dZ = dA * (1 - self._A ** 2)
        return dZ