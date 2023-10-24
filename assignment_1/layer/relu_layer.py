from .layer import Layer
import numpy as np

class ReluLayer(Layer):
    def activation(self, Z):
        return np.maximum(0, Z)
    
    def backward_activation(self, dA):
        dZ = np.array(dA, copy=True)
        dZ[self._Z <= 0] = 0
        return dZ