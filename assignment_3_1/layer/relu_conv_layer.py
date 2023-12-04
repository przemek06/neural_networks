import numpy as np
from utils import he_initialization

class ReluConvLayer():

    # def batch_norm(self, x, epsilon=1e-5):
    #     mean = np.mean(x, axis=0)
    #     variance = np.var(x, axis=0)
    #     x_normalized = (x - mean) / np.sqrt(variance + epsilon)
        
    #     return x_normalized


    def update(self, learning_rate):
        pass
    
    def forward_propagation(self, Z):
        self._Z = Z
        return np.maximum(0, self._Z)
    
    def backward_propagation(self, dA):
        dZ = np.array(dA, copy=True)
        dZ[self._Z <= 0] = 0
        return dZ
    