import numpy as np
from utils import he_initialization

alpha = 0.1

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
        A = np.array(Z, copy=True)
        A = np.maximum(A, alpha*A)

        if (A == 0).all():
            print("A zeroed")
        # else:
        #     print("A not zeroed")
        
        return A
    
    def backward_propagation(self, dA):
        dZ = np.array(dA, copy=True)
        mask = (self._Z > 0) + (alpha * (self._Z <= 0))
        dZ = mask * dZ

        if (dZ == 0).all():
            print("dZ zeroed")
        # else:
        #     print("dZ not zeroed")

        return dZ
        
    