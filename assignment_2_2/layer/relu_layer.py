from .layer import Layer
import numpy as np
from utils import he_initialization

class ReluLayer(Layer):

    def get_dropout_matrix(self, n):
        identity_matrix = np.eye(n, dtype=int)
        mask = np.random.rand(n, n) < 0.1
        noisy_matrix = identity_matrix * (1 - mask)

        return noisy_matrix
    
    def activation(self, Z, eval):
        A = np.maximum(0, Z)
        # print(A.shape)
        if not eval:
            U = self.get_dropout_matrix(A.shape[0])
            A = np.dot(A.T, U).T
        # print(A)
        # print(A)
        return A
    
    def backward_activation(self, dA):
        dZ = np.array(dA, copy=True)
        dZ[self._Z <= 0] = 0
        return dZ
    
    def initialization(self):
        return he_initialization