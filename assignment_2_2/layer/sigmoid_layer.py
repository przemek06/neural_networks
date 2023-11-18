from .layer import Layer
import numpy as np
from utils import xavier_initialization

class SigmoidLayer(Layer):

    def __init__(self, in_size, out_size) -> None:
        super().__init__(in_size, out_size)

    def get_dropout_matrix(self, n):
        identity_matrix = np.eye(n, dtype=int)
        mask = np.random.rand(n, n) < 0.25
        noisy_matrix = identity_matrix * (1 - mask)

        return noisy_matrix

    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))
    
    def activation(self, Z, eval):
        self._A = self.sigmoid(Z)
        if not eval:
            U = self.get_dropout_matrix(self._A.shape[0])
            self._A = np.dot(self._A.T, U).T
        return self._A
        
    def backward_activation(self, dA):
        dZ = dA * self._A * (1 - self._A)
        return dZ
    
    def initialization(self):
        return xavier_initialization