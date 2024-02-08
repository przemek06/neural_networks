from .layer import Layer
import numpy as np
from utils import xavier_initialization

class SoftmaxLayer(Layer):

    def __init__(self, in_size, out_size) -> None:
        super().__init__(in_size, out_size)

    def softmax(self, X):
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        
        # Compute softmax probabilities
        softmax_probs = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        
        return softmax_probs
    
    def activation(self, Z):
        self._A = self.softmax(Z.T).T
        return self._A
        
    def backward_activation(self, dA):
        dZ = dA
        return dZ
    
    def initialization(self):
        return xavier_initialization