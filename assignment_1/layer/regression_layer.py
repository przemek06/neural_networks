from .layer import Layer

class RegressionLayer(Layer):   
    def activation(self, Z):
        self._A = Z
        return self._A
        
    def backward_activation(self, dA):
        dZ = dA
        return dZ