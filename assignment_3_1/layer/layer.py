import numpy as np

class Layer:
    def __init__(self, in_size, out_size) -> None:
        self._dW = None
        self._db = None
        self._A_prev = None
        self._Z = None
        initialization = self.initialization()
        self._W = initialization(out_size, in_size)
        self._b = initialization(out_size, 1)
        self._in_size = in_size
        self._out_size = out_size

    # def batch_norm(self, x, epsilon=1e-5):
    #     mean = np.mean(x, axis=1)
    #     variance = np.var(x, axis=1)
    #     x_normalized = (x.T - mean) / np.sqrt(variance + epsilon)
        
    #     return x_normalized.T

    def update(self, learning_rate):
        self._W=self._W - learning_rate*self._dW
        self._b=self._b - learning_rate*self._db

    def forward_propagation(self, A_prev):
        self._A_prev = np.copy(A_prev)
        self._Z = np.matmul(self._W, A_prev)
        self._Z=self._Z+self._b

        return self.activation(self._Z)

    def backward_propagation(self, dA):
        dZ = self.backward_activation(dA)
        n=1/self._A_prev.shape[1]
        self._dW = n*np.matmul(dZ, self._A_prev.T)
        self._db=n*np.sum(dZ, axis=1, keepdims=True)
        return np.matmul(self._W.T, dZ)

    def activation(self, Z):
        pass

    def backward_activation(self, dA):
        pass

    def __repr__(self) -> str:
        return str(self.__class__) + " " + str(self._in_size) + " " + str(self._out_size)

    def __str__(self) -> str:
        return str(self.__class__) + " " + str(self._in_size) + " " + str(self._out_size)
