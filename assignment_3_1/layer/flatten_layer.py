import numpy as np

class FlattenLayer:

    def update(self, learning_rate):
        pass

    def forward_propagation(self, A_prev):
        self.batch_size, self.channels, self.length, self.length = A_prev.shape
        return A_prev.reshape(self.batch_size, self.channels * self.length * self.length).T

    def backward_propagation(self, dA):
        # print(f"before flatten = {dA.T.shape}")
        reshaped = dA.T.reshape(self.batch_size, self.channels, self.length, self.length)
        # print(f"flattend = {reshaped.shape}")
        return reshaped