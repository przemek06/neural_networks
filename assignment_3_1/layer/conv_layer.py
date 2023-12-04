import numpy as np
import sys

class ConvLayer:

    def __init__(self, in_size, out_size, kernel_size) -> None:
        self._dW = None
        self._db = None
        self._A_prev = None
        self._Z = None
        self._in_size = in_size
        self._out_size = out_size
        self.kernel_size = kernel_size
        self._W = self.kernel_initialization(out_size, in_size, kernel_size)
        self._b = self.initialization(out_size)

    def add_out_dimension(self, original_tensor, out):
        original_shape = original_tensor.shape

        reshaped_tensor = np.reshape(original_tensor, (original_shape[0], 1, *original_shape[1:]))

        new_tensor = np.tile(reshaped_tensor, (1, out, 1, 1, 1, 1, 1))

        final_shape = (original_shape[0], out, *original_shape[1:])
        new_tensor = np.reshape(new_tensor, final_shape)
        return new_tensor
    
    
    def add_in_dimension(self, original_tensor, _in):
        original_shape = original_tensor.shape

        reshaped_tensor = np.reshape(original_tensor, (original_shape[0], original_shape[1], 1, *original_shape[2:]))

        new_tensor = np.tile(reshaped_tensor, (1, 1, _in, 1, 1, 1, 1))

        final_shape = (original_shape[0], original_shape[1], _in, *original_shape[2:])
        new_tensor = np.reshape(new_tensor, final_shape)
        return new_tensor

    def convolve_images_with_kernels(self, images, kernels):
        out, _in, n, _ = kernels.shape

        # if _in != images.shape[1]:
        #     raise Exception("Dimensions of input and kernels are not the same.")
        
        modified_images = self.add_out_dimension(images, out)

        windows = np.lib.stride_tricks.sliding_window_view(modified_images, (n, n), axis=(-1, -2))

        feature_maps = np.sum(windows * kernels.reshape(1, out ,_in, 1, 1, n, n), axis=(-5, -2, -1))
        return np.reshape(feature_maps, (feature_maps.shape[0], out, *feature_maps.shape[2:]))
    
    def forward_propagation(self, A_prev):
        self._A_prev = A_prev
        if self._W.shape[1] != A_prev.shape[1]:
            raise Exception("Dimensions of input and kernels are not the same.")
    
        reshaped_b = np.reshape(self._b, (self._b.shape[0], 1, 1))
        return self.convolve_images_with_kernels(A_prev, self._W) + reshaped_b
    
    def backward_propagation(self, dA):

        self._dW = self.calculate_dW(dA)
        self._db = self.calculate_db(dA)
        dX = self.calculate_dX(dA)

        # print(f"dW = {self._dW.shape}")
        # print(f"db = {self._db.shape}")
        # print(f"dX = {dX.shape}")

        return dX



    # NOT SURE ABOUT SHAPES
    def calculate_dW(self, dA):
        reshaped_dA = np.reshape(dA, (dA.shape[0], self._out_size, 1, dA.shape[-2], dA.shape[-1]))
        # print(f"dW reshaped dA = {reshaped_dA.shape}")

        batch_size, out, _in, n, _ = reshaped_dA.shape

        # if _in != images.shape[1]:
        #     raise Exception("Dimensions of input and kernels are not the same.")
        
        modified_images = self.add_out_dimension(self._A_prev, out)

        windows = np.lib.stride_tricks.sliding_window_view(modified_images, (n, n), axis=(-1, -2))

        feature_maps = np.sum(windows * reshaped_dA.reshape(batch_size, out, 1, 1, 1, n, n), axis=(0, -2, -1))
        return np.reshape(feature_maps, (out, self._in_size, self.kernel_size, self.kernel_size)) / batch_size

    def calculate_db(self, dA):
        return np.sum(dA, axis = (0, -1, -2)) / dA.shape[0]

    # NOT SURE ABOUT SHAPES
    def calculate_dX(self, dA):
        rotated_kernels = self.rotate_180(self._W)
        padded_dA = self.pad_images(dA, self.kernel_size - 1)

        reshaped_dA = self.add_in_dimension(padded_dA, self._in_size)
        windows = np.lib.stride_tricks.sliding_window_view(reshaped_dA, (self.kernel_size, self.kernel_size), axis=(-1, -2))

        feature_maps = np.sum(windows * rotated_kernels.reshape(1, self._out_size, self._in_size, 1, 1, self.kernel_size, self.kernel_size), axis=(-6, -2, -1))
        return np.reshape(feature_maps, (feature_maps.shape[0], self._in_size, *feature_maps.shape[2:]))


    def update(self, learning_rate):
        self._W=self._W - learning_rate*self._dW
        self._b=self._b - learning_rate*self._db

        if (self._dW == 0).all():
            print("dW zeroed")

        if (self._db == 0).all():
            print("db zeroed")

    def kernel_initialization(self, out_size, in_size, kernel_size):
        boundary = np.sqrt(2.0)
        kernel = np.random.normal(loc=0.0, scale=boundary, size=(out_size, in_size, kernel_size, kernel_size))

        return kernel

    def initialization(self, out_size):
        boundary = np.sqrt(2.0)
        biases = np.random.normal(0, boundary, (out_size))
        return biases
    
    def pad_images(self, tensor, n):
        padded_tensor = np.pad(tensor, ((0, 0), (0, 0), (n, n), (n, n)), mode='constant', constant_values=0)
        return padded_tensor
    
    def rotate_180(self, matrix):
        rotated_matrix = np.rot90(matrix, 2, axes=(-1, -2))
        return rotated_matrix