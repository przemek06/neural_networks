import numpy as np

class MaxPoolingLayer:

    def __init__(self, pool_size = (2, 2), stride = 2) -> None:
        self.pool_size = pool_size
        self.stride = stride

    def update(self, learning_rate):
        pass

    def forward_propagation(self, A_prev):
        
        batch_size, channels, input_height, input_width = A_prev.shape
        pool_height, pool_width = self.pool_size

        output_height = (input_height - pool_height) // self.stride + 1
        output_width = (input_width - pool_width) // self.stride + 1

        A = np.zeros((batch_size, channels, output_height, output_width))
        mask = np.zeros_like(A_prev)

        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + pool_height
                w_start = j * self.stride
                w_end = w_start + pool_width

                local_region = A_prev[:, :, h_start:h_end, w_start:w_end] + np.random.rand(batch_size, channels, pool_height, pool_width)/10000
# 
                max_values = np.max(local_region, axis=(2, 3))

                A[:, :, i, j] = max_values
                mask[:, :, h_start:h_end, w_start:w_end] = (local_region == max_values[:, :, None, None])

        self.mask = mask
        return A

    def backward_propagation(self, dA):
        # print(f"max poooling dA = {dA.shape}")
        _, _, output_height, output_width = dA.shape
        pool_height, pool_width = self.pool_size

        dX = np.zeros_like(self.mask)

        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + pool_height
                w_start = j * self.stride
                w_end = w_start + pool_width

                mask_slice = self.mask[:, :, h_start:h_end, w_start:w_end]

                dX[:, :, h_start:h_end, w_start:w_end] += dA[:, :, i, j][:, :, None, None] * mask_slice

        # print(f"max poooling dX = {dX.shape}")

        return dX