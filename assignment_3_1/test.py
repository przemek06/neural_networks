# import numpy as np

# # Set the random seed for reproducibility
# np.random.seed(42)

# # Define the shape parameters
# b = 30
# n = 16  # You can choose any value for n

# # Generate a random tensor of shape (b, n)
# random_tensor = np.random.rand(b, n)

# # Reshape it into shape (b, 1, m, m), where m is the square root of n
# m = int(np.sqrt(n))
# reshaped_tensor = random_tensor.reshape(b, 1, m, m)

# # Reshape it back into shape (b, n)
# original_shape = reshaped_tensor.reshape(b, n)

# # Print the results
# print("Random Tensor:")
# print(random_tensor)
# print("\nReshaped Tensor (b, 1, m, m):")
# print(reshaped_tensor)
# print("\nOriginal Shape (b, n):")
# print(original_shape)

# print(random_tensor.shape == original_shape.shape)


# import numpy as np

# def add_dimension(original_tensor, m):
#     original_shape = original_tensor.shape

#     reshaped_tensor = np.reshape(original_tensor, (original_shape[0], 1, *original_shape[1:]))

#     new_tensor = np.tile(reshaped_tensor, (1, m, 1, 1, 1, 1, 1))

#     final_shape = (original_shape[0], m, *original_shape[1:])
#     new_tensor = np.reshape(new_tensor, final_shape)
#     return new_tensor

# def convolve_images_with_kernels(images, kernels):
#     out, _in, n, _ = kernels.shape
    
#     modified_images = add_dimension(images, out)

#     windows = np.lib.stride_tricks.sliding_window_view(modified_images, (n, n), axis=(-1, -2))

#     feature_maps = np.sum(windows * kernels.reshape(1, out * _in, 1, 1, 1, n, n), axis=(-2, -1))
#     return np.reshape(feature_maps, (feature_maps.shape[0], out, *feature_maps.shape[2:]))

# # Example usage
# batch_size = 1
# length = 10
# _in = 1
# n = 3
# out = 2



# # Create random input images and kernels for testing
# images = np.asarray([[[[ 2,  2,  3,  4], [ 5,  6,  7,  8], [ 9, 10, 11, 12], [13, 14, 15, 16]], [[ 1,  2,  3,  4], [ 5,  6,  7,  8], [ 9, 10, 11, 12], [13, 14, 15, 16]], [[ 1,  2,  3,  4], [ 5,  6,  7,  8], [ 9, 10, 11, 12], [13, 14, 15, 16]]], [[[ 1,  2,  3,  4], [ 5,  6,  7,  8], [ 9, 10, 11, 12], [13, 14, 15, 16]], [[ 1,  2,  3,  4], [ 5,  6,  7,  8], [ 9, 10, 11, 12], [13, 14, 15, 16]], [[ 1,  2,  3,  4], [ 5,  6,  7,  8], [ 9, 10, 11, 12], [13, 14, 15, 16]]]])
# # kernels = np.random.rand(out, _in, n, n)
# kernels = np.asarray([[[[1, 1], [1, 1]]], [[[3, 3], [3, 3]]]])
# print(kernels.shape)

# # Perform convolution
# result = convolve_images_with_kernels(images, kernels)

# # Print the result shape
# print(result)
# print(result.shape)

# import numpy as np

# def max_pooling_forward(input_data, pool_size=(2, 2), stride=2):
#     batch_size, channels, input_height, input_width = input_data.shape
#     pool_height, pool_width = pool_size

#     output_height = (input_height - pool_height) // stride + 1
#     output_width = (input_width - pool_width) // stride + 1

#     output_data = np.zeros((batch_size, channels, output_height, output_width))
#     mask = np.zeros_like(input_data)

#     for i in range(output_height):
#         for j in range(output_width):
#             h_start = i * stride
#             h_end = h_start + pool_height
#             w_start = j * stride
#             w_end = w_start + pool_width

#             # Extract the local region
#             local_region = input_data[:, :, h_start:h_end, w_start:w_end]

#             # Perform max pooling
#             max_values = np.max(local_region, axis=(2, 3))

#             # Store the max values in the output and update the mask
#             output_data[:, :, i, j] = max_values
#             mask[:, :, h_start:h_end, w_start:w_end] = (local_region == max_values[:, :, None, None])

#     return output_data, mask

# def max_pooling_backward(d_output, mask, pool_size=(2, 2), stride=2):
#     _, _, output_height, output_width = d_output.shape
#     pool_height, pool_width = pool_size

#     d_input = np.zeros_like(mask)

#     for i in range(output_height):
#         for j in range(output_width):
#             h_start = i * stride
#             h_end = h_start + pool_height
#             w_start = j * stride
#             w_end = w_start + pool_width

#             # Extract the corresponding local region in the mask
#             mask_slice = mask[:, :, h_start:h_end, w_start:w_end]

#             # Distribute the gradient to the positions of the max values in the mask
#             d_input[:, :, h_start:h_end, w_start:w_end] += d_output[:, :, i, j][:, :, None, None] * mask_slice

#     return d_input

# # Example usage
# batch_size = 2
# m = 3
# n = 4
# tensor = np.asarray([[[[1, 2, 3, 4], [5,6, 7, 8], [9, 10, 11, 12], [13, 14, 15,16]]]])

# pooled_result, mask = max_pooling_forward(tensor)
# print(tensor)
# print(pooled_result)
# print(mask)
# dA = np.asarray([[[[1, 2], [3, 4]]]])
# b = max_pooling_backward(dA, mask)
# print(b)
# # print("Binary Mask Shape:", binary_mask.shape)


# import numpy as np

# Z = np.asarray([[-1, -2, -3], [-1, 2, 5], [6, -6, 10]])

# dZ = np.asarray([[1, 1, 1], [2, 2, 2], [3, 3, 4]])
# dZ[Z < 0] = 0
# print(dZ)

# import numpy as np

# Z = np.asarray([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]], [[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]])

# b = np.asarray([1, 2])
# reshaped_b = np.reshape(b, (b.shape[0], 1, 1))
# print(Z + reshaped_b)

import numpy as np

# def pad_images(tensor, n):
#     batch_size, channels, length, _ = tensor.shape

#     # Pad each image in the tensor
#     padded_tensor = np.pad(tensor, ((0, 0), (0, 0), (n, n), (n, n)), mode='constant', constant_values=0)

#     return padded_tensor

# X = np.asarray([[[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]], [[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]])

# padded_X = pad_images(X, 2)
# print(padded_X)

# def rotate_180(matrix):
#     rotated_matrix = np.rot90(matrix, 2, axes=(-1, -2))
#     return rotated_matrix

# X = np.asarray([[[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[1, 1, 1], [2, 2, 2], [3, 3, 3]]], [[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[1, 1, 1], [2, 2, 2], [3, 3, 3]]]])

# print(rotate_180(X))
stride = 2
pool_size = (2, 2)

def forward_propagation(A_prev):
        
        batch_size, channels, input_height, input_width = A_prev.shape
        pool_height, pool_width = pool_size

        output_height = (input_height - pool_height) // stride + 1
        output_width = (input_width - pool_width) // stride + 1

        A = np.zeros((batch_size, channels, output_height, output_width))
        mask = np.zeros_like(A_prev)

        for i in range(output_height):
            for j in range(output_width):
                h_start = i * stride
                h_end = h_start + pool_height
                w_start = j * stride
                w_end = w_start + pool_width

                local_region = A_prev[:, :, h_start:h_end, w_start:w_end]

                max_values = np.max(local_region, axis=(2, 3))

                A[:, :, i, j] = max_values
                mask[:, :, h_start:h_end, w_start:w_end] = (local_region == max_values[:, :, None, None])

        mask = mask
        return A, mask

def backward_propagation(dA, mask):
        # print(f"max poooling dA = {dA.shape}")
        _, _, output_height, output_width = dA.shape
        pool_height, pool_width = pool_size

        dX = np.zeros_like(mask)

        for i in range(output_height):
            for j in range(output_width):
                h_start = i * stride
                h_end = h_start + pool_height
                w_start = j * stride
                w_end = w_start + pool_width

                mask_slice = mask[:, :, h_start:h_end, w_start:w_end]

                dX[:, :, h_start:h_end, w_start:w_end] += dA[:, :, i, j][:, :, None, None] * mask_slice

        # print(f"max poooling dX = {dX.shape}")

        return dX
X = np.asarray([[[[00.0, 2.0, 3.0, 4.0], [50.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]], [[55.0, 2.0, 3.0, 4.0], [50.0, 6.0, 70.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]], [[[1.0, 2.0, 3.0, 4.0], [50.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]], [[1.0, 2.0, 3.0, 4.0], [50.0, 6.0, 70.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]])

A, mask = forward_propagation(X)
print(A)
print(mask)

dA = np.asarray([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[11, 22], [33, 44]], [[55, 66], [77, 88]]]])
dX = backward_propagation(dA, mask)
print(dX)
# def add_out_dimension_2(original_tensor, out):
#         original_shape = original_tensor.shape

#         reshaped_tensor = np.reshape(original_tensor, (original_shape[0], 1, *original_shape[1:]))

#         new_tensor = np.tile(reshaped_tensor, (1, out, 1, 1, 1, 1, 1))

#         final_shape = (original_shape[0], out, *original_shape[1:])
#         new_tensor = np.reshape(new_tensor, final_shape)
#         return new_tensor

# def add_out_dimension(tensor, m):
#     batch_size, in_size, n, _ = tensor.shape

#     # Reshape to (batch_size, 1, in_size, n, n)
#     tensor = tensor[:, np.newaxis, :, :, :]

#     # Replicate along the new axis
#     tensor = np.tile(tensor, (1, m, 1, 1, 1))

#     # Reshape to the final shape
#     tensor = tensor.reshape(batch_size, m, in_size, n, n)

#     return tensor

# def add_in_dimension_2(original_tensor, _in):
#         original_shape = original_tensor.shape

#         reshaped_tensor = np.reshape(original_tensor, (original_shape[0], original_shape[1], 1, *original_shape[2:]))

#         new_tensor = np.tile(reshaped_tensor, (1, 1, _in, 1, 1, 1, 1))

#         final_shape = (original_shape[0], original_shape[1], _in, *original_shape[2:])
#         new_tensor = np.reshape(new_tensor, final_shape)
#         return new_tensor

# def add_in_dimension(tensor, in_size):
#     batch_size, out_size, n, _ = tensor.shape

#     tensor = tensor[:, :, np.newaxis, :, :]

#     tensor = np.tile(tensor, (1, 1, in_size, 1, 1))

#     tensor = tensor.reshape(batch_size, out_size, in_size, n, n)

#     return tensor

# def rotate_180(tensor):
#         rotated_tensor = np.rot90(tensor, 2, axes=(-1, -2))
#         return rotated_tensor

# X = np.asarray([[[[00.0, 2.0, 3.0, 4.0], [50.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]], [[55.0, 2.0, 3.0, 4.0], [50.0, 6.0, 70.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]], [[[1.0, 2.0, 3.0, 4.0], [50.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]], [[1.0, 2.0, 3.0, 4.0], [50.0, 6.0, 70.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]])

# print(add_in_dimension(X, 3))