import numpy as np
import matplotlib.pyplot as plt
import umap

def save_unseen_examples(name, model, X_test_reshaped, X_test, n, side_length):
    for i in range(n):
        X_reshaped_sample = X_test_reshaped[i]
        X_reshaped_sample = X_reshaped_sample.reshape(1, *X_reshaped_sample.shape)
        X_pred = model.predict(X_reshaped_sample)

        expected = vector_descaling(X_test[i])
        actual = vector_descaling(X_pred)

        save_image(expected, f"{name}_expected_{i}.png")
        save_image(actual, f"{name}_actual_{i}.png")

        X_encoded = model.encode(X_reshaped_sample)
        map_number = int(len(X_encoded)/(side_length*side_length))
        X_encoded_reshaped = np.reshape(X_encoded, (map_number, side_length, side_length))

        for j, feature_map in enumerate(X_encoded_reshaped):
            feature_map_reshaped = np.reshape(feature_map, (side_length*side_length))
            feature_map_descaled = vector_descaling(feature_map_reshaped)
            save_image(feature_map_descaled, f"{name}_encoded_{i}_{j}.png")



def min_max_scaling(column):
    min_val = column.min()
    max_val = column.max()
    return (column - min_val) / (max_val - min_val)

def pixel_scaling(column):
    return column/255

def vector_descaling(vector):
    return vector * 255/np.max(vector)

def binary_accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()    

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def uniform_initialization(out_size, in_size):
    return np.random.uniform(low=-0.1, high=0.1, size=(out_size, in_size))

def xavier_initialization(out_size, in_size):
    boundary = np.sqrt(6 / (out_size + in_size))
    weights = np.random.uniform(-boundary, boundary, (out_size, in_size))
    return weights

def he_initialization(out_size, in_size):
    boundary = np.sqrt(2.0 / in_size)
    weights = np.random.normal(0, boundary, (out_size, in_size))
    
    return weights

def random_initializaton(out_size, in_size, lower=-0.1, upper=0.1):
    weights = np.random.randn(out_size, in_size) * 0.1
    
    return weights

def show_image(vector):
    matrix_size = int(vector.shape[0] ** (1/2))
    matrix = vector.reshape((matrix_size, matrix_size))
    plt.imshow(matrix, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis('off')
    plt.show()

def save_image(vector, name):
    matrix_size = int(vector.shape[0] ** (1/2))
    matrix = vector.reshape((matrix_size, matrix_size))
    plt.imshow(matrix, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis('off') 
    plt.savefig(f"images/{name}.png", bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()

def plot(data, labels, y_axis, title):

    for d, label in zip(data, labels):
        plt.plot(d, label = label)

    plt.xlabel('Epoch')
    plt.ylabel(y_axis)
    plt.title(title)
    plt.legend()
    plt.savefig(f"images/{title}.png", bbox_inches='tight', pad_inches=0, transparent=False)
    plt.show()

def generate_layers(layer_sizes, hidden_layer_class, activation_layer_class):
    layers = []
    for i in range(len(layer_sizes) - 2):
        in_size = layer_sizes[i]
        out_size = layer_sizes[i + 1]
        layers.append(hidden_layer_class(in_size, out_size))
    
    final_in_size = layer_sizes[-2]
    final_out_size = layer_sizes[-1]
    layers.append(activation_layer_class(final_in_size, final_out_size))

    return layers

def generate_conv_layers(layer_sizes, activation_layer_class, conv_layer_class, pooling_layer_class, flatten_layer_class):
    layers = []

    for layer_size in layer_sizes:
        conv_layer = conv_layer_class(layer_size[0], layer_size[1], layer_size[2])
        activation_layer = activation_layer_class()
        max_pooling_layer = pooling_layer_class()

        layers.append(conv_layer)
        layers.append(activation_layer)
        layers.append(max_pooling_layer)

    flatten_layer = flatten_layer_class()
    layers.append(flatten_layer)

    return layers

def pad_images(tensor, n):
    padded_tensor = np.pad(tensor, ((0, 0), (0, 0), (n, n), (n, n)), mode='constant', constant_values=1)
    return padded_tensor
    
def plot_kernel_weights(title, weights):
    weights = min_max_scaling(weights)
    padded_kernels = pad_images(weights, 1)
    out, _in, padded_kernel_size, _ = padded_kernels.shape 

    larger_matrix_width = _in * padded_kernel_size
    larger_matrix_height = out * padded_kernel_size
    larger_matrix = np.zeros((larger_matrix_height, larger_matrix_width))

    for i in range(out):
        for j in range(_in):
            larger_matrix[i * (padded_kernel_size): (i + 1) * (padded_kernel_size), j * (padded_kernel_size): (j + 1) * (padded_kernel_size)] = padded_kernels[i, j]

    reversed_color_matrix = 255 - larger_matrix
    plt.imshow(reversed_color_matrix, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis('off') 
    plt.savefig(f"images/{title}", bbox_inches='tight', pad_inches=0, transparent=True, dpi = 960)
    plt.show()

def plot_weights(title, weights):
    normalized_weights = min_max_scaling(weights) * 255

    shape_of_neurons_sqr_root = int(normalized_weights.shape[0] ** 0.5)
    num_of_neurons_sqr_root = int(normalized_weights.shape[1] ** 0.5)
    line_width = 2

    square_matrices = normalized_weights.reshape((-1, shape_of_neurons_sqr_root, shape_of_neurons_sqr_root))
    padded_square_matrices = np.zeros((num_of_neurons_sqr_root * num_of_neurons_sqr_root, shape_of_neurons_sqr_root + line_width, shape_of_neurons_sqr_root + line_width))

    padded_shape_of_neurons_sqr_root = shape_of_neurons_sqr_root + line_width

    for i in range(num_of_neurons_sqr_root * num_of_neurons_sqr_root):
        padded_square_matrices[i, :, :] = np.pad(square_matrices[i, :, :], ((0, line_width), (0, line_width)), mode='constant', constant_values=0)

    larger_matrix_length = padded_shape_of_neurons_sqr_root * num_of_neurons_sqr_root
    larger_matrix = np.zeros((larger_matrix_length, larger_matrix_length))

    for i in range(num_of_neurons_sqr_root):
        for j in range(num_of_neurons_sqr_root):
            larger_matrix[i * (padded_shape_of_neurons_sqr_root): (i + 1) * (padded_shape_of_neurons_sqr_root), j * (padded_shape_of_neurons_sqr_root): (j + 1) * (padded_shape_of_neurons_sqr_root)] = padded_square_matrices[i * num_of_neurons_sqr_root + j]

    reversed_color_matrix = 255 - larger_matrix
    plt.imshow(reversed_color_matrix, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis('off') 
    plt.savefig(f"images/{title}", bbox_inches='tight', pad_inches=0, transparent=True, dpi = 960)
    plt.show()

def plot_umap(name, X, y):
    trans = umap.UMAP(n_neighbors=5, random_state=42).fit(X)
    plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s= 5, c=y, cmap='Spectral')
    plt.title('Embedding of the training set by UMAP', fontsize=24)
    plt.savefig(f"images/{name}", bbox_inches='tight', pad_inches=0, transparent=False, dpi = 960)
    plt.show()