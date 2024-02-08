import numpy as np
import matplotlib.pyplot as plt
import umap

def save_unseen_examples(name, model, X_test, n):
    for i in range(n):
        X_pred = model.predict(X_test)

        expected = vector_descaling(X_test[i])
        actual = vector_descaling(X_pred.T[i])

        save_image(expected, f"{name}_expected_{i}.png")
        save_image(actual, f"{name}_actual_{i}.png")

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
    matrix_size = int(np.sqrt(len(vector)))
    matrix = vector.reshape((matrix_size, matrix_size))
    plt.imshow(matrix, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis('off')
    plt.show()

def save_image(vector, name):
    matrix_size = int(np.sqrt(len(vector)))
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
    plt.colorbar()
    plt.show()