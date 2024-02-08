import numpy as np
import matplotlib.pyplot as plt

def binary_accuracy(y_true, y_pred):
    return (y_true == y_pred).all(axis=0).mean()

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def uniform_initialization(out_size, in_size):
    return np.random.uniform(low=-0.1, high=0.1, size=(out_size, in_size))

def he_initialization(out_size, in_size):
    std_dev = np.sqrt(2.0 / in_size)
    weights = np.random.normal(0, std_dev, size=(out_size, in_size))
    
    return weights

def random_initializaton(out_size, in_size, lower=-0.1, upper=0.1):
    weights = np.random.randn(out_size, in_size) * 0.1
    
    return weights

def plot(data, y_axis, title):
    plt.plot(data)
    plt.xlabel('epoch')
    plt.ylabel(y_axis)
    plt.title(title)
    plt.show()

def generate_layers(layer_sizes, initialization, hidden_layer_class, activation_layer_class):
    layers = []
    for i in range(len(layer_sizes) - 1):
        in_size = layer_sizes[i]
        out_size = layer_sizes[i + 1]
        layers.append(hidden_layer_class(in_size, out_size, initialization))
    
    final_in_size = layer_sizes[-1]
    final_out_size = 1
    layers.append(activation_layer_class(final_in_size, final_out_size, initialization))

    return layers
