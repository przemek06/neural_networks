from model.conv_autoencoder_model import ConvAutoencoderModel
from utils import vector_descaling, plot, save_image, plot_weights, plot_umap, vector_descaling, save_unseen_examples, plot_kernel_weights
import numpy as np

def experiment(X_train, X_val, X_test, y_test):
    lr = 0.001
    batch_size = 50
    epochs = 50
    
    dims = [([144, 512, 784], [(1, 4, 3), (4, 4, 2)]), ([288, 512, 784], [(1, 8, 3), (8, 8, 2)]), ([288, 512, 784], [(1, 16, 3), (16, 8, 2)])]

    loss_data_points_arr = []
    accuracy_data_points_arr = []
    train_loss_data_points_arr = []

    side_length = int(X_train.shape[1] ** (1/2))

    X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 1, side_length, side_length))
    X_val_reshaped = np.reshape(X_val, (X_val.shape[0], 1, side_length, side_length))
    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, side_length, side_length))

    for i, dim in enumerate(dims):
        model = ConvAutoencoderModel(lr, epochs, batch_size, dim[0], dim[1])
        model.train(X_train_reshaped, X_train, X_val_reshaped, X_val)

        loss_data_points_arr.append(model.loss_data_points)
        accuracy_data_points_arr.append(model.accuracy_data_points)
        train_loss_data_points_arr.append(model.train_loss_data_point)

        after_side_length = int(((side_length - dim[1][0][2] + 1)/2 - dim[1][1][2] + 1)/2)

        save_unseen_examples(f"experiment_dims_reconstruction_{i}", model, X_test_reshaped, X_test, 3, after_side_length)
        # plot_weights(f"experiment_dims_hidden_layer_repr_{i}.png", model._layers[0]._W)
        plot_kernel_weights(f"kernelweights_for_dim_1", model._layers[0]._W)

        pred = model.encode(X_test_reshaped)
        plot_umap(f"experiment_dims_{i}.png", pred.T, y_test)

    plot(loss_data_points_arr, ["dims 1", "dims 2", "dims 3"], "Validation loss", f"experiment_dims_val_loss_{i}")
    plot(accuracy_data_points_arr, ["dims 1", "dims 2", "dims 3"], "Accuracy", f"experiment_dims_accuracy_{i}")
    plot(train_loss_data_points_arr, ["dims 1", "dims 2", "dims 3"], "Train loss", f"experiment_dims_train_loss_{i}")

