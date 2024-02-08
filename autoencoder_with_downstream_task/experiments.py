from model.autoencoder_model import AutoencoderModel
from utils import vector_descaling, plot, save_image, plot_weights, plot_umap, vector_descaling, save_unseen_examples

def batch_size_experiment(X_train, X_val, X_test, y_test):
    lr = 0.02
    batch_sizes = [5, 50, 500]
    epochs = 50
    dimensions = [784, 256, 64, 256, 784]

    loss_data_points_arr = []
    accuracy_data_points_arr = []
    train_loss_data_points_arr = []


    for i, batch_size in enumerate(batch_sizes):
        model = AutoencoderModel(lr, epochs, batch_size, dimensions)
        model.train(X_train, X_train, X_val, X_val)

        loss_data_points_arr.append(model.loss_data_points)
        accuracy_data_points_arr.append(model.accuracy_data_points)
        train_loss_data_points_arr.append(model.train_loss_data_point)

        save_unseen_examples(f"experiment_dims_reconstruction_{i}", model, X_test, 3)
        plot_weights(f"experiment_dims_hidden_layer_repr_{i}.png", model._layers[0]._W)

        pred = model.encode(X_test.T)
        plot_umap(f"experiment_dims_{i}.png", pred.T, y_test)

    plot(loss_data_points_arr, ["batch size = 5", "batch size = 50", "batch size = 500"], "Validation loss", f"experiment_dims_val_loss_{i}")
    plot(accuracy_data_points_arr, ["batch size = 5", "batch size = 50", "batch size = 500"], "Accuracy", f"experiment_dims_accuracy_{i}")
    plot(train_loss_data_points_arr, ["batch size = 5", "batch size = 50", "batch size = 500"], "Train loss", f"experiment_dims_train_loss_{i}")

