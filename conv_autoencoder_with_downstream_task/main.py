from loader.fashion_mnist_data_loader import DataLoader
from model.conv_autoencoder_model import ConvAutoencoderModel
from model.conv_autoencoder_model import ConvAutoencoderModel
from selector.selector import Selector
from utils import vector_descaling, plot, save_image, plot_weights, plot_umap, vector_descaling, save_unseen_examples
from experiments import experiment
import numpy as np
from model.softmax_classification_model import SoftmaxClassificationModel

def no_autoencoder_execution(X_train, y_train, X_val, y_val, X_test, y_test):
    model = SoftmaxClassificationModel(0.01, 50, 200, [784, 512, 256, 10])
    model.train(X_train, y_train, X_val, y_val)

    # plot([model.loss_data_points], [""], "Validation loss", "no_autoencoder_val_loss")
    # plot([model.accuracy_data_points], [""], "Accuracy", "no_autoencoder_accuracy")
    # plot([model.train_loss_data_point], [""], "Train loss", "no_autoencoder_train_loss")

    test_pred = model.predict(X_test)
    test_accuracy = model.evaluation(y_test, test_pred)
    print(f"Test accuracy with no autoencoder = {test_accuracy}")
    return model.loss_data_points, model.accuracy_data_points, model.train_loss_data_point

def autoencoder_execution(X_train, y_train, X_val, y_val, X_test, y_test):
    side_length = int(X_train.shape[1] ** (1/2))

    X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 1, side_length, side_length))
    X_val_reshaped = np.reshape(X_val, (X_val.shape[0], 1, side_length, side_length))
    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, side_length, side_length))

    autoencoder = ConvAutoencoderModel(0.001, 100, 200, [144, 512, 784], [(1, 8, 3), (8, 4, 2)])
    autoencoder.train(X_train_reshaped, X_train, X_val_reshaped, X_val)

    # pred = autoencoder.encode(X_test_reshaped)
    # plot_umap("main_model_umap.png", pred.T, y_test)

    X_train_encoded = autoencoder.encode(X_train_reshaped).T
    X_val_encoded = autoencoder.encode(X_val_reshaped).T
    X_test_encoded = autoencoder.encode(X_test_reshaped).T

    model = SoftmaxClassificationModel(0.01, 50, 200, [144, 256, 10])
    model.train(X_train_encoded, y_train, X_val_encoded, y_val)

    # plot([model.loss_data_points], [""], "Validation loss", "autoencoder_val_loss")
    # plot([model.accuracy_data_points], [""], "Accuracy", "autoencoder_accuracy")
    # plot([model.train_loss_data_point], [""], "Train loss", "autoencoder_train_loss")

    test_pred = model.predict(X_test_encoded)
    test_accuracy = model.evaluation(y_test, test_pred)
    print(f"Test accuracy with autoencoder = {test_accuracy}")
    return model.loss_data_points, model.accuracy_data_points, model.train_loss_data_point

def standard_execution(X_train, X_val, X_test, y_test):
    # selector = Selector()
    # lr, batch_size, dims = selector.select_best_params(AutoencoderModel, X_train, X_val)

    side_length = int(X_train.shape[1] ** (1/2))

    X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 1, side_length, side_length))
    X_val_reshaped = np.reshape(X_val, (X_val.shape[0], 1, side_length, side_length))
    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, side_length, side_length))

    model = ConvAutoencoderModel(0.001, 50, 100, [144, 512, 784], [(1, 8, 3, 26), (8, 4, 2, 12)])
    # model = AutoencoderModel(0.1, 20, 100, [784,108, 256, 784])
    # plot_weights("before_learning_hidden_layer.png", model._layers[0]._W)

    model.train(X_train_reshaped, X_train, X_val_reshaped, X_val)

    plot([model.loss_data_points], [""], "Validation loss", "main_model_val_loss")
    plot([model.accuracy_data_points], [""], "Accuracy", "main_model_accuracy")
    plot([model.train_loss_data_point], [""], "Train loss", "main_model_train_loss")

    save_unseen_examples("main_model_reconstruction", model, X_test_reshaped, 3)
    # plot_weights("after_learning_hidden_layer.png", model._layers[0]._W)

    pred = model.encode(X_test_reshaped)
    plot_umap("main_model_umap.png", pred.T, y_test)

def main():
    loader = DataLoader()
    X_train, y_train, X_val, y_val, X_test, y_test = loader.preprocess()
    # experiment(X_train, X_val, X_test, y_test)
    # standard_execution(X_train, X_val, X_test, y_test)

    no_autoencoder_loss, no_autoencoder_acc, no_autoencoder_train_loss = no_autoencoder_execution(X_train, y_train, X_val, y_val, X_test, y_test)
    autoencoder_loss, autoencoder_acc, autoencoder_train_loss = autoencoder_execution(X_train, y_train, X_val, y_val, X_test, y_test)
    
    loss_arr = [no_autoencoder_loss, autoencoder_loss]
    acc_arr = [no_autoencoder_acc, autoencoder_acc]
    train_loss_arr = [no_autoencoder_train_loss, autoencoder_train_loss]

    plot(loss_arr, ["Dims 1", "Dims 2", "Dims 3"], "Validation loss", f"experiment_dims_val_loss")
    plot(acc_arr, ["Dims 1", "Dims 2", "Dims 3"], "Negative mean absolute error", f"experiment_dims_accuracy")
    plot(train_loss_arr, ["Dims 1", "Dims 2", "Dims 3"], "Train loss", f"experiment_dims_train_loss")


if __name__ == "__main__":
    main()