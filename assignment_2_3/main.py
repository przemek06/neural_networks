from loader.fashion_mnist_data_loader import DataLoader
from model.autoencoder_model import AutoencoderModel
from model.softmax_classification_model import SoftmaxClassificationModel
from selector.selector import Selector
from utils import vector_descaling, plot, save_image, plot_weights, plot_umap, vector_descaling, save_unseen_examples
from experiments import batch_size_experiment

def no_autoencoder_execution(X_train, y_train, X_val, y_val, X_test, y_test):
    model = SoftmaxClassificationModel(0.001, 100, 200, [784, 512, 256, 10])
    model.train(X_train, y_train, X_val, y_val)

    plot([model.loss_data_points], [""], "Validation loss", "no_autoencoder_val_loss")
    plot([model.accuracy_data_points], [""], "Accuracy", "no_autoencoder_accuracy")
    plot([model.train_loss_data_point], [""], "Train loss", "no_autoencoder_train_loss")

    test_pred = model.predict(X_test)
    test_accuracy = model.evaluation(y_test, test_pred)
    print(f"Test accuracy with no autoencoder = {test_accuracy}")

def autoencoder_execution(X_train, y_train, X_val, y_val, X_test, y_test):
    autoencoder = AutoencoderModel(0.001, 50, 200, [784, 256, 64, 256, 784])
    autoencoder.train(X_train, X_train, X_val, X_val)

    X_train_encoded = autoencoder.encode(X_train.T).T
    X_val_encoded = autoencoder.encode(X_val.T).T
    X_test_encoded = autoencoder.encode(X_test.T).T

    model = SoftmaxClassificationModel(0.01, 100, 50, [64, 512, 256, 10])
    model.train(X_train_encoded, y_train, X_val_encoded, y_val)

    plot([model.loss_data_points], [""], "Validation loss", "autoencoder_val_loss")
    plot([model.accuracy_data_points], [""], "Accuracy", "autoencoder_accuracy")
    plot([model.train_loss_data_point], [""], "Train loss", "autoencoder_train_loss")

    test_pred = model.predict(X_test_encoded)
    test_accuracy = model.evaluation(y_test, test_pred)
    print(f"Test accuracy with autoencoder = {test_accuracy}")

def standard_execution(X_train, X_val, X_test, y_test):
    selector = Selector()
    lr, batch_size, dims = selector.select_best_params(AutoencoderModel, X_train, X_val)

    model = AutoencoderModel(lr, 200, batch_size, dims)
    plot_weights("before_learning_hidden_layer.png", model._layers[0]._W)

    model.train(X_train, X_train, X_val, X_val)

    plot([model.loss_data_points], [""], "Validation loss", "main_model_val_loss")
    plot([model.accuracy_data_points], [""], "Accuracy", "main_model_accuracy")
    plot([model.train_loss_data_point], [""], "Train loss", "main_model_train_loss")

    save_unseen_examples("main_model_reconstruction", model, X_test, 3)
    plot_weights("after_learning_hidden_layer.png", model._layers[0]._W)

    pred = model.encode(X_test.T)
    plot_umap("main_model_umap.png", pred.T, y_test)

def main():
    loader = DataLoader()
    X_train, y_train, X_val, y_val, X_test, y_test = loader.preprocess()
    no_autoencoder_execution(X_train, y_train, X_val, y_val, X_test, y_test)
    autoencoder_execution(X_train, y_train, X_val, y_val, X_test, y_test)

if __name__ == "__main__":
    main()