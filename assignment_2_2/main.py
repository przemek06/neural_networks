from loader.fashion_mnist_data_loader import DataLoader
from model.autoencoder_model import AutoencoderModel
from selector.selector import Selector
from utils import vector_descaling, plot, save_image, plot_weights, plot_umap, vector_descaling, save_unseen_examples
from experiments import batch_size_experiment

weight_decay = 0.1

def standard_execution(X_train, X_val, X_test, y_test):
    selector = Selector()
    lr, batch_size, dims = selector.select_best_params(AutoencoderModel, X_train, X_val)

    model = AutoencoderModel(lr, 200, batch_size, dims, weight_decay)
    plot_weights("before_learning_hidden_layer.png", model._layers[0]._W)

    model.train(X_train, X_train, X_val, X_val)
    print(model.loss_data_points == model.train_loss_data_point)

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
    # batch_size_experiment(X_train, X_val, X_test, y_test)
    standard_execution(X_train, X_val, X_test, y_test)

if __name__ == "__main__":
    main()