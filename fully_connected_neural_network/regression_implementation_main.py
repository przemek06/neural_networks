from loader.age_prediction_data_loader import DataLoader
from selection import Selector
from utils import he_initialization, plot, mean_absolute_error
from model.regression_model import RegressionModel

learning_rates = [0.01, 0.002, 0.001]
epochs = [50, 100]
mini_batch_sizes = [100, 200]

layer_sizes = [3072, 512, 64]


def main():
    loader = DataLoader()
    X_train, y_train, X_val, y_val, X_test, y_test = loader.preprocess()
    selector = Selector(learning_rates, epochs, mini_batch_sizes)
    lr, epoch_no, mini_batch_size = selector.select_best(X_train, y_train, X_val, y_val, RegressionModel, layer_sizes)
    print("Best parameters: ", lr, epoch_no, mini_batch_size)
    model = RegressionModel(lr, epoch_no, mini_batch_size, he_initialization, layer_sizes)
    model.train(X_train, y_train, X_val, y_val)
    y_pred = model.predict(X_test)
    accuracy = mean_absolute_error(y_test.T, y_pred)
    print("Accuracy on test data: " + str(accuracy))
    plot(model.accuracy_data_points, "accuracy", "Accuracy")
    plot(model.loss_data_points, "loss", "Loss")


if __name__ == "__main__":
    main()