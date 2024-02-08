from model.binary_classification_model import BinaryClassificationModel

from loader.heart_disease_data_loader import DataLoader
from selection import Selector
from utils import binary_accuracy, he_initialization, plot

learning_rates = [0.002, 0.001, 0.0005]
epochs = [1000, 2000, 5000]
mini_batch_sizes = [10, 20, 50]

layer_sizes = [13, 64, 32]

def main():
    loader = DataLoader()
    X_train, y_train, X_val, y_val, X_test, y_test = loader.preprocess()
    selector = Selector(learning_rates, epochs, mini_batch_sizes)
    lr, epoch_no, mini_batch_size = selector.select_best(X_train, y_train, X_val, y_val, BinaryClassificationModel, layer_sizes)
    print("Best parameters: ", lr, epoch_no, mini_batch_size)
    model = BinaryClassificationModel(lr, epoch_no, mini_batch_size, he_initialization, layer_sizes)
    model.train(X_train, y_train, X_val, y_val)
    y_pred = model.predict(X_test)
    accuracy = binary_accuracy(y_test.T, y_pred)
    print("Accuracy on test data: " + str(accuracy))
    plot(model.accuracy_data_points, "accuracy", "Accuracy")
    plot(model.loss_data_points, "loss", "Loss")

if __name__ == "__main__":
    main()