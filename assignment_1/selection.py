from utils import he_initialization

class Selector:

    def __init__(self, learning_rates, epochs, mini_batch_sizes):
        self.learning_rates = learning_rates
        self.epochs = epochs
        self.mini_batch_sizes = mini_batch_sizes

    def select_best(self, X_train, y_train, X_val, y_val, model_class, layer_sizes):
        best_accuracy = float("-inf")
        best_params = ()
        for lr in self.learning_rates:
            for epoch_no in self.epochs:
                for mini_batch_size in self.mini_batch_sizes:
                    model = model_class(lr, epoch_no, mini_batch_size, he_initialization, layer_sizes)
                    y_pred = model.predict(X_val)
                    accuracy = model.evaluation(y_val.T, y_pred)
                    print("Learning rate: " + str(lr))
                    print("Epochs number: " + str(epoch_no))
                    print("Mini batch size: " + str(mini_batch_size))
                    print("Before training accuracy on validation data: " + str(accuracy))

                    model.train(X_train, y_train, X_val, y_val)
                    y_pred = model.predict(X_val)

                    accuracy = model.evaluation(y_val.T, y_pred)
                    print("After training accuracy on validation data: " + str(accuracy))
                    print("\n==========\n")
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = (lr, epoch_no, mini_batch_size)

        return best_params