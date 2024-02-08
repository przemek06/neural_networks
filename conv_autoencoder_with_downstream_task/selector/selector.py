
from utils import plot_umap

learning_rates = [0.02, 0.01]
batch_sizes = [50, 100, 200]
dimensions_set = [[784, 256, 64, 256, 784], [784, 256, 128, 64, 128, 256, 784]]

# learning_rates = [0.01]
# batch_sizes = [100]
# dimensions_set = [[784, 256, 64, 256, 784]]

class Selector():

    def select_best_params(self, model_class, X_train, X_val):
        best_accuracy = float("-inf")
        best_params = ()
        for lr in learning_rates:
            for batch_size in batch_sizes:
                for dims in dimensions_set:
                    model = model_class(lr, 20, batch_size, dims)
                    model.train(X_train, X_train, X_val, X_val)

                    X_pred = model.predict(X_val)
                    accuracy = model.evaluation(X_val, X_pred.T)
                    
                    result = "Model parameters: \n Learning rate = {} \n Batch size = {} \n Dimensions = {} \n Accuracy = {} \n\n".format(lr, batch_size, len(dims) - 1, accuracy)
                    print(result)

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = (lr, batch_size, dims)

        return best_params