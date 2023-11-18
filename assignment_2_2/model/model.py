import math
from utils import generate_layers
from layer.relu_layer import ReluLayer

class Model:
    def __init__(self, learning_rate, epochs, batch_size, layer_sizes, weight_decay) -> None:
        self._layers = generate_layers(layer_sizes, ReluLayer, self.last_layer_class())
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        self._weight_decay = weight_decay
        self.loss_data_points = []
        self.accuracy_data_points = []
        self.train_loss_data_point = []

    def forward_propagation(self, X, eval):
        A = X
        for layer in self._layers:
            A = layer.forward_propagation(A, eval)
        return A
    
    def backward_propagation(self, dA):
        dA_curr = dA
        for layer in reversed(self._layers):
            dA_curr = layer.backward_propagation(dA_curr)

    def update(self, lr):
        for layer in self._layers:
            layer.update(lr, self._weight_decay)

    def train(self, X, y, X_valid, y_valid):

        for epoch in range(self._epochs):
            i=0
            num_of_batches = int(len(X)/self._batch_size)

            for m in range(num_of_batches):
                x = (X[i:i+self._batch_size]).T
                y_true = (y[i:i+self._batch_size]).T
                y_pred = self.forward_propagation(x, False)  
                dA = self.calculate_dA(y_pred, y_true)
                self.backward_propagation(dA)
                self.update(self._learning_rate/(1 + math.log(epoch + 1, 2)))
                i=i+self._batch_size

            y_probabilities = self.forward_propagation(X_valid.T, True)
            y_pred = self.predict(X_valid)
            loss = self.calculate_loss(y_probabilities, y_valid.T)
            accuracy = self.evaluation(y_valid.T, y_pred)
            self.loss_data_points.append(loss)
            self.accuracy_data_points.append(accuracy)
            y_train_probabilities = self.forward_propagation(X.T, True)
            train_loss = self.calculate_loss(y_train_probabilities, y.T)
            self.train_loss_data_point.append(train_loss)


    def calculate_loss(self, y_pred, y):
        pass

    def calculate_dA(self, y_pred, y):
        pass

    def evaluation(self, y_true, y_pred):
        pass

    def predict(self, X):
        pass

    def last_layer_class(self):
        pass
    