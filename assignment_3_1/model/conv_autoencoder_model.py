from model.model import Model
from utils import mean_absolute_error, generate_layers, generate_conv_layers
from layer.sigmoid_layer import SigmoidLayer
from layer.regression_layer import RegressionLayer
from layer.relu_layer import ReluLayer
from layer.relu_conv_layer import ReluConvLayer
from layer.flatten_layer import FlattenLayer
from layer.max_pooling_layer import MaxPoolingLayer
from layer.conv_layer import ConvLayer
import math
import sys

class ConvAutoencoderModel(Model):

    def __init__(self, learning_rate, epochs, batch_size, dense_layer_sizes, conv_layers_sizes) -> None:
        encoder_layers = generate_conv_layers(conv_layers_sizes, ReluConvLayer, ConvLayer, MaxPoolingLayer, FlattenLayer)
        decoder_layers = generate_layers(dense_layer_sizes, ReluLayer, self.last_layer_class())
        self._layers = encoder_layers + decoder_layers
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        self.loss_data_points = []
        self.accuracy_data_points = []
        self.train_loss_data_point = []  


    def train(self, X, y, X_valid, y_valid):
        y_probabilities = self.forward_propagation(X_valid)
        y_pred = self.predict(X_valid)
        loss = self.calculate_loss(y_probabilities, y_valid.T)
        accuracy = self.evaluation(y_valid.T, y_pred)
        self.loss_data_points.append(loss)
        self.accuracy_data_points.append(accuracy)
        y_train_probabilities = self.forward_propagation(X)
        loss = self.calculate_loss(y_train_probabilities, y.T)
        self.train_loss_data_point.append(loss)
        print(-1)
        print(loss)
        print(accuracy)

        for epoch in range(self._epochs):
            i=0
            num_of_batches = int(X.shape[0]/self._batch_size)

            for m in range(num_of_batches):
                x = (X[i:i+self._batch_size])
                y_true = (y[i:i+self._batch_size]).T
                y_pred = self.forward_propagation(x)  
                dA = self.calculate_dA(y_pred, y_true)
                self.backward_propagation(dA)
                self.update(self._learning_rate)
                i=i+self._batch_size
            
            y_probabilities = self.forward_propagation(X_valid)
            y_pred = self.predict(X_valid)
            loss = self.calculate_loss(y_probabilities, y_valid.T)
            accuracy = self.evaluation(y_valid.T, y_pred)
            self.loss_data_points.append(loss)
            self.accuracy_data_points.append(accuracy)
            y_train_probabilities = self.forward_propagation(X)
            loss = self.calculate_loss(y_train_probabilities, y.T)
            self.train_loss_data_point.append(loss)
            print(epoch)
            print(loss)
            print(accuracy)
  

    def predict(self, X):
        return self.forward_propagation(X)

    def calculate_loss(self, y_pred, y):
        n = 1/y_pred.shape[1]
        return n/2 * ((y - y_pred) ** 2).sum()
    
    def calculate_dA(self, y_pred, y):
        return y_pred - y

    def last_layer_class(self):
        return RegressionLayer
    
    def evaluation(self, y_true, y_pred):
        return -mean_absolute_error(y_true, y_pred)
    
    def encode(self, X):
        A = X
        for i in range(len(self._layers)):
            A = self._layers[i].forward_propagation(A)
            if type(self._layers[i]) == FlattenLayer:
                return A
        return A