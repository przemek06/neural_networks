from model.regression_model import RegressionModel
from utils import he_initialization, uniform_initialization, random_initializaton, plot
from loader.age_prediction_data_loader import DataLoader

def model_dimensionality_experiment(X_train, y_train, X_val, y_val, X_test, y_test):

    print("DIMENSIONALITY EXPERIMENT")
    print("\n=================\n")
    
    model1 = RegressionModel(0.002, 50, 200, he_initialization,  [3072, 512])
    model1.train(X_train, y_train, X_val, y_val)

    y_pred = model1.predict(X_test)

    evaluation = model1.evaluation(y_test.T, y_pred)
    plot(model1.accuracy_data_points, "accuracy", "One hidden layer accuracy")
    plot(model1.loss_data_points, "loss", "One hidden layer loss")
    print("Absolute error for one hidden layer in binary classification: " + str(evaluation))
    print("\n=================\n")

    model2 = RegressionModel(0.002, 50, 200, he_initialization,  [3072, 512, 256])
    model2.train(X_train, y_train, X_val, y_val)
    y_pred = model2.predict(X_test)

    evaluation = model2.evaluation(y_test.T, y_pred)
    plot(model2.accuracy_data_points, "absolute error", "Two hidden layers absolute error")
    plot(model2.loss_data_points, "loss", "Two hidden layers loss")
    print("Absolute error for two hidden layers in binary classification: " + str(evaluation))
    print("\n=================\n")

    model3 = RegressionModel(0.002, 50, 200, he_initialization,  [3072, 512, 256, 64, 32])
    model3.train(X_train, y_train, X_val, y_val)
    y_pred = model3.predict(X_test)

    evaluation = model3.evaluation(y_test.T, y_pred)
    plot(model3.accuracy_data_points, "absolute error", "Four hidden layers absolute error")
    plot(model3.loss_data_points, "loss", "Four hidden layers loss")
    print("Absolute error for four hidden layers in binary classification: " + str(evaluation))
    print("\n=================\n")

    model3 = RegressionModel(0.002, 50, 200, he_initialization,  [3072, 256, 128, 512, 256, 64, 32])
    model3.train(X_train, y_train, X_val, y_val)
    y_pred = model3.predict(X_test)

    evaluation = model3.evaluation(y_test.T, y_pred)
    plot(model3.accuracy_data_points, "absolute error", "Six hidden layers absolute error")
    plot(model3.loss_data_points, "loss", "Six hidden layers loss")
    print("Absolute error for six hidden layers in binary classification: " + str(evaluation))
    print("\n=================\n")

def model_learning_rate_experiment(X_train, y_train, X_val, y_val, X_test, y_test):
    print("LEARNING RATE EXPERIMENT")
    print("\n=================\n")
    
    model1 = RegressionModel(0.1, 50, 200, he_initialization,  [3072, 512, 256])
    model1.train(X_train, y_train, X_val, y_val)

    y_pred = model1.predict(X_test)

    evaluation = model1.evaluation(y_test.T, y_pred)
    plot(model1.accuracy_data_points, "absolute error", "0.1 learning rate absolute error")
    plot(model1.loss_data_points, "loss", "0.1 learning rate loss")
    print("Absolute error for 0.1 learning rate in binary classification: " + str(evaluation))
    print("\n=================\n")

    model2 = RegressionModel(0.01, 50, 200, he_initialization,  [3072, 512, 256])
    model2.train(X_train, y_train, X_val, y_val)

    y_pred = model2.predict(X_test)

    evaluation = model2.evaluation(y_test.T, y_pred)
    plot(model2.accuracy_data_points, "absolute error", "0.01 learning rate absolute error")
    plot(model2.loss_data_points, "loss", "0.01 learning rate loss")
    print("Absolute error for 0.01 learning rate in binary classification: " + str(evaluation))
    print("\n=================\n")

    model3 = RegressionModel(0.001, 50, 200, he_initialization,  [3072, 512, 256])
    model3.train(X_train, y_train, X_val, y_val)

    y_pred = model3.predict(X_test)

    evaluation = model3.evaluation(y_test.T, y_pred)
    plot(model3.accuracy_data_points, "absolute error", "0.001 learning rate absolute error")
    plot(model3.loss_data_points, "loss", "0.001 learning rate loss")
    print("Absolute error for 0.001 learning rate in binary classification: " + str(evaluation))
    print("\n=================\n")

    model4 = RegressionModel(0.0001, 50, 200, he_initialization,  [3072, 512, 256])
    model4.train(X_train, y_train, X_val, y_val)

    y_pred = model4.predict(X_test)

    evaluation = model4.evaluation(y_test.T, y_pred)
    plot(model4.accuracy_data_points, "absolute error", "0.0001 learning rate absolute error")
    plot(model4.loss_data_points, "loss", "0.0001 learning rate loss")
    print("Absolute error for 0.0001 learning rate in binary classification: " + str(evaluation))
    print("\n=================\n")

def model_batch_size_experiment(X_train, y_train, X_val, y_val, X_test, y_test):
    print("MINI BATCH SIZE EXPERIMENT")
    print("\n=================\n")
    
    model1 = RegressionModel(0.002, 50, 1, he_initialization,  [3072, 512, 256])
    model1.train(X_train, y_train, X_val, y_val)

    y_pred = model1.predict(X_test)

    evaluation = model1.evaluation(y_test.T, y_pred)
    plot(model1.accuracy_data_points, "absolute error", "batch size = 1 absolute error")
    plot(model1.loss_data_points, "loss", "batch size = 1 loss")
    print("Absolute error for mini batch size = 1 in binary classification: " + str(evaluation))
    print("\n=================\n")

    model2 = RegressionModel(0.002, 50, 50, he_initialization,  [3072, 512, 256])
    model2.train(X_train, y_train, X_val, y_val)

    y_pred = model2.predict(X_test)

    evaluation = model2.evaluation(y_test.T, y_pred)
    plot(model2.accuracy_data_points, "absolute error", "batch size = 10 absolute error")
    plot(model2.loss_data_points, "loss", "batch size = 10 loss")
    print("Absolute error for mini batch size = 10 in binary classification: " + str(evaluation))
    print("\n=================\n")

    model3 = RegressionModel(0.002, 50, 200, he_initialization,  [3072, 512, 256])
    model3.train(X_train, y_train, X_val, y_val)

    y_pred = model3.predict(X_test)

    evaluation = model3.evaluation(y_test.T, y_pred)
    plot(model3.accuracy_data_points, "absolute error", "batch size = 50 absolute error")
    plot(model3.loss_data_points, "loss", "batch size = 50 loss")
    print("Absolute error for mini batch size = 50 in binary classification: " + str(evaluation))
    print("\n=================\n")

    model4 = RegressionModel(0.002, 50, 500, he_initialization,  [3072, 512, 256])
    model4.train(X_train, y_train, X_val, y_val)

    y_pred = model4.predict(X_test)

    evaluation = model4.evaluation(y_test.T, y_pred)
    plot(model4.accuracy_data_points, "absolute error", "batch size = 100 absolute error")
    plot(model4.loss_data_points, "loss", "batch size = 100 loss")
    print("Absolute error for mini batch size = 100 in binary classification: " + str(evaluation))
    print("\n=================\n")

def model_epochs_number_experiment(X_train, y_train, X_val, y_val, X_test, y_test):
    print("EPOCH NUMBER EXPERIMENT")
    print("\n=================\n")
    
    model1 = RegressionModel(0.001, 10, 200, he_initialization,  [3072, 512, 256])
    model1.train(X_train, y_train, X_val, y_val)

    y_pred = model1.predict(X_test)

    evaluation = model1.evaluation(y_test.T, y_pred)
    plot(model1.accuracy_data_points, "absolute error", "epochs number = 10 absolute error")
    plot(model1.loss_data_points, "loss", "epochs number = 10 loss")
    print("Absolute error for epochs number = 10 in binary classification: " + str(evaluation))
    print("\n=================\n")

    model2 = RegressionModel(0.001, 100, 200, he_initialization,  [3072, 512, 256])
    model2.train(X_train, y_train, X_val, y_val)

    y_pred = model2.predict(X_test)

    evaluation = model2.evaluation(y_test.T, y_pred)
    plot(model2.accuracy_data_points, "absolute error", "epochs number = 100 absolute error")
    plot(model2.loss_data_points, "loss", "epochs number = 100 loss")
    print("Absolute error for epochs number = 100 in binary classification: " + str(evaluation))
    print("\n=================\n")

    model3 = RegressionModel(0.001, 1000, 200, he_initialization,  [3072, 512, 256])
    model3.train(X_train, y_train, X_val, y_val)

    y_pred = model3.predict(X_test)

    evaluation = model3.evaluation(y_test.T, y_pred)
    plot(model3.accuracy_data_points, "absolute error", "epochs number = 1000 absolute error")
    plot(model3.loss_data_points, "loss", "epochs number = 1000 loss")
    print("Absolute error for epochs number = 1000 in binary classification: " + str(evaluation))
    print("\n=================\n")

    model4 = RegressionModel(0.001, 10000, 200, he_initialization,  [3072, 512, 256])
    model4.train(X_train, y_train, X_val, y_val)

    y_pred = model4.predict(X_test)

    evaluation = model4.evaluation(y_test.T, y_pred)
    plot(model4.accuracy_data_points, "absolute error", "epochs number = 10000 absolute error")
    plot(model4.loss_data_points, "loss", "epochs number = 10000 loss")
    print("Absolute error for epochs number = 10000 in binary classification: " + str(evaluation))
    print("\n=================\n")

def model_initialization_experiment(X_train, y_train, X_val, y_val, X_test, y_test):

    print("WEIGHT INITIALIZATION EXPERIMENT")
    print("\n=================\n")
    
    model1 = RegressionModel(0.001, 50, 200, he_initialization,  [3072, 512, 256])
    model1.train(X_train, y_train, X_val, y_val)

    y_pred = model1.predict(X_test)

    evaluation = model1.evaluation(y_test.T, y_pred)
    plot(model1.accuracy_data_points, "absolute error", "he initialization absolute error")
    plot(model1.loss_data_points, "loss", "he initialization layer loss")
    print("Absolute error for he initialization in binary classification: " + str(evaluation))
    print("\n=================\n")

    model2 = RegressionModel(0.001, 50, 200, uniform_initialization,  [3072, 512, 256])
    model2.train(X_train, y_train, X_val, y_val)
    y_pred = model2.predict(X_test)

    evaluation = model2.evaluation(y_test.T, y_pred)
    plot(model2.accuracy_data_points, "absolute error", "uniform initialization absolute error")
    plot(model2.loss_data_points, "loss", "uniform initialization loss")
    print("Absolute error for uniform initialization in binary classification: " + str(evaluation))
    print("\n=================\n")

    model3 = RegressionModel(0.001, 50, 200, random_initializaton,  [3072, 512, 256])
    model3.train(X_train, y_train, X_val, y_val)
    y_pred = model3.predict(X_test)

    evaluation = model3.evaluation(y_test.T, y_pred)
    plot(model3.accuracy_data_points, "absolute error", "radnom initialization absolute error")
    plot(model3.loss_data_points, "loss", "radnom initialization loss")
    print("Absolute error for radnom initialization in binary classification: " + str(evaluation))
    print("\n=================\n")

def main():
    loader = DataLoader()
    X_train, y_train, X_val, y_val, X_test, y_test = loader.preprocess()
    # model_dimensionality_experiment(X_train, y_train, X_val, y_val, X_test, y_test)
    # model_learning_rate_experiment(X_train, y_train, X_val, y_val, X_test, y_test)
    # model_batch_size_experiment(X_train, y_train, X_val, y_val, X_test, y_test)
    #model_epochs_number_experiment(X_train, y_train, X_val, y_val, X_test, y_test)
    model_initialization_experiment(X_train, y_train, X_val, y_val, X_test, y_test)

if __name__ == "__main__":
    main()