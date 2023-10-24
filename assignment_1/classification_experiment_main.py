from model.binary_classification_model import BinaryClassificationModel
from utils import he_initialization, uniform_initialization, random_initializaton, plot
from loader.heart_disease_data_loader import DataLoader

def model_dimensionality_experiment(X_train, y_train, X_val, y_val, X_test, y_test):

    print("DIMENSIONALITY EXPERIMENT")
    print("\n=================\n")
    
    model1 = BinaryClassificationModel(0.002, 2000, 20, he_initialization,  [13, 64])
    model1.train(X_train, y_train, X_val, y_val)

    y_pred = model1.predict(X_test)

    evaluation = model1.evaluation(y_test.T, y_pred)
    plot(model1.accuracy_data_points, "accuracy", "One hidden layer accuracy")
    plot(model1.loss_data_points, "loss", "One hidden layer loss")
    print("Accuracy for one hidden layer in binary classification: " + str(evaluation))
    print("\n=================\n")

    model2 = BinaryClassificationModel(0.002, 2000, 20, he_initialization,  [13, 64, 32])
    model2.train(X_train, y_train, X_val, y_val)
    y_pred = model2.predict(X_test)

    evaluation = model2.evaluation(y_test.T, y_pred)
    plot(model2.accuracy_data_points, "accuracy", "Two hidden layers accuracy")
    plot(model2.loss_data_points, "loss", "Two hidden layers loss")
    print("Accuracy for two hidden layers in binary classification: " + str(evaluation))
    print("\n=================\n")

    model3 = BinaryClassificationModel(0.002, 2000, 20, he_initialization,  [13, 64, 32, 16, 8])
    model3.train(X_train, y_train, X_val, y_val)
    y_pred = model3.predict(X_test)

    evaluation = model3.evaluation(y_test.T, y_pred)
    plot(model3.accuracy_data_points, "accuracy", "Four hidden layers accuracy")
    plot(model3.loss_data_points, "loss", "Four hidden layers loss")
    print("Accuracy for four hidden layers in binary classification: " + str(evaluation))
    print("\n=================\n")

    model3 = BinaryClassificationModel(0.002, 2000, 20, he_initialization,  [13, 256, 128, 64, 32, 16, 8])
    model3.train(X_train, y_train, X_val, y_val)
    y_pred = model3.predict(X_test)

    evaluation = model3.evaluation(y_test.T, y_pred)
    plot(model3.accuracy_data_points, "accuracy", "Six hidden layers accuracy")
    plot(model3.loss_data_points, "loss", "Six hidden layers loss")
    print("Accuracy for six hidden layers in binary classification: " + str(evaluation))
    print("\n=================\n")

def model_learning_rate_experiment(X_train, y_train, X_val, y_val, X_test, y_test):
    print("LEARNING RATE EXPERIMENT")
    print("\n=================\n")
    
    model1 = BinaryClassificationModel(0.1, 2000, 20, he_initialization,  [13, 64, 32])
    model1.train(X_train, y_train, X_val, y_val)

    y_pred = model1.predict(X_test)

    evaluation = model1.evaluation(y_test.T, y_pred)
    plot(model1.accuracy_data_points, "accuracy", "0.1 learning rate accuracy")
    plot(model1.loss_data_points, "loss", "0.1 learning rate loss")
    print("Accuracy for 0.1 learning rate in binary classification: " + str(evaluation))
    print("\n=================\n")

    model2 = BinaryClassificationModel(0.01, 2000, 20, he_initialization,  [13, 64, 32])
    model2.train(X_train, y_train, X_val, y_val)

    y_pred = model2.predict(X_test)

    evaluation = model2.evaluation(y_test.T, y_pred)
    plot(model2.accuracy_data_points, "accuracy", "0.01 learning rate accuracy")
    plot(model2.loss_data_points, "loss", "0.01 learning rate loss")
    print("Accuracy for 0.01 learning rate in binary classification: " + str(evaluation))
    print("\n=================\n")

    model3 = BinaryClassificationModel(0.001, 2000, 20, he_initialization,  [13, 64, 32])
    model3.train(X_train, y_train, X_val, y_val)

    y_pred = model3.predict(X_test)

    evaluation = model3.evaluation(y_test.T, y_pred)
    plot(model3.accuracy_data_points, "accuracy", "0.001 learning rate accuracy")
    plot(model3.loss_data_points, "loss", "0.001 learning rate loss")
    print("Accuracy for 0.001 learning rate in binary classification: " + str(evaluation))
    print("\n=================\n")

    model4 = BinaryClassificationModel(0.0001, 2000, 20, he_initialization,  [13, 64, 32])
    model4.train(X_train, y_train, X_val, y_val)

    y_pred = model4.predict(X_test)

    evaluation = model4.evaluation(y_test.T, y_pred)
    plot(model4.accuracy_data_points, "accuracy", "0.0001 learning rate accuracy")
    plot(model4.loss_data_points, "loss", "0.0001 learning rate loss")
    print("Accuracy for 0.0001 learning rate in binary classification: " + str(evaluation))
    print("\n=================\n")

def model_batch_size_experiment(X_train, y_train, X_val, y_val, X_test, y_test):
    print("MINI BATCH SIZE EXPERIMENT")
    print("\n=================\n")
    
    model1 = BinaryClassificationModel(0.002, 2000, 1, he_initialization,  [13, 64, 32])
    model1.train(X_train, y_train, X_val, y_val)

    y_pred = model1.predict(X_test)

    evaluation = model1.evaluation(y_test.T, y_pred)
    plot(model1.accuracy_data_points, "accuracy", "batch size = 1 accuracy")
    plot(model1.loss_data_points, "loss", "batch size = 1 loss")
    print("Accuracy for mini batch size = 1 in binary classification: " + str(evaluation))
    print("\n=================\n")

    model2 = BinaryClassificationModel(0.002, 2000, 10, he_initialization,  [13, 64, 32])
    model2.train(X_train, y_train, X_val, y_val)

    y_pred = model2.predict(X_test)

    evaluation = model2.evaluation(y_test.T, y_pred)
    plot(model2.accuracy_data_points, "accuracy", "batch size = 10 accuracy")
    plot(model2.loss_data_points, "loss", "batch size = 10 loss")
    print("Accuracy for mini batch size = 10 in binary classification: " + str(evaluation))
    print("\n=================\n")

    model3 = BinaryClassificationModel(0.002, 2000, 50, he_initialization,  [13, 64, 32])
    model3.train(X_train, y_train, X_val, y_val)

    y_pred = model3.predict(X_test)

    evaluation = model3.evaluation(y_test.T, y_pred)
    plot(model3.accuracy_data_points, "accuracy", "batch size = 50 accuracy")
    plot(model3.loss_data_points, "loss", "batch size = 50 loss")
    print("Accuracy for mini batch size = 50 in binary classification: " + str(evaluation))
    print("\n=================\n")

    model4 = BinaryClassificationModel(0.002, 2000, 100, he_initialization,  [13, 64, 32])
    model4.train(X_train, y_train, X_val, y_val)

    y_pred = model4.predict(X_test)

    evaluation = model4.evaluation(y_test.T, y_pred)
    plot(model4.accuracy_data_points, "accuracy", "batch size = 100 accuracy")
    plot(model4.loss_data_points, "loss", "batch size = 100 loss")
    print("Accuracy for mini batch size = 100 in binary classification: " + str(evaluation))
    print("\n=================\n")

def model_epochs_number_experiment(X_train, y_train, X_val, y_val, X_test, y_test):
    print("EPOCH NUMBER EXPERIMENT")
    print("\n=================\n")
    
    model1 = BinaryClassificationModel(0.002, 10, 20, he_initialization,  [13, 64, 32])
    model1.train(X_train, y_train, X_val, y_val)

    y_pred = model1.predict(X_test)

    evaluation = model1.evaluation(y_test.T, y_pred)
    plot(model1.accuracy_data_points, "accuracy", "epochs number = 10 accuracy")
    plot(model1.loss_data_points, "loss", "epochs number = 10 loss")
    print("Accuracy for epochs number = 10 in binary classification: " + str(evaluation))
    print("\n=================\n")

    model2 = BinaryClassificationModel(0.002, 100, 20, he_initialization,  [13, 64, 32])
    model2.train(X_train, y_train, X_val, y_val)

    y_pred = model2.predict(X_test)

    evaluation = model2.evaluation(y_test.T, y_pred)
    plot(model2.accuracy_data_points, "accuracy", "epochs number = 100 accuracy")
    plot(model2.loss_data_points, "loss", "epochs number = 100 loss")
    print("Accuracy for epochs number = 100 in binary classification: " + str(evaluation))
    print("\n=================\n")

    model3 = BinaryClassificationModel(0.002, 1000, 20, he_initialization,  [13, 64, 32])
    model3.train(X_train, y_train, X_val, y_val)

    y_pred = model3.predict(X_test)

    evaluation = model3.evaluation(y_test.T, y_pred)
    plot(model3.accuracy_data_points, "accuracy", "epochs number = 1000 accuracy")
    plot(model3.loss_data_points, "loss", "epochs number = 1000 loss")
    print("Accuracy for epochs number = 1000 in binary classification: " + str(evaluation))
    print("\n=================\n")

    model4 = BinaryClassificationModel(0.002, 10000, 20, he_initialization,  [13, 64, 32])
    model4.train(X_train, y_train, X_val, y_val)

    y_pred = model4.predict(X_test)

    evaluation = model4.evaluation(y_test.T, y_pred)
    plot(model4.accuracy_data_points, "accuracy", "epochs number = 10000 accuracy")
    plot(model4.loss_data_points, "loss", "epochs number = 10000 loss")
    print("Accuracy for epochs number = 10000 in binary classification: " + str(evaluation))
    print("\n=================\n")

def model_initialization_experiment(X_train, y_train, X_val, y_val, X_test, y_test):

    print("WEIGHT INITIALIZATION EXPERIMENT")
    print("\n=================\n")
    
    model1 = BinaryClassificationModel(0.002, 2000, 20, he_initialization,  [13, 64, 32])
    model1.train(X_train, y_train, X_val, y_val)

    y_pred = model1.predict(X_test)

    evaluation = model1.evaluation(y_test.T, y_pred)
    plot(model1.accuracy_data_points, "accuracy", "he initialization accuracy")
    plot(model1.loss_data_points, "loss", "he initialization layer loss")
    print("Accuracy for he initialization in binary classification: " + str(evaluation))
    print("\n=================\n")

    model2 = BinaryClassificationModel(0.002, 2000, 20, uniform_initialization,  [13, 64, 32])
    model2.train(X_train, y_train, X_val, y_val)
    y_pred = model2.predict(X_test)

    evaluation = model2.evaluation(y_test.T, y_pred)
    plot(model2.accuracy_data_points, "accuracy", "uniform initialization accuracy")
    plot(model2.loss_data_points, "loss", "uniform initialization loss")
    print("Accuracy for uniform initialization in binary classification: " + str(evaluation))
    print("\n=================\n")

    model3 = BinaryClassificationModel(0.002, 2000, 20, random_initializaton,  [13, 64, 32])
    model3.train(X_train, y_train, X_val, y_val)
    y_pred = model3.predict(X_test)

    evaluation = model3.evaluation(y_test.T, y_pred)
    plot(model3.accuracy_data_points, "accuracy", "radnom initialization accuracy")
    plot(model3.loss_data_points, "loss", "radnom initialization loss")
    print("Accuracy for radnom initialization in binary classification: " + str(evaluation))
    print("\n=================\n")

def main():
    loader = DataLoader()
    X_train, y_train, X_val, y_val, X_test, y_test = loader.preprocess()
    #model_dimensionality_experiment(X_train, y_train, X_val, y_val, X_test, y_test)
    # model_learning_rate_experiment(X_train, y_train, X_val, y_val, X_test, y_test)
    # model_batch_size_experiment(X_train, y_train, X_val, y_val, X_test, y_test)
    # model_epochs_number_experiment(X_train, y_train, X_val, y_val, X_test, y_test)
    model_initialization_experiment(X_train, y_train, X_val, y_val, X_test, y_test)

if __name__ == "__main__":
    main()