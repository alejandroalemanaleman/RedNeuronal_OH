import numpy as np
from RedNeuronal.neural_network_propagation import forward_propagation
from sklearn.metrics import accuracy_score

def predict(parameters, X_test, y_test):

    a3, _ = forward_propagation(X_test, parameters)
    predictions = np.argmax(a3, axis=1)

    y_test_labels = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_test_labels, predictions)

    return predictions, accuracy