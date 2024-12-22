import numpy as np
from RedNeuronal.activation_functions.relu import relu_derivative

def backward_propagation(X, y, cache, parameters):
    """
    Realiza la propagación hacia atrás para cualquier número de capas.
    """
    L = len(parameters) // 2  # Número de capas
    m = X.shape[0]
    gradients = {}

    # Gradientes para la última capa
    dZ = cache[f'A{L}'] - y
    for l in reversed(range(1, L + 1)):
        dW = np.dot(cache[f'A{l-1}'].T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        gradients[f'dW{l}'] = dW
        gradients[f'db{l}'] = db

        if l > 1:  # No calcular da para la primera capa
            dA = np.dot(dZ, parameters[f'W{l}'].T)
            dZ = dA * relu_derivative(cache[f'Z{l-1}'])

    return gradients