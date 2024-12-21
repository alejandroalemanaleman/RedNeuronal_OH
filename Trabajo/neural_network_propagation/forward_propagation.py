import numpy as np
from Trabajo.activation_functions.relu import relu
from Trabajo.activation_functions.softmax import softmax

def forward_propagation(X, parameters):
    """
    Realiza la propagación hacia adelante para cualquier número de capas.
    """
    L = len(parameters) // 2  # Número de capas (W, b cuentan como dos claves por capa)
    cache = {'A0': X}

    A = X
    for l in range(1, L + 1):
        Z = np.dot(A, parameters[f'W{l}']) + parameters[f'b{l}']
        if l == L:  # Última capa: Softmax
            A = softmax(Z)
        else:  # Capas ocultas: ReLU
            A = relu(Z)
        cache[f'Z{l}'] = Z
        cache[f'A{l}'] = A

    return A, cache