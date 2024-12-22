import numpy as np
from RedNeuronal.activation_functions.relu import relu
from RedNeuronal.activation_functions.softmax import softmax

def forward_propagation(X, parameters):

    L = len(parameters) // 2
    cache = {'A0': X}

    A = X
    for l in range(1, L + 1):
        Z = np.dot(A, parameters[f'W{l}']) + parameters[f'b{l}']
        if l == L:
            A = softmax(Z)
        else:
            A = relu(Z)
        cache[f'Z{l}'] = Z
        cache[f'A{l}'] = A

    return A, cache