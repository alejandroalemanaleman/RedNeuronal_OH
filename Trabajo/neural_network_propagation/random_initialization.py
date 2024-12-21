import numpy as np


def random_initialization(layers_dims):
    """
    Inicializa los parámetros para cualquier número de capas.
    """
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(layers_dims[l - 1], layers_dims[l]) * 0.01
        parameters[f'b{l}'] = np.zeros((1, layers_dims[l]))

    return parameters
