import numpy as np
from matplotlib import pyplot as plt

from RedNeuronal.neural_network_propagation import *
from RedNeuronal.optimizers import *
from RedNeuronal.batches import *
from RedNeuronal.loss_functions import *

def model(X, Y, layers_dims, learning_rate=0.01, num_epochs=500, batch_size=32, optimizer_use="Adam", print_cost=False):
    """
    Entrena el modelo utilizando mini-batches.
    """
    L = len(layers_dims)
    costs = []
    t = 0

    parameters = random_initialization(layers_dims)
    if optimizer_use == "Adam": optimizer = AdamOptimizer(parameters, learning_rate=learning_rate)
    if optimizer_use == "Estocastic": optimizer = EstocasticOptimizer(learning_rate=learning_rate)

    for i in range(num_epochs):
        batches = create_batches(X, Y, batch_size)  # Crear lotes
        for batch in batches:
            X_batch, Y_batch = batch

            # Propagación hacia adelante
            aL, cache = forward_propagation(X_batch, parameters)

            # Costo
            cost = cross_entropy(aL, Y_batch)

            # Propagación hacia atrás
            grads = backward_propagation(X_batch, Y_batch, cache, parameters)

            # Actualización de parámetros
            t += 1
            if optimizer_use == "Adam":
                parameters = optimizer.update(parameters, grads, t)
            elif optimizer_use == "Estocastic":
                parameters = optimizer.update(parameters, grads)

        # Imprimir y registrar costos periódicamente
        if print_cost and i % 1000 == 0:
            print(f"Costo tras epoch {i}: {cost:.6f}")
        if i % 100 == 0:
            costs.append(cost)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title(f"Optimizador: {optimizer_use}; Learning rate = {learning_rate}; Batch size = {batch_size}")
    plt.show()

    return parameters
