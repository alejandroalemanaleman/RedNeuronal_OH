import numpy as np


def create_batches(X, Y, batch_size):
    """
    Divide los datos en lotes (batches) de tamaño batch_size.
    """
    m = X.shape[0]
    indices = np.random.permutation(m)
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]

    batches = []
    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        Y_batch = Y_shuffled[i:i+batch_size]
        batches.append((X_batch, Y_batch))

    return batches