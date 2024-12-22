import numpy as np


def cross_entropy(a3, y_true):

    m = y_true.shape[0]

    cost = -np.sum(y_true * np.log(a3 + 1e-8)) / m

    return cost