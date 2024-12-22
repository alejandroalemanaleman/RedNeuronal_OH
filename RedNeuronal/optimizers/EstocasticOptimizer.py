class EstocasticOptimizer():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, parameters, grads):
        L = len(parameters) // 2

        for l in range(L):
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - self.learning_rate * grads["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - self.learning_rate * grads["db" + str(l + 1)]
        return parameters