from itertools import product
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from RedNeuronal.data_representation.TSNE import graficar_TSNE
from model.model import model
from neural_network_propagation.predict import predict
import numpy as np


# Función para inicializar datos
def initialize_data():
    iris = load_iris()
    X = iris.data
    y = iris.target.reshape(-1, 1)

    # Visualización con t-SNE
    graficar_TSNE(X, y, iris)

    # One-hot encoding
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y)

    # Dividir en conjuntos de entrenamiento, validación y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.3, random_state=156477)
    X_train_2, X_val, y_train_2, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=15477)

    return X_train, X_test, y_train, y_test, X_train_2, X_val, y_train_2, y_val

# Función principal
def model_train_and_validation(layers_dims, learning_rates, batches, X_train_2, X_val, y_train_2, y_val):

    optimizers = ['Adam', 'Estocastic']
    resultados = {}

    combinaciones = product(layers_dims, optimizers, learning_rates, batches)

    for layer_dim, optimizer, learning_rate, batch in combinaciones:
        print(f'------------------------\n')
        print(
            f'OPTIMIZADOR: {optimizer}    LEARNING_RATE: {learning_rate}    Nº BATCHES: {batch}    ESTRUCTURA CAPAS: {layer_dim}')
        parameters = model(
            X_train_2, y_train_2, layer_dim,
            optimizer_use=optimizer,
            learning_rate=learning_rate,
            num_epochs=10000,
            batch_size=batch,
            print_cost=True
        )
        predictions, accuracy = predict(parameters, X_val, y_val)
        resultados[(optimizer, learning_rate, batch, layer_dim)] = accuracy
        print(
            f'La exactitud para optimizador {optimizer} con learning rate {learning_rate}, nº batches {batch} y estructura de capas {layer_dim}\nes de: {accuracy * 100:.6f} %')
        print(f'------------------------\n')

    claves_ordenadas = sorted(resultados.keys(), key=lambda k: resultados[k], reverse=True)
    print("Claves ordenadas según el tamaño del valor:")
    for clave in claves_ordenadas:
        print(f"{clave}: {resultados[clave]:.4f}")

    return claves_ordenadas[:11]

def model_test(combination, X_train, X_test, y_train, y_test):
    parameters = model(
        X_train, y_train, combination[-1],
        optimizer_use=combination[0],
        learning_rate=combination[1],
        num_epochs=10000,
        batch_size=combination[2]
        )
    predictions, accuracy = predict(parameters, X_test, y_test)
    print(f'------------------------\n')
    print(f'\nLas predicciones son: {predictions}')
    print(f'La exactitud es de: {accuracy * 100:.2f} %')
    y_test_labels = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_test_labels, predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=load_iris().target_names)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.show()


# Punto de entrada
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, X_train_2, X_val, y_train_2, y_val = initialize_data()
    layers_dims = [(X_train_2.shape[1], 16, 12, 8, 3), (X_train_2.shape[1], 16, 6, 3), (X_train_2.shape[1], 16, 18, 16, 8, 3)]
    learning_rates = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    batches = [8, 16, 32]


    combinations = model_train_and_validation(layers_dims, learning_rates, batches, X_train_2, X_val, y_train_2, y_val)
    for combination in combinations:
        model_test(combination, X_train, X_test, y_train, y_test)