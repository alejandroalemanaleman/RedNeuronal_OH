import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def graficar_TSNE(X, y, iris):


    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    X_reduced = tsne.fit_transform(X)

    df = pd.DataFrame(X_reduced, columns=['Dim1', 'Dim2', 'Dim3'])
    df['target'] = y

    colors = ['red', 'blue', 'green']
    target_names = iris.target_names

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, target_name in enumerate(target_names):
        ax.scatter(df.loc[df['target'] == i, 'Dim1'],
                   df.loc[df['target'] == i, 'Dim2'],
                   df.loc[df['target'] == i, 'Dim3'],
                   label=target_name, color=colors[i])

    ax.set_xlabel('Dim1')
    ax.set_ylabel('Dim2')
    ax.set_zlabel('Dim3')
    ax.set_title('Iris Dataset en 3D usando t-SNE')
    ax.legend()
    plt.show()