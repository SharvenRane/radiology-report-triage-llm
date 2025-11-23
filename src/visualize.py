import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import umap

def visualize_embeddings(
    embeddings="results/embeddings.npy",
    labels="results/labels.npy"
):
    X = np.load(embeddings)
    y = np.load(labels, allow_pickle=True)

    reducer = umap.UMAP()
    X_2d = reducer.fit_transform(X)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=y, palette="deep")
    plt.title("UMAP Projection of Radiology Report Embeddings")
    plt.savefig("results/umap_plot.png")
    plt.show()

if __name__ == "__main__":
    visualize_embeddings()
