import numpy as np
import umap
import matplotlib.pyplot as plt

def plot_umap(
    embeddings_path="results/embeddings.npy",
    labels_path="results/labels.npy"
):
    emb = np.load(embeddings_path)
    labels = np.load(labels_path)

    reducer = umap.UMAP(random_state=42)
    X = reducer.fit_transform(emb)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X[:,0], X[:,1], c=[0 if l=="normal" else 1 for l in labels], cmap="coolwarm")

    plt.legend(
        handles=scatter.legend_elements()[0],
        labels=["Normal", "Abnormal"]
    )

    plt.title("UMAP Projection of Report Embeddings")
    plt.savefig("results/umap_projection.png")
    plt.close()

    print(" UMAP visualization saved in results/umap_projection.png")

if __name__ == "__main__":
    plot_umap()
