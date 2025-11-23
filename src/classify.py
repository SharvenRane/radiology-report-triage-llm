import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

def classify_embeddings(
    embeddings_path="results/embeddings.npy",
    labels_path="results/labels.npy"
):
    emb = np.load(embeddings_path)
    labels = np.load(labels_path)

    # Convert labels to numeric
    label_to_id = {"normal": 0, "abnormal": 1}
    true = np.array([label_to_id[l] for l in labels])

    km = KMeans(n_clusters=2, random_state=42)
    pred = km.fit_predict(emb)

    # Try to match cluster IDs to true labels
    if pred[0] != true[0]:
        pred = 1 - pred

    cm = confusion_matrix(true, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal","Abnormal"])
    disp.plot(cmap="Blues")

    plt.title("Confusion Matrix (Zero-Shot Triage)")
    plt.savefig("results/confusion_matrix.png")
    plt.close()

    print("🔥 Confusion matrix saved in results/confusion_matrix.png")

if __name__ == "__main__":
    classify_embeddings()
