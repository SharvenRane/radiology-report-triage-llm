import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def train_classifier(
    embeddings="results/embeddings.npy",
    labels="results/labels.npy",
    model_path="results/classifier.pkl"
):
    X = np.load(embeddings)
    y = np.load(labels, allow_pickle=True)

    clf = LogisticRegression(max_iter=500)
    clf.fit(X, y)

    preds = clf.predict(X)
    print("=== Classification Report ===")
    print(classification_report(y, preds))

    joblib.dump(clf, model_path)
    print(f"✔ Classifier saved to {model_path}")

if __name__ == "__main__":
    train_classifier()
