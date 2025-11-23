import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os

def generate_embeddings(input_csv="data/sample_reports.csv",
                        output_npy="results/embeddings.npy",
                        output_labels="results/labels.npy"):
    df = pd.read_csv(input_csv)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    embeddings = model.encode(df["report"].tolist(), convert_to_numpy=True)
    labels = df["label"].tolist()

    os.makedirs("results", exist_ok=True)
    np.save(output_npy, embeddings)
    np.save(output_labels, labels)

    print("🔥 Embeddings saved to results folder.")

if __name__ == "__main__":
    generate_embeddings()
