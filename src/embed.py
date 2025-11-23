import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

def generate_embeddings(
    output_npy="results/embeddings.npy",
    output_labels="results/labels.npy"
):
    # Resolve project root (one level above /src)
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv = os.path.join(ROOT, "data", "sample_reports.csv")

    print(f"Loading CSV from: {input_csv}")

    df = pd.read_csv(input_csv)

    print("Loading SentenceTransformer model on GPU...")
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Encoding reports (GPU)...")
    embeddings = model.encode(df["report"].tolist(), convert_to_numpy=True)
    labels = df["label"].tolist()

    # SAVE RESULTS (always inside project-root/results)
    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    np.save(os.path.join(results_dir, "embeddings.npy"), embeddings)
    np.save(os.path.join(results_dir, "labels.npy"), labels)

    print("\n Embeddings successfully generated and saved.")
    print(f"Saved embeddings: {os.path.join(results_dir, 'embeddings.npy')}")
    print(f"Saved labels:     {os.path.join(results_dir, 'labels.npy')}")

if __name__ == "__main__":
    import torch
    generate_embeddings()
