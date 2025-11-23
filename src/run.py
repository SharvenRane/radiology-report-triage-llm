import embed
import classify
import visualize

def run():
    print("\nSTEP 1 — Embedding reports...")
    embed.generate_embeddings()

    print("\nSTEP 2 — Training classifier...")
    classify.train_classifier()

    print("\nSTEP 3 — Visualizing embeddings...")
    visualize.visualize_embeddings()

    print("\n Pipeline complete!")

if __name__ == "__main__":
    run()
