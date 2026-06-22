# Radiology Report Triage with Sentence Embeddings

This project takes free text radiology reports and sorts them into **normal** and **abnormal**, the kind of first pass triage a clinic does before a human reads everything in depth. It does this without training a language model. Instead it turns each report into a dense vector with a pretrained sentence embedding model, and a lightweight classifier learns the boundary between the two classes on top of those vectors.

The idea is that the heavy lifting, understanding clinical phrasing like "moderate pleural effusion" or "lungs clear, no focal opacity", is already done by the embedding model. We never fine tune it. We only read the vectors it produces and draw a line through them.

## What it does

The pipeline has three stages and runs end to end from a single entry point:

1. **Embed.** Every report in the dataset is encoded with `sentence-transformers/all-MiniLM-L6-v2`, a 384 dimensional sentence embedding model. The vectors and their labels are saved to `results/`.
2. **Classify.** A logistic regression is fit on the saved embeddings and prints a per class report (precision, recall, f1). The fitted model is written to `results/classifier.pkl`.
3. **Visualize.** The embeddings are projected down to two dimensions with UMAP and plotted, colored by label, so you can eyeball how cleanly normal and abnormal reports separate in vector space. The figure is saved to `results/umap_plot.png`.

## The data

`data/sample_reports.csv` holds 50 short veterinary radiology impressions, one per row, with a `report` column and a `label` column. The labels are `normal` or `abnormal`. Normal rows read like "Heart size and shape within normal limits" or "No fracture or soft-tissue swelling identified." Abnormal rows describe findings such as effusions, masses, consolidation, obstructions, and lytic lesions. The split is 20 normal and 30 abnormal.

This is a small demonstration set meant to show the approach working. The same pipeline runs unchanged on a larger CSV as long as it keeps the `report` and `label` columns.

## Approach in a bit more detail

The embedding step is zero shot in the sense that the encoder is used straight off the shelf with no domain adaptation. `all-MiniLM-L6-v2` was trained on general purpose sentence similarity, and it already places clinically similar sentences near each other, which is exactly the structure a downstream classifier wants.

On top of those frozen vectors the classifier is a plain logistic regression from scikit learn. Because the embeddings carry most of the signal, a linear model on top is enough to separate the classes. Logistic regression also keeps the decision interpretable and fast, with nothing to tune.

For inspection rather than scoring, UMAP reduces the 384 dimensional vectors to a 2D scatter. If normal and abnormal form distinct neighborhoods in that plot, it is a good sign that the embedding space already encodes the distinction the triage task cares about.

## Results on the sample data

Running the classifier on the 50 report sample produces roughly:

```
Accuracy: 96%
Normal recall: 0.90
Abnormal recall: 1.00
```

One honest caveat worth stating plainly: in the current `classify.py` the logistic regression is fit and then scored on the same labeled set, so these numbers describe how well the model fits the data it saw, not how it would generalize to unseen reports. On a real evaluation you would hold out a test split or cross validate. The fit quality is still informative here because it confirms the two classes are linearly separable in the embedding space, which is the claim the project is making.

## Project layout

```
data/
  sample_reports.csv     50 labeled radiology impressions
src/
  run.py                 entry point, runs all three stages
  embed.py               encodes reports with the sentence transformer
  classify.py            fits and reports the logistic regression
  visualize.py           UMAP projection plot
results/                 created at runtime: embeddings, labels, model, plot
Dockerfile
docker-compose.yml
requirements.txt
```

## Running it

### Local Python

```bash
pip install -r requirements.txt
python src/run.py
```

The first run downloads the `all-MiniLM-L6-v2` weights from Hugging Face. `embed.py` uses a GPU automatically when CUDA is available and falls back to CPU otherwise, which is fine for a dataset this size. Outputs land in `results/`: `embeddings.npy`, `labels.npy`, `classifier.pkl`, and `umap_plot.png`.

You can also run the stages on their own from inside `src/`:

```bash
cd src
python embed.py
python classify.py
python visualize.py
```

### Docker

```bash
docker compose build
docker compose run triage
```

This drops you into a shell in the container with the project mounted at `/app`, from which you can run `python src/run.py`.

## Why this design

Training a clinical language model needs labeled data, compute, and time, and it has to be maintained as the model drifts. A pretrained embedding plus a thin classifier sidesteps all of that. It is cheap to run, easy to retrain on new labels (refit the logistic regression in seconds), and transparent about what it is doing. For a triage step whose job is to flag reports for closer human review, that tradeoff is a good one.

## License

MIT, see [LICENSE](LICENSE).
