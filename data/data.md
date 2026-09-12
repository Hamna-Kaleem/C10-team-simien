
# Data: Agricultural Extension RAG — Smart Retrieval for Farmers

This document explains what the dataset is, how to download it, and how to
load and use it. Put this file at `docs/DATA.md` (or merge its content into
your main `README.md`'s Dataset section).

**Source:** [Kaggle competition — Agricultural Extension RAG: Smart
Retrieval for Farmers](https://www.kaggle.com/competitions/agricultural-extension-rag-smart-retrieval-for-farmers/data)
**License:** CC BY-NC-SA 4.0 (competition bundle); individual documents carry
their own `license`/`source_url` (see [Licensing](#licensing--attribution)).

## 1. What the dataset is

A document-retrieval benchmark for a farmer-facing RAG (retrieval-augmented
generation) system, in the style of TREC/BEIR. Given a smallholder farmer's
natural-language question (e.g. *"Why are my maize leaves turning
yellow?"*), the task is to rank the 5 most relevant documents out of a fixed
knowledge base of 695 short agricultural-extension articles covering crop
diseases, pests, nutrient deficiencies, soil, climate, and fertiliser
topics, focused on Sub-Saharan African smallholder agriculture.

Relevance is graded 0–3 per document:

| Score | Meaning |
|---|---|
| 3 | Directly answers the question (right crop, right issue, right intent) |
| 2 | A genuinely complementary facet of the same topic (e.g. symptoms supporting an identification query) |
| 1 | The same issue, but on a different crop |
| 0 | Not relevant — includes lexically-similar hard negatives that answer the wrong intent (e.g. prevention vs. causes) or a confusable problem |

The scoring metric is **nDCG@5** on both the public and private leaderboard;
only the query split differs between them. A provided TF-IDF baseline
scores **nDCG@5 = 0.551** on hidden test labels — that's the bar to beat.

## 2. Files

| File | Rows | Description |
|---|---|---|
| `documents.csv` | 695 | The knowledge base to retrieve from |
| `train_queries.csv` | 308 | Training queries, each with a convenience list of its relevant document IDs |
| `qrels_train.csv` | 4,194 | Graded relevance labels (0–3) for every (query, document) pair judged for the training queries |
| `test_queries.csv` | 200 | Held-out queries with no labels — what your submission is scored against |
| `sample_submission.csv` | 1,000 | The exact required submission format (200 queries × 5 rows) |
| `baseline_submission.csv` | 1,000 | A working TF-IDF submission (nDCG@5 = 0.551) to compare against |

### Column schemas

**`documents.csv`**

| Column | Type | Description |
|---|---|---|
| `document_id` | str | Unique ID — use as `DocumentId` in submissions |
| `title` | str | Document title |
| `text` | str | Document body (a short extension factsheet) |
| `source` | str | Attributed publisher style (FAO, CGIAR, Plantwise, IITA, ICRISAT, AGRA, …) |
| `crop` | str | Primary crop, or `(general)` for soil/climate topics |
| `country` | str | Example country context |
| `origin` | str | `synthetic` or `llm_grounded` |
| `source_url` | str | Source link for grounded documents (blank for synthetic ones) |
| `license` | str | License of the underlying source (CC-BY for grounded docs) |

**`train_queries.csv`**

| Column | Type | Description |
|---|---|---|
| `query_id` | int | Unique ID, training IDs start at 1 |
| `query` | str | The farmer's question |
| `positive_docs` | str | Space-separated `document_id`s with relevance ≥ 1 |

**`test_queries.csv`**

| Column | Type | Description |
|---|---|---|
| `query_id` | int | Unique ID, test IDs start at 1001 — submit as `QueryId` |
| `query` | str | The farmer's question |

**`qrels_train.csv`**

| Column | Type | Description |
|---|---|---|
| `query_id` | str | Query ID |
| `document_id` | str | Document ID |
| `relevance` | double | Graded relevance: 3, 2, 1, or 0 |

**`sample_submission.csv` / `baseline_submission.csv`**

| Column | Type | Description |
|---|---|---|
| `QueryId` | str | Test query ID |
| `DocumentId` | str | A retrieved document ID |

Row order within a `QueryId` block **is the ranking** — first row = rank 1
(best guess), fifth row = rank 5. Exactly 5 rows per test query.

## 3. How to download the data

You need a free Kaggle account and, for the CLI/API options, to have joined
the competition (accept its rules on the competition page first — the API
download will fail with a 403 until you do).

### Option A — Kaggle CLI (recommended for reproducibility)

```bash
pip install kaggle --break-system-packages   # or: pip install kaggle

# Get an API token: Kaggle → your profile picture → Settings → API →
# "Create New Token". This downloads kaggle.json.
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Download and unzip into a local data/ folder
kaggle competitions download \
  -c agricultural-extension-rag-smart-retrieval-for-farmers \
  -p data/
unzip -o data/agricultural-extension-rag-smart-retrieval-for-farmers.zip -d data/
```

You should now have `data/documents.csv`, `data/train_queries.csv`,
`data/qrels_train.csv`, `data/test_queries.csv`,
`data/sample_submission.csv`, and `data/baseline_submission.csv`.

### Option B — Kaggle web UI

1. Go to the [competition Data
   tab](https://www.kaggle.com/competitions/agricultural-extension-rag-smart-retrieval-for-farmers/data).
2. Click **Join Competition** (accept the rules) if you haven't already.
3. Click **Download All** and unzip the archive into your repo's `data/`
   folder.

### Option C — Inside a Kaggle Notebook

If you're running on Kaggle itself (as the original pipeline notebook
does), just attach the competition as a data source via the Notebook's
**Input** panel — no download needed. The data will appear under
`/kaggle/input/competitions/agricultural-extension-rag-smart-retrieval-for-farmers/`.

> **Do not commit the raw CSVs to a public GitHub repo** unless you've
> checked the competition rules permit redistribution — commit this guide
> and your download script instead, and instruct users to fetch the data
> themselves via Option A/B above.

## 4. Loading the data

```python
import pandas as pd
from pathlib import Path

data_dir = Path("data")

documents      = pd.read_csv(data_dir / "documents.csv")
train_queries  = pd.read_csv(data_dir / "train_queries.csv")
qrels          = pd.read_csv(data_dir / "qrels_train.csv")
test_queries   = pd.read_csv(data_dir / "test_queries.csv")

print(documents.shape, train_queries.shape, qrels.shape, test_queries.shape)
# (695, 8) (308, 3) (4194, 3) (200, 2)
```

## 5. Minimal working example (TF-IDF baseline)

This reproduces the kind of baseline that scores nDCG@5 = 0.551 — a good
smoke test that your data is loaded correctly before building anything
more elaborate.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
doc_mat = vec.fit_transform(documents["title"] + ". " + documents["text"])
sims = cosine_similarity(vec.transform(test_queries["query"]), doc_mat)

rows = []
for i, qid in enumerate(test_queries["query_id"]):
    for rank in sims[i].argsort()[::-1][:5]:      # best-first order
        rows.append({
            "QueryId": str(qid),
            "DocumentId": str(documents.iloc[rank]["document_id"]),
        })

pd.DataFrame(rows).to_csv("submission.csv", index=False)
```

## 6. Submission format checklist

- Long format CSV: one row per retrieved document, columns `QueryId,DocumentId` (no score column).
- Exactly **5 rows per test `QueryId`**, **1,000 rows total** for 200 test queries.
- Row order within a query's block **is the ranking** (row 1 = your best guess).
- Every test `QueryId` from `test_queries.csv` must appear exactly once as a group of 5.
- Compare your file's shape/format against `sample_submission.csv` before submitting.

A quick validation snippet:

```python
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["QueryId", "DocumentId"]
assert len(sub) == len(test_queries) * 5
assert sub.groupby("QueryId").size().eq(5).all()
assert set(sub["QueryId"].astype(str)) == set(test_queries["query_id"].astype(str))
assert not sub.duplicated(["QueryId", "DocumentId"]).any()
print("Submission format OK:", sub.shape)
```

## 7. Evaluation metric (nDCG@5), for local validation

```python
import numpy as np

def dcg(rels, k=5):
    v = np.asarray(list(rels)[:k], dtype=float)
    if len(v) == 0: return 0.0
    return float(np.sum(v / np.log2(np.arange(2, len(v) + 2))))

def evaluate_ndcg_at_5(predictions, qrels_frame=qrels):
    lookup = {qid: dict(zip(g["document_id"], g["relevance"]))
              for qid, g in qrels_frame.groupby("query_id")}
    scores = {}
    for qid, judged in lookup.items():
        ranked = predictions.get(qid, [])[:5]
        gains  = [judged.get(d, 0) for d in ranked]
        ideal  = sorted(judged.values(), reverse=True)[:5]
        idcg   = dcg(ideal)
        scores[qid] = dcg(gains) / idcg if idcg else 0.0
    return float(np.mean(list(scores.values()))), scores
```

Use this against `train_queries.csv`/`qrels_train.csv` (never against the
hidden test labels) to validate any model before submitting.

## Licensing & attribution

- **Synthetic documents** (`origin = synthetic`) are original content
  created for this competition, released under **CC0** (public domain).
  The `source` field is a stylistic label only, not a claim the text was
  copied from that organisation.
- **Grounded documents** (`origin = llm_grounded`) are derived from CC-BY /
  CC0 open-access materials, chiefly CGIAR/CGSpace. Their `source_url` and
  `license` columns must be retained on reuse. Materials with
  NonCommercial or NoDerivatives restrictions were excluded during
  dataset construction.
- The overall competition bundle is distributed under **CC BY-NC-SA 4.0**.
  When in doubt about a specific document, check its own `license` and
  `source_url` columns.
- This dataset is for **education and benchmarking only** — it is
  explicitly **not agronomic advice**, and synthetic/LLM-rewritten text may
  contain simplifications or errors.

Thanks to the CGIAR centres and partners whose open-access extension
research (via CGSpace) underlies the grounded portion of this corpus.
