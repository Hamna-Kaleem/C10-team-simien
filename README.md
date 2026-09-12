#Agricultural Extension RAG: Smart Retrieval for Farmers

# Agricultural Extension RAG — Smart Retrieval for Farmers

A two-stage retrieval-and-rerank system that returns the 5 most relevant
agricultural-extension documents for a farmer's natural-language query
(crop diseases, pests, deficiencies, weather risks, etc.), built for the
*Agricultural Extension RAG: Smart Retrieval for Farmers* Kaggle competition.

**Pipeline:** BM25 candidate retrieval → slot-based feature engineering →
LightGBM LambdaMART reranker → `BAAI/bge-reranker-base` cross-encoder
fine-tuned on the task, evaluated with leak-free cross-validation.

## 1. Dataset

The data is the official competition dataset (Kaggle: *Agricultural
Extension RAG — Smart Retrieval for Farmers*), consisting of four CSVs:

- `documents.csv` — the extension-service knowledge base: `document_id`,
  `title`, `text`, and a `crop` tag (or `(general)` when crop-agnostic).
- `train_queries.csv` — farmer queries with `query_id`, `query`.
- `qrels_train.csv` — relevance judgments per query (`document_id`,
  `relevance` on a 0–3 scale).
- `test_queries.csv` — held-out queries requiring a top-5 ranking each.

The dataset was provided by the competition organizers; no external data
was collected. All text is short, structured extension-service articles
(titles follow predictable patterns, e.g. *"Controlling Fall Armyworm in
Maize (Northern Zone)"*), which we exploit heavily in feature design (see
below).

## 2. Training Pipeline

**Preprocessing / slot extraction.** A shared regex-based parser
(`parse_slots`) extracts four "slots" from any text (query or document
title): **crop**, **issue** (e.g. "fall armyworm", "nitrogen deficiency"),
**intent** (prevention / symptom / cause / adaptation / impact /
treatment), and **zone** (agro-ecological zone parsed from a trailing
`(...)` in titles). Vocabularies for crops/issues/zones are mined
automatically from the document titles and the `crop` column, so the
parser needs no external gazetteer.

**Candidate generation.** For each query we retrieve the BM25 top-150
documents, then inject any document sharing a specific issue or crop slot
with the query (slot injection), to guarantee recall isn't limited purely
by lexical overlap.

**Feature engineering.** 16 features per (query, candidate) pair: BM25
score/rank, TF-IDF cosine similarity, a deterministic **rubric score**
(`expected_rel`, a hand-coded relevance rule based on issue/crop/intent
agreement — used both as a feature and as a standalone sanity-check
baseline), slot-match indicators (issue, crop, intent, zone), token
overlap with title/body, and length features.

**Stage 1 — LightGBM LambdaMART.** An `LGBMRanker` (`objective=lambdarank`,
250 trees, `learning_rate=0.05`, `num_leaves=15`, `label_gain=[0,1,2,3]`)
is trained on the 16 features to produce a first re-ranking of the BM25
candidates. Hyperparameters were chosen manually (shallow trees + strong
subsampling/regularization) to avoid overfitting given the small number
of labeled queries, and validated with leak-free group cross-validation.

**Stage 2 — Cross-encoder fine-tuning.** `BAAI/bge-reranker-base` is
fine-tuned (`sentence-transformers` `CrossEncoder`, 3 epochs, batch size
16, regression on `relevance/3`) directly on the qrels. At inference, the
LightGBM model's top-50 candidates are re-scored by the fine-tuned
cross-encoder, and the top 5 are returned.

**Design rationale.** The rubric scorer gives a deterministic, leak-free
performance ceiling to sanity-check every downstream stage; LightGBM adds
cheap, interpretable tabular signal on top of BM25; the cross-encoder adds
deep semantic matching where lexical/slot overlap is insufficient. Each
stage is only kept if it beats the previous stage's leak-free CV score.

## 3. Evaluation

- **Metric:** nDCG@5, matching the competition's official metric.
- **Recall check:** before reranking, we verify what fraction of
  known-relevant documents fall inside the BM25 top-150 candidate pool
  (recall must be high or downstream reranking can never recover).
- **Leak-free cross-validation:** `GroupKFold` on `query_id` (never
  splitting a query's candidates across folds) is used for both the
  LightGBM-only CV and the **full pipeline CV** (LightGBM → cross-encoder,
  cell 11 — "the gate"), where the cross-encoder is *re-trained per fold*
  on only that fold's training queries to avoid leakage. This full-pipeline
  score is what we compare against the last submitted leaderboard score
  before deciding whether to resubmit.
- **In-sample check:** cell 10 also reports an in-sample nDCG@5 using the
  cross-encoder trained on all data — clearly labeled "optimistic" and
  never used for model-selection decisions.

## 4. Reproduction

Requirements: Python 3, `pandas`, `numpy`, `scikit-learn`, `lightgbm`,
`sentence-transformers`, `torch` (auto-installed if missing). Internet
access is required to download `BAAI/bge-reranker-base`.

1. Place the four competition CSVs under a `data/` folder (or point the
   `base` path in the first cell to their location).
2. Run the notebook top to bottom (**Kernel → Restart & Run All**) — it is
   a single linear pipeline:
   `01_load_data → 02_metric → 03_slots → 04_bm25_tfidf → 05_rubric →
   06_candidates_features → 07_recall_check → 08_lightgbm_cv →
   09_train_final_lightgbm → 10_crossencoder → 11_full_pipeline_cv →
   12_build_submission`.
3. Check the printed **leak-free CV score in step 11** before trusting the
   output — it is the gate that decides whether the submission is an
   improvement.
4. The final ranked predictions are written to `submission.csv`
   (`QueryId, DocumentId`, 5 rows per query, ranked best-first).

## Appendix

**Team / Contributors:** _<Hamna Kaleem, Kamaya Ndigwa Espérance Martine>_

