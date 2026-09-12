# Notebook Walkthrough: Agricultural Extension RAG — LightGBM → Cross-Encoder

This document explains, cell by cell, what `91only.ipynb` does, why it's
built this way, and how the pieces fit together. The notebook implements
a **retrieve-then-rerank** system: cheap lexical retrieval narrows tens of
thousands of documents down to a shortlist, then two learned rerankers
(a gradient-boosted tree model, then a fine-tuned cross-encoder) refine
that shortlist into the final top-5 answer per query.

---

## Cell 0 (Markdown) — Overview

States the 12-step pipeline and two operational warnings: (1) internet
must be on during "Save & Run All" because step 10 downloads the
`bge-reranker-base` model from Hugging Face; (2) the reader should check
the leak-free CV number computed in cell 11 before trusting any submitted
result — that number is the actual estimate of unseen-data performance,
whereas earlier numbers (rubric score, in-sample cross-encoder score) are
either ceilings or optimistic estimates, not honest estimates.

## Cell 1 — Imports + Data Load

Loads four competition CSVs (`documents`, `train_queries`, `qrels_train`,
`test_queries`) from the Kaggle input path, with a fallback search
(`Path('/kaggle/input').glob('**/documents.csv')`) in case the exact
competition slug/path changes. It then builds several convenience arrays:

- `document_ids`: numpy array of every document's ID, used throughout as
  the canonical index into `documents`.
- `documents_titles`, `documents_bodies`, `documents_crop`: plain Python
  lists (faster iteration than repeated `.iloc` calls).
- `document_text`: `title + ". " + text`, i.e. the string that BM25/TF-IDF
  and the cross-encoder will actually see.
- `index_by_id`: a dict mapping `document_id -> row index`, since
  document IDs are not guaranteed to equal row position.

## Cell 2 — nDCG@5 Metric

Implements the competition's own evaluation metric so results can be
checked locally without submitting to the leaderboard.

- `dcg(rels, k=5)`: standard discounted cumulative gain,
  `sum(rel_i / log2(i+2))` for the first `k` ranked items.
- `evaluate_ndcg_at_5(predictions, qrels_frame)`: for every query with
  ground-truth relevance judgments, it looks up the relevance of each of
  the model's top-5 predicted documents (`gains`), computes the **ideal**
  DCG (`idcg`) by sorting the true relevance labels for that query in
  descending order and taking the top 5, and returns `dcg/idcg` per query
  (0 if there's nothing relevant to find). The function returns both the
  mean score across queries and a per-query score dictionary, which is
  useful for later error analysis.

## Cell 3 — Shared Slot Extractor (crop / issue / intent / zone)

This is the most bespoke part of the pipeline: a regex-based mini
information-extraction system that turns any free-text string (a query,
or a document title) into a structured dictionary of "slots". It is
applied identically to both queries and documents so that features built
from it are directly comparable.

- **Crops** (`CROPS`): collected from the `documents['crop']` column
  (excluding the generic `(general)` placeholder), sorted longest-first
  so multi-word crop names match before their substrings.
- **Zones** (`ZONES`): document titles that end in a parenthetical, e.g.
  `"... (Northern Zone)"`, are mined for the zone name via regex.
- **Issues** (`ISSUES`): built from three heuristics on document titles:
  (a) `"<nutrient> deficiency"` patterns; (b) a hand-built list of action
  verbs/heads (`what causes`, `controlling`, `preventing`,
  `identifying`, `correcting`, ...) followed by a noun phrase, cut off at
  a connective word (`in`, `on`, `apart`, `spreads`, `tolerance`); (c)
  `"adapting to X (...)"` and `"X: the risk to crops"` title patterns.
  A small generic set (`pests`, `weeds`, `disease`, ...) is added
  separately (`GENERIC_ISSUES`) since these are common but non-specific.
- **Intent** (`detect_intent`): a query/title is classified into one of
  six categories — `prevention`, `symptom`, `cause`, `adaptation`,
  `impact`, `treatment` — using an ordered list of keyword/phrase
  triggers (`INTENT_ORDER`). Order matters: the first category whose
  trigger phrase appears wins, so more specific categories are checked
  before more general ones.
- `parse_slots(text)`: applies all of the above to a lowercased string,
  returning `crop`, `issue` (all matches), `issue_spec` (issue matches
  excluding the generic set), `issue_gen` (only the generic matches),
  `intent`, and `zone`. It also de-duplicates overlapping issue matches
  (drops an issue string if it is a substring of another matched issue).
- `doc_slots`: `parse_slots` applied to every document title, then the
  document's own `crop` column value is merged in (since crop is given
  as structured metadata for documents but must be inferred from text for
  queries).
- `coverage(...)`: a diagnostic that prints, for a set of texts, what
  fraction have a detected crop / issue / intent, and what fraction are
  "fully resolved" (have both an issue and an intent) — run once each on
  document titles, training queries, and test queries, as a data-quality
  check on the parser before it's relied on for features.

## Cell 4 — BM25 + TF-IDF

Two classic lexical retrieval/similarity models, built from scratch and
via scikit-learn respectively:

- `tokenize`: lowercases and strips punctuation, then whitespace-splits.
- `BM25` (custom class): standard Okapi BM25 with `k1=1.5`, `b=0.75`.
  Builds document-frequency-based IDF weights and an inverted index
  (`postings`: word → list of `(doc_index, term_frequency)`), so
  `scores(query_tokens)` only touches documents that actually contain a
  query term rather than scanning the whole corpus.
- `TfidfVectorizer` (scikit-learn): unigrams+bigrams, English stop-words
  removed, `min_df=2`, sublinear TF scaling, L2-normalized. Used later
  for a cosine-similarity feature that captures softer lexical overlap
  than BM25's exact term matching (bigrams especially help match short
  phrases like "fall armyworm").

BM25 is used for the fast **candidate generation** stage (cell 6); TF-IDF
cosine similarity is used as one input **feature** to the rerankers
(cells 6, 8, 10), not for candidate generation itself.

## Cell 5 — Rubric Scorer (Deterministic Ceiling)

A hand-written, fully deterministic relevance rule, independent of any
learned model. Given a query's slots (`qs`) and a document's slots
(`ds`):

- `crop_relation(qc, dc)`: `'same'` if both are empty (crop-agnostic
  content is compatible with any query) or if they share at least one
  crop; `'diff'` otherwise.
- `expected_rel(qs, ds)`: assigns 0–3 by a small decision tree — 3 if the
  query has a specific issue that matches, crops agree, and intent
  agrees; 2 if issue+crop match but intent doesn't (or a generic issue
  matches with crop and intent agreement); 1 for weaker partial matches;
  0 otherwise. This mirrors, in code, what a human annotator likely used
  to assign the competition's own relevance labels.
- `rank_by_rubric(query, k=5)`: ranks all documents by `(-expected_rel,
  -bm25_score)` — i.e. the rubric score is the primary sort key and BM25
  breaks ties — and returns the top 5.
- The rubric's nDCG@5 on the **training set** is printed as a sanity
  ceiling: since it uses no learned parameters and can directly "cheat"
  by encoding the presumed labeling logic, if the learned models can't
  beat or approach it, something is likely wrong with feature engineering
  or training.

## Cell 6 — Candidate Generation + Feature Builder

- `get_candidates(query, inject=None)`: retrieves the BM25 top-150
  documents (`RECALL_K=150`), then adds any document that shares a
  specific issue or crop slot with the query but fell outside the BM25
  top-150 ("slot injection" — a recall safety net for cases where
  wording differs but the topic slot matches). `inject` optionally forces
  specific document IDs into the candidate set — used only during
  **training-feature construction** (cell 6/7) to guarantee every known
  positive document is represented in the training data, even if BM25
  and slot injection miss it; it is never used at inference/test time.
- `FEATURES`: the fixed list of 16 column names used by both the
  LightGBM and (indirectly, as an ensembling signal) cross-encoder
  stages: BM25 score/rank, TF-IDF cosine, rubric `expected_rel`, several
  binary/count slot-match features, token-overlap ratios with title and
  body, and length features.
- `build_rows(query, cand_ids, bm_scores, label_lookup=None)`: for each
  candidate document, computes all 16 features plus (during training)
  the ground-truth `relevance` label from `label_lookup`. Uses
  `crop_relation`/`expected_rel` from cell 5 and the slot sets computed
  in cell 3.
- The training loop builds `train_features`, a long-format DataFrame with
  one row per (query, candidate document) pair, positives injected so
  they're always present, and prints the resulting shape and count of
  positive-relevance rows — a check that there's enough positive signal
  to train a ranker.

## Cell 7 — Recall Check (Test-Time Realism)

Re-runs `get_candidates` **without** the `inject` argument (i.e. exactly
as it will run at test time) and measures what fraction of known-relevant
training documents are actually captured in the BM25(+slot) candidate
pool. This is the single most important sanity check in the notebook:
no downstream reranker — however good — can retrieve a relevant document
that never made it into the candidate set. A low recall here would mean
`RECALL_K` needs to increase or the slot-injection logic needs to be
broadened.

## Cell 8 — LightGBM LambdaMART Reranker + Leak-Free CV

- `make_ranker()`: constructs an `LGBMRanker` configured for **learning
  to rank** (`objective='lambdarank'`), with `label_gain=[0,1,2,3]`
  matching the 0–3 relevance scale. Hyperparameters (`num_leaves=15`,
  `min_child_samples=10`, `subsample=0.9`, `colsample_bytree=0.8`,
  `reg_lambda=0.5`) are all on the conservative side — shallow trees and
  heavy regularization/subsampling — appropriate given a relatively small
  number of labeled queries, to reduce the LightGBM model's tendency to
  overfit the 16-feature training set.
- `cv_lgb(features_df, n_splits=5)`: **leak-free** cross-validation using
  `GroupKFold` grouped by `query_id`, so all rows belonging to one query
  always stay in the same fold (never split a query's candidates across
  train/validation). For each fold, a fresh ranker is trained on the
  other folds' data, then used to re-score that fold's queries — but
  critically, the validation predictions are generated by re-running
  `get_candidates`/`build_rows` **without label injection**, so the
  evaluation reflects genuine test-time candidate generation, not the
  training-time candidate set that had positives force-injected. The
  resulting predictions across all folds are scored with
  `evaluate_ndcg_at_5` from cell 2.
- The printed `lgb_cv` score is the first honest (non-ceiling,
  non-optimistic) estimate of how well the learned reranker generalizes.

## Cell 9 — Train Final LightGBM

Once cross-validation in cell 8 has validated the approach, this cell
trains one final `LGBMRanker` (`final_model`) on **all** training queries
(no held-out fold) using the same hyperparameters, since more training
data generally improves a production model and CV has already given an
honest performance estimate.

## Cell 10 — Cross-Encoder (bge-reranker-base) Fine-Tuning

Adds a second, more powerful reranking stage using a pretrained
cross-encoder (`BAAI/bge-reranker-base`, a BERT-style model that scores a
`(query, document)` pair jointly, rather than embedding them separately).

- Ensures `sentence-transformers` is installed and pins
  `CUDA_VISIBLE_DEVICES=0` to avoid a known `DataParallel` bug when
  multiple GPUs are visible.
- `doc_text_by_id`: a dict for O(1) lookup of a document's full text by
  ID (needed repeatedly for building query/document pairs).
- `train_ce(train_qids=None)`: builds `InputExample(texts=[query,
  document_text], label=relevance/3)` training pairs from `qrels`
  (relevance rescaled from 0–3 to 0–1, since the cross-encoder is trained
  as a regression head with `num_labels=1`), optionally restricted to a
  subset of query IDs (used later for fold-specific fine-tuning in cell
  11). Fine-tunes for `CE_EPOCHS=3` epochs, batch size 16, with 100
  warmup steps.
- `retrieve_ce(query, k=5, model=None, first_stage_n=50)`: the actual
  two-stage inference function used both for evaluation and for the
  final submission. It (1) gets BM25+slot candidates, (2) scores them
  with the trained LightGBM `final_model`, (3) keeps only the top 50
  by LightGBM score (`FIRST_STAGE_N`) to limit how many pairs the
  (expensive) cross-encoder has to score, (4) re-scores those 50 with the
  fine-tuned cross-encoder, and (5) returns the final top-`k` by
  cross-encoder score.
- An **in-sample** nDCG@5 is printed using the cross-encoder trained on
  the full training set and evaluated on the same training queries — the
  code explicitly labels this "optimistic" because the cross-encoder has
  already seen these exact queries/labels during fine-tuning, so this
  number is not a valid generalization estimate. It exists only as a
  quick smoke test that the cross-encoder stage is wired up correctly.

## Cell 11 — Leak-Free CV of the Full Pipeline ("The Gate")

The most important cell for making a go/no-go submission decision.
`cv_full_pipeline(n_splits=5)` repeats the same `GroupKFold`-by-`query_id`
strategy as cell 8, but now for the **entire two-stage pipeline**:

1. Split training queries into 5 folds by `query_id`.
2. For each fold: train a fold-specific LightGBM ranker on the other
   folds' feature rows, **and** fine-tune a fold-specific cross-encoder
   (`train_ce(train_qids=tr_qids)`) using only that fold's training
   queries — this is essential to avoid leakage, since a cross-encoder
   trained on the held-out queries' labels would otherwise "already know
   the answer" for its own validation fold.
3. For each held-out query in that fold, generate candidates exactly as
   at test time, re-rank with the fold's LightGBM model, take the top 50,
   re-rank those with the fold's cross-encoder, and record the top 5.
4. Score all held-out predictions together with `evaluate_ndcg_at_5`.

The final printed line explicitly compares this score against the
LightGBM-only CV score (`lgb_cv`) and against the last leaderboard score
(hardcoded as `0.92` here) — the notebook's own guidance is: **only
submit the cross-encoder pipeline if this leak-free score beats the
current leaderboard score**, otherwise the added complexity/training time
of the cross-encoder isn't paying for itself and the LightGBM-only
submission should be preferred instead.

## Cell 12 — Build Submission

Runs `retrieve_ce` (with the full-data `final_model` and `ce` from cells
9–10, not fold-specific models) over every query in `test_queries`,
collecting the top-5 document IDs per query into a long-format
`(QueryId, DocumentId)` DataFrame matching the competition's required
submission format. Before writing the file, five assertions validate
structural correctness:

- exact column names/order (`QueryId`, `DocumentId`);
- exactly `5 × num_test_queries` rows;
- query order matches `test_queries`' original order (rank position
  within a query is implied by row order);
- each query has exactly 5 predicted documents;
- no duplicate `(QueryId, DocumentId)` pairs;
- every predicted `DocumentId` actually exists in `documents.csv`.

The validated submission is written to `/kaggle/working/submission.csv`
and its head is displayed as a final visual check.

---

## Summary of Design Choices

| Stage | Purpose | Why this design |
|---|---|---|
| Slot extractor | Turn free text into structured crop/issue/intent/zone fields | Titles/queries follow predictable patterns; regex is fast, needs no labeled data, and is fully interpretable |
| BM25 + slot injection | Cheap, high-recall candidate generation | Keeps the expensive rerankers from having to score the entire corpus per query |
| Rubric scorer | Deterministic sanity ceiling & feature | Encodes a plausible labeling rule directly; used to check learned models aren't underperforming an obvious heuristic |
| LightGBM LambdaMART | Fast, tabular first-stage reranker | Learns to combine 16 heterogeneous features; cheap enough to score hundreds of candidates per query |
| bge-reranker cross-encoder | Deep semantic second-stage reranker | Catches relevant matches that lexical/slot features miss; only applied to the LightGBM top-50 to control cost |
| Leak-free GroupKFold CV | Honest generalization estimate | Prevents both feature leakage (label-injected candidates) and cross-encoder leakage (fold-specific fine-tuning) |
