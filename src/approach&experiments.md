# Approach & Experiments

This explains what we changed and why, relative to the provided student
starter notebook.

## Where the starter left off

The starter notebook's ceiling was a single unsupervised lexical baseline
(TF-IDF cosine similarity) plus an *optional*, separately-scored dense
retrieval path (mean-pooled MiniLM embeddings, cosine similarity, no
fine-tuning). Both approaches treat retrieval as "does this document's
vocabulary or meaning overlap with the query" — neither distinguishes
*why* a document is on-topic. The dataset itself is explicit that this is
the exact failure mode being tested: relevance is graded partly on
**intent** (a document about the right crop and problem but the wrong
intent — prevention vs. causes, for example — is deliberately marked not
relevant), and the corpus contains hard negatives built specifically to
fool bag-of-words matching. A pure similarity score can't see that
distinction; it can only see word overlap.

That diagnosis is what everything below was built to fix.

## Experiment 1: Replace "similarity" with structured intent matching

Instead of trying to find a better embedding, the first real change was to
stop relying on free-text similarity as the *only* signal and parse both
queries and document titles into structured slots: **crop**, **issue**
(pest/disease/deficiency name), **intent** (prevention / symptom / cause /
adaptation / impact / treatment), and **zone**. This is a hand-written
regex layer, not a model — mined automatically from the document titles'
own vocabulary, so it needed no extra labeled data.

The payoff is a deterministic rule (`expected_rel`) that assigns a 0–3
relevance guess purely from slot agreement between a query and a
document — crop match + issue match + intent match → 3, issue match with
the wrong crop → 1, and so on. That rule was run as a **standalone
baseline ranker** before any learning was involved, specifically to
answer: *is structured slot-matching already a stronger signal than
TF-IDF similarity, on its own?* It also became a reusable sanity ceiling —
later learned models were only trusted once they could approach or beat
this deterministic rule, since if a trained model can't beat a fixed
if/else rule, the problem is almost certainly upstream (features or
leakage), not the model itself.

## Experiment 2: Swap the retriever for BM25 + a recall safety net

In the starter, candidate generation *was* the final ranking step —
TF-IDF similarity directly produced the top 5. We split that into two
separate jobs: a cheap, high-recall **candidate generator** (BM25
top-150, using a hand-built inverted index rather than TF-IDF) and a
separate **reranker** downstream. On top of BM25 we added slot injection —
any document sharing a specific issue or crop with the query gets added
to the candidate pool even if BM25's lexical score missed it — because a
plausible failure mode is a document phrased very differently from the
query but slot-identical to it.

Before trusting anything downstream, we explicitly measured what fraction
of known-relevant training documents survive into that candidate pool (a
recall check), since no reranker can recover a document that never made
the shortlist. That check is what justified `RECALL_K=150` and the
slot-injection fallback, rather than just trusting a larger BM25 cutoff on
faith.

## Experiment 3: Learn a reranker on structured features, not raw text

Rather than going down the starter's dense-retrieval path (embed
everything, cosine-similarity rank), we trained a supervised **LightGBM
LambdaMART** ranker on 16 engineered features per (query, candidate)
pair: BM25 score/rank, TF-IDF cosine, the rubric's `expected_rel`, and a
set of slot-match indicators (issue/crop/intent/zone agreement, token
overlap, lengths). The reasoning: with only 308 labeled training queries,
a small gradient-boosted model over interpretable structured features is
far less likely to overfit than fine-tuning or heavily tuning a dense
embedding model from scratch — and it directly exploits the exact signal
(slot agreement) the dataset's grading rubric is built around.

This is also where cross-validation had to be built more carefully than a
simple held-out split: rows had to be grouped by `query_id` (`GroupKFold`)
so that a single query's candidates never leak across train/validation —
the starter's own warning about keeping paraphrased/positive-doc-sharing
queries in the same fold generalizes here to "never split one query's own
candidate rows across folds."

## Experiment 4: Second-stage reranking — and the mistake that shaped the CV design

The starter's optional section stopped at embedding similarity. We went a
step further and fine-tuned a **cross-encoder** (`BAAI/bge-reranker-base`)
directly on the qrels, so it learns to score a `(query, document)` pair
jointly rather than comparing two separately-computed vectors — and
applied it only to the LightGBM model's top-50 candidates, to keep
inference cheap.

The important experiment here was a negative result: an early in-sample
check (cross-encoder trained on all queries, evaluated on those same
queries) looked almost perfect — but that's an artifact of the model
having already seen those exact labels during fine-tuning, not a real
generalization estimate. That result is what forced building a proper
**leak-free full-pipeline CV**: for each fold, both the LightGBM model
*and* the cross-encoder are retrained from scratch using only that fold's
training queries, then evaluated on the held-out fold. Only that number
is trustworthy.

## The decision rule used before submitting

Because the cross-encoder stage is far more expensive to train and run
than LightGBM alone, we didn't assume it was worth shipping just because
it's more sophisticated. The rule was: compare the leak-free
full-pipeline CV score against (a) the LightGBM-only CV score and (b) the
current leaderboard score, and only submit the two-stage pipeline if it
actually beat both. That comparison — not an assumption about which model
"should" be better — is what determined the final submission.

## Summary

| Stage | What changed vs. the starter | Why |
|---|---|---|
| Slot extractor | New: crop/issue/intent/zone parsing, absent from the starter | Targets the dataset's explicit intent-aware hard negatives, which pure similarity can't detect |
| Rubric scorer | New: deterministic relevance rule from slot agreement | Gives a leak-free sanity ceiling and a strong standalone feature |
| Candidate generation | BM25 + slot injection, replacing TF-IDF-as-final-ranker | Higher recall than TF-IDF alone before any reranking happens |
| Recall check | New: explicit measurement of positive-doc recall in the candidate pool | No reranker can fix candidates that never make the shortlist |
| Reranker (stage 1) | LightGBM LambdaMART on 16 engineered features, replacing raw cosine ranking | Learns to combine heterogeneous structured signals; low overfitting risk on a small labeled set |
| Reranker (stage 2) | Fine-tuned cross-encoder, replacing the starter's untuned embedding similarity | Jointly scores query-document pairs instead of comparing separate embeddings |
| Validation | Leak-free `GroupKFold` CV of the *entire* pipeline, with per-fold cross-encoder fine-tuning | Prevents the in-sample overfitting trap and gives an honest go/no-go signal before submitting |
