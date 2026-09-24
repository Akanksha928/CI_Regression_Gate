# RAG Quality Dashboard

## Baseline (last known-good)

| Metric             | Score |
|--------------------|-------|
| Answer Accuracy    | 0.833 |
| Retrieval Recall   | 0.833 |

## Check Definitions

| Check | Type | What it tests |
|-------|------|---------------|
| Answer accuracy | **Deterministic** | Key token from gold answer present in prediction |
| Retrieval recall | **Deterministic** | Correct doc appears in retrieved_doc_ids |
| Token overlap score | **AI-as-Judge** (offline rubric — no live model call) | Fraction of gold answer tokens matched in prediction |
| Release-note retrieval | **Adversarial** | Flags predictions where a `note-*` doc is top result — release notes never contain pricing/feature/latency info |

## Gate Logic

- **FAIL** (exit 1): `answer_accuracy < 0.833` OR `retrieval_recall < 0.833`
- **PASS** (exit 0): both metrics at or above baseline

## Run Results

| Run | Predictions File | Answer Acc | Retrieval Recall | Result |
|-----|-----------------|------------|-----------------|--------|
| Baseline | baseline_predictions.jsonl | 0.833 | 0.833 | ✅ PASS |
| Regressed | regressed_predictions.jsonl | 0.556 | 0.556 | ❌ FAIL |
| Improved | improved_predictions.jsonl | 0.944 | 0.944 | ✅ PASS |