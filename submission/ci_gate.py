#!/usr/bin/env python3
"""
CI Regression Gate for RAG system.
Exits 0 (pass) if quality >= baseline, exits 1 (fail) if quality regressed.
"""

import json
import sys
import argparse

# ── hardcoded baseline ────────────────────────────────────────────────────────

BASELINE = {
    "answer_accuracy": 0.833,
    "retrieval_recall": 0.833
}

GROUND_TRUTH = [
    {"question_id": "q-price-0",   "answer": "$19 per month",  "doc_id": "plan-starter"},
    {"question_id": "q-refund-0",  "answer": "7 days",         "doc_id": "plan-starter"},
    {"question_id": "q-price-1",   "answer": "$49 per month",  "doc_id": "plan-team"},
    {"question_id": "q-refund-1",  "answer": "14 days",        "doc_id": "plan-team"},
    {"question_id": "q-price-2",   "answer": "$99 per month",  "doc_id": "plan-business"},
    {"question_id": "q-refund-2",  "answer": "30 days",        "doc_id": "plan-business"},
    {"question_id": "q-price-3",   "answer": "$299 per month", "doc_id": "plan-enterprise"},
    {"question_id": "q-refund-3",  "answer": "60 days",        "doc_id": "plan-enterprise"},
    {"question_id": "q-latency-0", "answer": "20 ms",          "doc_id": "region-us-east"},
    {"question_id": "q-latency-1", "answer": "28 ms",          "doc_id": "region-us-west"},
    {"question_id": "q-latency-2", "answer": "36 ms",          "doc_id": "region-eu-central"},
    {"question_id": "q-latency-3", "answer": "44 ms",          "doc_id": "region-ap-southeast"},
    {"question_id": "q-feature-0", "answer": "Team plan",      "doc_id": "feature-SSO"},
    {"question_id": "q-feature-1", "answer": "Team plan",      "doc_id": "feature-audit-logs"},
    {"question_id": "q-feature-2", "answer": "Business plan",  "doc_id": "feature-custom-roles"},
    {"question_id": "q-feature-3", "answer": "Business plan",  "doc_id": "feature-API"},
    {"question_id": "q-feature-4", "answer": "Enterprise plan","doc_id": "feature-webhooks"},
    {"question_id": "q-feature-5", "answer": "Enterprise plan","doc_id": "feature-data-export"},
]

# ── helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def index_by(records, key):
    return {r[key]: r for r in records}

# ── CHECK 1: DETERMINISTIC ────────────────────────────────────────────────────
# Hard token-match on answer + exact match on retrieved doc.

def check_deterministic(preds_by_id, truth_by_id):
    answer_hits = 0
    retrieval_hits = 0
    n = len(truth_by_id)

    for qid, gold in truth_by_id.items():
        pred = preds_by_id.get(qid, {})
        pred_answer = pred.get("answer", "").lower()
        pred_docs   = pred.get("retrieved_doc_ids", [])

        key_token = gold["answer"].lower().replace("per month", "").replace("plan", "").strip()
        if key_token in pred_answer:
            answer_hits += 1

        if gold["doc_id"] in pred_docs:
            retrieval_hits += 1

    return round(answer_hits / n, 3), round(retrieval_hits / n, 3)

# ── CHECK 2: AI-AS-JUDGE (offline rubric — no live model call) ────────────────
# Scores how many critical tokens from the gold answer appear in the prediction.

def check_judge(preds_by_id, truth_by_id):
    scores = []

    for qid, gold in truth_by_id.items():
        pred = preds_by_id.get(qid, {})
        pred_answer = pred.get("answer", "").lower()

        gold_tokens = set(
            gold["answer"].lower()
            .replace("$", "").replace("per month", "").replace("about", "")
            .split()
        )
        gold_tokens.discard("")

        if not gold_tokens:
            scores.append(0.0)
            continue

        matched = sum(1 for t in gold_tokens if t in pred_answer)
        scores.append(matched / len(gold_tokens))

    return round(sum(scores) / len(scores), 3)

# ── CHECK 3: ADVERSARIAL ──────────────────────────────────────────────────────
# Release notes (note-*) never contain pricing/feature/latency info.
# If one is the top retrieved doc, the RAG is retrieving junk.

def check_adversarial(preds_by_id):
    violations = []

    for qid, pred in preds_by_id.items():
        top_doc = pred.get("retrieved_doc_ids", [""])[0]
        if top_doc.startswith("note-"):
            violations.append({
                "question_id": qid,
                "bad_doc": top_doc,
                "answer": pred.get("answer", ""),
                "issue": "Release note retrieved for factual question"
            })

    return len(violations) == 0, violations

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()

    truth_by_id = index_by(GROUND_TRUTH, "question_id")
    preds_by_id = index_by(load_jsonl(args.predictions), "question_id")

    acc, recall     = check_deterministic(preds_by_id, truth_by_id)
    judge_score     = check_judge(preds_by_id, truth_by_id)
    adv_ok, adv_vio = check_adversarial(preds_by_id)

    print(f"answer_accuracy  : {acc}  (baseline {BASELINE['answer_accuracy']})")
    print(f"retrieval_recall : {recall}  (baseline {BASELINE['retrieval_recall']})")
    print(f"judge_score      : {judge_score}")
    print(f"adversarial_pass : {adv_ok}")
    if adv_vio:
        print(f"violations       : {json.dumps(adv_vio, indent=2)}")

    regressed = (
        acc    < BASELINE["answer_accuracy"] or
        recall < BASELINE["retrieval_recall"]
    )

    if regressed:
        print("\n❌ GATE FAILED — quality regressed vs baseline")
        sys.exit(1)
    else:
        print("\n✅ GATE PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()