"""
Autoresearch search algorithm. Single-file, pure Python.
This is the ONLY file the autonomous agent modifies.

Goal: Beat the BM25 baseline on composite_score = QPS * recall@K.

Usage: uv run search.py
"""

import math
import time
import heapq
import array
from collections import Counter, defaultdict

from prepare_search import (
    TIME_BUDGET, TOP_K, tokenize,
    load_corpus, load_queries, load_ground_truth,
    evaluate_search, BM25Baseline,
)

# ---------------------------------------------------------------------------
# Search Algorithm (edit everything below)
# ---------------------------------------------------------------------------

# BM25 hyperparameters
BM25_K1 = 1.2
BM25_B = 0.75


class FastSearchEngine:
    """
    High-throughput BM25 engine with precomputed partial scores.

    Key optimizations:
    1. Store precomputed idf * tf_norm in postings (no per-query math)
    2. Flat array score accumulator (no dict overhead)
    3. heapq.nlargest for top-K (avoids full sort)
    4. Posting lists as parallel arrays (doc_ids + scores) for locality
    5. Query term dedup to avoid redundant posting traversals
    """

    def __init__(self):
        self.N = 0
        # Postings stored as: term -> (doc_id_array, score_array)
        # where score = idf * tf_norm (fully precomputed)
        self.posting_docs = {}    # term -> array('i', [...])
        self.posting_scores = {}  # term -> array('f', [...])

    def build_index(self, docs):
        self.N = len(docs)
        k1, b = BM25_K1, BM25_B

        # Single pass: tokenize + build raw postings
        raw_postings = defaultdict(list)  # term -> [(doc_id, tf)]
        doc_lens = array.array('i', [0] * self.N)
        df = Counter()
        total_len = 0

        for doc_id, doc in enumerate(docs):
            tokens = tokenize(doc)
            dl = len(tokens)
            doc_lens[doc_id] = dl
            total_len += dl
            tf = Counter(tokens)
            for term, count in tf.items():
                df[term] += 1
                raw_postings[term].append((doc_id, count))

        avgdl = total_len / self.N if self.N > 0 else 1.0

        # Precompute full BM25 partial scores into posting lists
        N = self.N
        for term, postings in raw_postings.items():
            n = df[term]
            idf = math.log((N - n + 0.5) / (n + 0.5) + 1.0)
            if idf <= 0:
                continue
            doc_ids = array.array('i')
            scores = array.array('f')
            for doc_id, tf in postings:
                dl = doc_lens[doc_id]
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                doc_ids.append(doc_id)
                scores.append(idf * tf_norm)
            self.posting_docs[term] = doc_ids
            self.posting_scores[term] = scores

    def search(self, query, top_k=10):
        query_terms = set(tokenize(query))  # dedup
        if not query_terms:
            return []

        # Flat array accumulator - much faster than defaultdict for dense scoring
        scores = [0.0] * self.N
        touched = []  # track which doc_ids got scores

        for term in query_terms:
            doc_ids = self.posting_docs.get(term)
            if doc_ids is None:
                continue
            score_arr = self.posting_scores[term]
            n = len(doc_ids)
            for i in range(n):
                did = doc_ids[i]
                if scores[did] == 0.0:
                    touched.append(did)
                scores[did] += score_arr[i]

        if not touched:
            return []

        # Extract top-k using nlargest (O(n log k) vs O(n log n) for sort)
        if len(touched) <= top_k:
            touched.sort(key=lambda d: scores[d], reverse=True)
            result = touched
        else:
            result = heapq.nlargest(top_k, touched, key=lambda d: scores[d])

        # Reset accumulator for touched docs
        for did in touched:
            scores[did] = 0.0

        return result

    def memory_usage_mb(self):
        total_postings = sum(len(v) for v in self.posting_docs.values())
        total_terms = len(self.posting_docs)
        return (total_postings * 8 + total_terms * 80 + self.N * 8) / (1024 * 1024)


# ---------------------------------------------------------------------------
# Main: run benchmark and report results
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()

    print("Loading benchmark data...")
    corpus = load_corpus()
    queries = load_queries()
    ground_truth = load_ground_truth()
    print(f"Corpus: {len(corpus)} docs, Queries: {len(queries)}")

    # Run our search engine
    print("\nBuilding index and running queries...")
    engine = FastSearchEngine()
    metrics = evaluate_search(engine, corpus, queries, ground_truth, TOP_K)

    # Also run baseline for comparison
    print("\nRunning BM25 baseline for comparison...")
    baseline = BM25Baseline()
    baseline_metrics = evaluate_search(baseline, corpus, queries, ground_truth, TOP_K)

    t_end = time.time()

    # Results
    print("\n---")
    print(f"composite_score:  {metrics.composite_score:.2f}")
    print(f"qps:              {metrics.qps:.1f}")
    print(f"recall_at_{TOP_K}:       {metrics.recall_at_k:.6f}")
    print(f"avg_latency_ms:   {metrics.avg_latency_ms:.3f}")
    print(f"p99_latency_ms:   {metrics.p99_latency_ms:.3f}")
    print(f"index_build_s:    {metrics.index_build_s:.2f}")
    print(f"index_memory_mb:  {metrics.index_memory_mb:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")

    # Comparison
    print("\n--- Comparison vs BM25 Baseline ---")
    print(f"baseline_composite: {baseline_metrics.composite_score:.2f}")
    print(f"our_composite:      {metrics.composite_score:.2f}")
    speedup = metrics.qps / baseline_metrics.qps if baseline_metrics.qps > 0 else 0
    print(f"qps_speedup:        {speedup:.2f}x")
    recall_delta = metrics.recall_at_k - baseline_metrics.recall_at_k
    print(f"recall_delta:       {recall_delta:+.6f}")
    composite_delta = metrics.composite_score - baseline_metrics.composite_score
    print(f"composite_delta:    {composite_delta:+.2f}")

    if metrics.composite_score > baseline_metrics.composite_score:
        print("\nRESULT: BEATS BASELINE")
    else:
        print("\nRESULT: BELOW BASELINE")
