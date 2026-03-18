"""
Autoresearch search algorithm. Single-file, pure Python.
This is the ONLY file the autonomous agent modifies.

Goal: Beat the BM25 baseline on composite_score = QPS * recall@K.

Usage: uv run search.py
"""

import math
import time
import heapq
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
    High-throughput BM25 with precomputed scores and tuple postings.

    Optimizations:
    1. Precomputed idf * tf_norm stored as (doc_id, score) tuples
    2. Persistent flat array accumulator (no per-query allocation)
    3. Tuple unpacking in for loop (faster than zip of two lists)
    4. Adaptive top-K extraction
    """

    def __init__(self):
        self.N = 0
        self.postings = {}    # term -> list of (doc_id, score) tuples
        self._scores = None
        self._touched = None

    def build_index(self, docs):
        self.N = len(docs)
        k1, b = BM25_K1, BM25_B

        raw_postings = defaultdict(list)
        doc_lens = [0] * self.N
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

        # Precompute BM25 partial scores as tuples
        N = self.N
        postings = {}
        for term, posts in raw_postings.items():
            n = df[term]
            idf = math.log((N - n + 0.5) / (n + 0.5) + 1.0)
            if idf <= 0:
                continue
            scored = []
            for doc_id, tf in posts:
                dl = doc_lens[doc_id]
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                scored.append((doc_id, idf * tf_norm))
            postings[term] = scored
        self.postings = postings

        # Persistent accumulators
        self._scores = [0.0] * self.N
        self._touched = []

    def search(self, query, top_k=10):
        query_terms = set(tokenize(query))
        if not query_terms:
            return []

        scores = self._scores
        touched = self._touched
        postings = self.postings
        touched_append = touched.append

        for term in query_terms:
            pl = postings.get(term)
            if pl is None:
                continue
            for did, sc in pl:
                if scores[did] == 0.0:
                    touched_append(did)
                scores[did] += sc

        if not touched:
            return []

        n_touched = len(touched)
        if n_touched <= top_k:
            touched.sort(key=lambda d: scores[d], reverse=True)
            result = list(touched)
        elif n_touched < top_k * 20:
            touched.sort(key=lambda d: scores[d], reverse=True)
            result = touched[:top_k]
        else:
            result = heapq.nlargest(top_k, touched, key=lambda d: scores[d])

        for did in touched:
            scores[did] = 0.0
        touched.clear()

        return result

    def memory_usage_mb(self):
        total_postings = sum(len(v) for v in self.postings.values())
        total_terms = len(self.postings)
        return (total_postings * 24 + total_terms * 80 + self.N * 8) / (1024 * 1024)


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

    print("\nBuilding index and running queries...")
    engine = FastSearchEngine()
    metrics = evaluate_search(engine, corpus, queries, ground_truth, TOP_K)

    print("\nRunning BM25 baseline for comparison...")
    baseline = BM25Baseline()
    baseline_metrics = evaluate_search(baseline, corpus, queries, ground_truth, TOP_K)

    t_end = time.time()

    print("\n---")
    print(f"composite_score:  {metrics.composite_score:.2f}")
    print(f"qps:              {metrics.qps:.1f}")
    print(f"recall_at_{TOP_K}:       {metrics.recall_at_k:.6f}")
    print(f"avg_latency_ms:   {metrics.avg_latency_ms:.3f}")
    print(f"p99_latency_ms:   {metrics.p99_latency_ms:.3f}")
    print(f"index_build_s:    {metrics.index_build_s:.2f}")
    print(f"index_memory_mb:  {metrics.index_memory_mb:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")

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
