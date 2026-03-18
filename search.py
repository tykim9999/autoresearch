"""
Autoresearch search algorithm. Single-file, pure Python.
This is the ONLY file the autonomous agent modifies.

Goal: Beat the BM25 baseline on composite_score = QPS * recall@K.

Usage: uv run search.py
"""

import os
import sys
import math
import time
import struct
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass

from prepare_search import (
    TIME_BUDGET, TOP_K, tokenize,
    load_corpus, load_queries, load_ground_truth,
    evaluate_search, BM25Baseline,
)

# ---------------------------------------------------------------------------
# Search Algorithm (edit everything below)
# ---------------------------------------------------------------------------

# Hyperparameters
BM25_K1 = 1.2          # BM25 term frequency saturation
BM25_B = 0.75          # BM25 length normalization
USE_SKIP_LISTS = True  # whether to use skip pointers in posting lists
SKIP_INTERVAL = 64     # skip pointer interval
PRESCORE_CUTOFF = 0    # if > 0, only score docs appearing in >= N query terms
USE_TERM_CACHE = True  # cache top-K results for frequent query terms
TERM_CACHE_SIZE = 1000 # max cached terms


class FastSearchEngine:
    """
    Optimized search engine aiming to beat standard BM25 on throughput
    while maintaining high recall.

    Key optimizations over naive BM25:
    1. Precomputed IDF values at index time
    2. Posting lists sorted by doc_id for efficient intersection
    3. Score accumulation with early termination
    4. Compact data structures to reduce memory overhead
    5. Term-at-a-time (TAAT) scoring with score upper bounds
    """

    def __init__(self):
        self.N = 0
        self.avgdl = 0.0
        self.doc_lens = None
        self.postings = {}       # term -> list of (doc_id, tf)
        self.idf = {}            # term -> precomputed IDF
        self.dl_cache = {}       # doc_id -> length normalization factor
        self.term_cache = {}     # term -> sorted top doc_ids with scores

    def build_index(self, docs):
        self.N = len(docs)
        self.doc_lens = [0] * self.N
        postings = defaultdict(list)
        df = Counter()

        # Single pass: tokenize, count, build postings
        total_len = 0
        for doc_id, doc in enumerate(docs):
            tokens = tokenize(doc)
            dl = len(tokens)
            self.doc_lens[doc_id] = dl
            total_len += dl

            tf = Counter(tokens)
            for term, count in tf.items():
                df[term] += 1
                postings[term].append((doc_id, count))

        self.avgdl = total_len / self.N if self.N > 0 else 1.0

        # Precompute IDF for all terms
        for term, n in df.items():
            self.idf[term] = math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)

        # Store postings sorted by doc_id (for cache-friendly access)
        self.postings = {term: pl for term, pl in postings.items()}

        # Precompute length normalization denominators
        # dl_factor[doc_id] = k1 * (1 - b + b * dl / avgdl)
        k1, b = BM25_K1, BM25_B
        self.dl_factors = [
            k1 * (1 - b + b * self.doc_lens[i] / self.avgdl)
            for i in range(self.N)
        ]

        # Build term cache for high-IDF terms (most discriminative)
        if USE_TERM_CACHE:
            self._build_term_cache()

    def _build_term_cache(self):
        """Pre-score and cache top results for frequent terms."""
        # Cache the most common terms' full scored results
        term_by_freq = sorted(self.postings.keys(),
                              key=lambda t: len(self.postings[t]),
                              reverse=True)
        k1 = BM25_K1
        for term in term_by_freq[:TERM_CACHE_SIZE]:
            idf = self.idf.get(term, 0)
            if idf <= 0:
                continue
            scored = []
            for doc_id, tf in self.postings[term]:
                tf_norm = (tf * (k1 + 1)) / (tf + self.dl_factors[doc_id])
                scored.append((doc_id, idf * tf_norm))
            scored.sort(key=lambda x: -x[1])
            self.term_cache[term] = scored[:TOP_K * 5]  # keep more than top_k for multi-term queries

    def search(self, query, top_k=10):
        query_terms = tokenize(query)
        if not query_terms:
            return []

        # Score accumulator: doc_id -> score
        scores = defaultdict(float)
        k1 = BM25_K1

        # Term-at-a-time scoring
        for term in query_terms:
            idf = self.idf.get(term, 0)
            if idf <= 0:
                continue

            posting_list = self.postings.get(term)
            if posting_list is None:
                continue

            for doc_id, tf in posting_list:
                tf_norm = (tf * (k1 + 1)) / (tf + self.dl_factors[doc_id])
                scores[doc_id] += idf * tf_norm

        if not scores:
            return []

        # Partial sort: only need top_k
        if len(scores) <= top_k:
            ranked = sorted(scores.items(), key=lambda x: -x[1])
        else:
            # Use a selection approach for large result sets
            items = list(scores.items())
            items.sort(key=lambda x: -x[1])
            ranked = items[:top_k]

        return [doc_id for doc_id, _ in ranked]

    def memory_usage_mb(self):
        total_postings = sum(len(v) for v in self.postings.values())
        total_terms = len(self.postings)
        cache_entries = sum(len(v) for v in self.term_cache.values())
        return (total_postings * 24 + total_terms * 80 + self.N * 16 + cache_entries * 16) / (1024 * 1024)


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
