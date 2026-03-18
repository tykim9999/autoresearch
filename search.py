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
        # Postings: term -> (doc_ids_list, scores_list)
        # Scores = precomputed idf * tf_norm
        self.posting_docs = {}
        self.posting_scores = {}
        self._scores = None   # persistent accumulator
        self._touched = None  # persistent touched list

    def build_index(self, docs):
        self.N = len(docs)
        k1, b = BM25_K1, BM25_B

        # Single pass: tokenize + build raw postings
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

        # Precompute full BM25 partial scores — use plain lists (fastest in CPython)
        # Also store max score per term for MaxScore pruning
        N = self.N
        self.term_max_score = {}  # term -> max score in its posting list
        for term, postings in raw_postings.items():
            n = df[term]
            idf = math.log((N - n + 0.5) / (n + 0.5) + 1.0)
            if idf <= 0:
                continue
            doc_ids = []
            scores = []
            max_sc = 0.0
            for doc_id, tf in postings:
                dl = doc_lens[doc_id]
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                sc = idf * tf_norm
                doc_ids.append(doc_id)
                scores.append(sc)
                if sc > max_sc:
                    max_sc = sc
            self.posting_docs[term] = doc_ids
            self.posting_scores[term] = scores
            self.term_max_score[term] = max_sc

        # Persistent accumulators
        self._scores = [0.0] * self.N
        self._touched = []

    def search(self, query, top_k=10):
        query_terms = set(tokenize(query))
        if not query_terms:
            return []

        scores = self._scores
        touched = self._touched

        # Gather matching terms and sort by max_score descending
        # Process high-impact terms first for better pruning potential
        posting_docs = self.posting_docs
        posting_scores = self.posting_scores
        term_max = self.term_max_score
        touched_append = touched.append

        matched_terms = []
        for term in query_terms:
            if term in posting_docs:
                matched_terms.append(term)

        if not matched_terms:
            return []

        # Sort terms: highest max-score first (process most impactful first)
        if len(matched_terms) > 1:
            matched_terms.sort(key=lambda t: term_max[t], reverse=True)

        # TAAT scoring with zip
        for term in matched_terms:
            for did, sc in zip(posting_docs[term], posting_scores[term]):
                if scores[did] == 0.0:
                    touched_append(did)
                scores[did] += sc

        if not touched:
            return []

        # Top-K: adaptive strategy
        n_touched = len(touched)
        if n_touched <= top_k:
            touched.sort(key=lambda d: scores[d], reverse=True)
            result = list(touched)
        elif n_touched < top_k * 20:
            touched.sort(key=lambda d: scores[d], reverse=True)
            result = touched[:top_k]
        else:
            result = heapq.nlargest(top_k, touched, key=lambda d: scores[d])

        # Sparse reset
        for did in touched:
            scores[did] = 0.0
        touched.clear()

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
