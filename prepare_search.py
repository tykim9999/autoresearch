"""
Search benchmark preparation and evaluation harness for autoresearch.
Downloads/builds a document corpus from existing data, generates queries,
and provides a fixed evaluation function.

This file is READ-ONLY for the autonomous agent. Do not modify.

Usage:
    python prepare_search.py                # full prep
    python prepare_search.py --num-docs 50000  # smaller corpus for testing
"""

import os
import sys
import time
import math
import json
import random
import argparse
import hashlib
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 60          # search benchmark time budget in seconds
NUM_QUERIES = 5000        # number of queries to evaluate
TOP_K = 10                # retrieve top-K results per query
MIN_DOC_LEN = 50          # minimum document length in characters
MAX_DOC_LEN = 2000        # maximum document length in characters
DEFAULT_NUM_DOCS = 20000  # default corpus size (fast ground truth computation)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
SEARCH_DIR = os.path.join(CACHE_DIR, "search_bench")
CORPUS_PATH = os.path.join(SEARCH_DIR, "corpus.json")
QUERIES_PATH = os.path.join(SEARCH_DIR, "queries.json")
GROUND_TRUTH_PATH = os.path.join(SEARCH_DIR, "ground_truth.json")

# ---------------------------------------------------------------------------
# Text processing utilities
# ---------------------------------------------------------------------------

# Simple but effective tokenizer for search - splits on non-alphanumeric,
# lowercases, filters short tokens. No external dependencies needed.
_STOP_WORDS = frozenset([
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "it", "its", "this",
    "that", "these", "those", "i", "you", "he", "she", "we", "they", "me",
    "him", "her", "us", "them", "my", "your", "his", "our", "their", "not",
    "no", "so", "if", "as", "up", "out", "about", "into", "over", "after",
])


def tokenize(text):
    """Simple whitespace + punctuation tokenizer with stopword removal."""
    tokens = []
    current = []
    for ch in text.lower():
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                word = "".join(current)
                if len(word) > 1 and word not in _STOP_WORDS:
                    tokens.append(word)
                current = []
    if current:
        word = "".join(current)
        if len(word) > 1 and word not in _STOP_WORDS:
            tokens.append(word)
    return tokens


# ---------------------------------------------------------------------------
# Corpus and query generation
# ---------------------------------------------------------------------------

def _load_texts_from_parquet(num_docs):
    """Load document texts from existing autoresearch data shards."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("pyarrow not available. Generating synthetic corpus instead.")
        return None

    data_dir = os.path.join(CACHE_DIR, "data")
    if not os.path.exists(data_dir):
        print(f"No data directory at {data_dir}. Run 'uv run prepare.py' first, or use synthetic data.")
        return None

    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet") and not f.endswith(".tmp"))
    if not files:
        return None

    docs = []
    for fname in files:
        if len(docs) >= num_docs:
            break
        pf = pq.ParquetFile(os.path.join(data_dir, fname))
        for rg_idx in range(pf.num_row_groups):
            if len(docs) >= num_docs:
                break
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                if len(text) >= MIN_DOC_LEN:
                    doc = text[:MAX_DOC_LEN]
                    docs.append(doc)
                    if len(docs) >= num_docs:
                        break
    return docs if docs else None


def _generate_synthetic_corpus(num_docs, seed=42):
    """Generate a synthetic corpus for benchmarking when real data unavailable."""
    rng = random.Random(seed)

    # Vocabulary pools for generating realistic-ish text
    topics = ["science", "technology", "history", "mathematics", "physics",
              "biology", "chemistry", "computer", "algorithm", "network",
              "database", "machine", "learning", "neural", "optimization"]
    words = [
        "research", "analysis", "method", "approach", "system", "model",
        "performance", "efficient", "parallel", "distributed", "index",
        "query", "search", "retrieval", "ranking", "document", "term",
        "frequency", "inverse", "weight", "score", "vector", "matrix",
        "sparse", "dense", "compression", "encoding", "hash", "table",
        "tree", "graph", "node", "edge", "path", "depth", "breadth",
        "binary", "linear", "logarithmic", "constant", "complexity",
        "benchmark", "evaluation", "precision", "recall", "accuracy",
        "throughput", "latency", "memory", "cache", "buffer", "stream",
        "batch", "pipeline", "concurrent", "atomic", "lock", "free",
        "structure", "implementation", "interface", "abstract", "concrete",
        "function", "variable", "parameter", "argument", "return", "value",
        "experiment", "result", "improvement", "baseline", "comparison",
        "significant", "statistical", "correlation", "distribution", "sample",
        "training", "validation", "testing", "cross", "fold", "split",
        "feature", "selection", "extraction", "transformation", "reduction",
        "cluster", "classification", "regression", "prediction", "inference",
    ]

    docs = []
    for i in range(num_docs):
        topic = rng.choice(topics)
        doc_len = rng.randint(20, 80)  # words per document
        doc_words = [topic]
        for _ in range(doc_len - 1):
            if rng.random() < 0.15:
                doc_words.append(rng.choice(topics))
            else:
                doc_words.append(rng.choice(words))
        docs.append(" ".join(doc_words))
    return docs


def _generate_queries_and_ground_truth(docs, num_queries, top_k, seed=123):
    """
    Generate queries by extracting key terms from documents,
    then compute ground truth using inverted-index BM25 scoring (fast).
    """
    rng = random.Random(seed)

    # Tokenize all documents
    doc_tokens = [tokenize(doc) for doc in docs]

    # Build inverted index and IDF for BM25 ground truth
    N = len(docs)
    df = Counter()
    inverted = defaultdict(list)  # term -> [(doc_id, tf), ...]
    for doc_id, tokens in enumerate(doc_tokens):
        tf = Counter(tokens)
        for term, count in tf.items():
            df[term] += 1
            inverted[term].append((doc_id, count))

    # Precompute document lengths
    doc_lens = [len(t) for t in doc_tokens]
    avgdl = sum(doc_lens) / N if N > 0 else 1.0

    # BM25 parameters for ground truth
    k1 = 1.2
    b = 0.75

    # Precompute IDF
    idf_cache = {}
    for term, n in df.items():
        idf_cache[term] = math.log((N - n + 0.5) / (n + 0.5) + 1.0)

    # Precompute length norm factors
    dl_factors = [k1 * (1 - b + b * doc_lens[i] / avgdl) for i in range(N)]

    # Generate queries from random document term samples
    queries = []

    # Prefer terms that appear in moderate number of docs
    interesting_terms = [
        term for term, count in df.items()
        if 5 <= count <= N * 0.3 and len(term) > 2
    ]
    if len(interesting_terms) < 100:
        interesting_terms = [term for term, count in df.items() if count >= 2]

    for _ in range(num_queries):
        num_terms = rng.randint(2, min(5, len(interesting_terms)))
        query_terms = rng.sample(interesting_terms, num_terms)
        queries.append(" ".join(query_terms))

    # Compute ground truth using inverted index (much faster than brute force)
    print("Computing ground truth BM25 rankings...")
    ground_truth = []
    for qi, query in enumerate(queries):
        if (qi + 1) % 2000 == 0:
            print(f"  Query {qi + 1}/{num_queries}...")
        q_terms = tokenize(query)
        scores = defaultdict(float)
        for term in q_terms:
            idf = idf_cache.get(term, 0)
            if idf <= 0:
                continue
            postings = inverted.get(term, [])
            for doc_id, tf in postings:
                tf_norm = (tf * (k1 + 1)) / (tf + dl_factors[doc_id])
                scores[doc_id] += idf * tf_norm
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        ground_truth.append([doc_id for doc_id, _ in ranked[:top_k]])

    return queries, ground_truth


def prepare_search_benchmark(num_docs=DEFAULT_NUM_DOCS):
    """Build corpus, queries, and ground truth for search benchmarking."""
    os.makedirs(SEARCH_DIR, exist_ok=True)

    # Check if already prepared
    if all(os.path.exists(p) for p in [CORPUS_PATH, QUERIES_PATH, GROUND_TRUTH_PATH]):
        print(f"Search benchmark already prepared at {SEARCH_DIR}")
        return

    print(f"Preparing search benchmark with {num_docs} documents...")

    # Try loading real data first, fall back to synthetic
    docs = _load_texts_from_parquet(num_docs)
    if docs is None:
        print("Using synthetic corpus...")
        docs = _generate_synthetic_corpus(num_docs)
    else:
        print(f"Loaded {len(docs)} documents from parquet data.")

    # Generate queries and ground truth
    queries, ground_truth = _generate_queries_and_ground_truth(
        docs, NUM_QUERIES, TOP_K
    )

    # Save
    with open(CORPUS_PATH, "w") as f:
        json.dump(docs, f)
    with open(QUERIES_PATH, "w") as f:
        json.dump(queries, f)
    with open(GROUND_TRUTH_PATH, "w") as f:
        json.dump(ground_truth, f)

    print(f"Saved {len(docs)} docs, {len(queries)} queries to {SEARCH_DIR}")


# ---------------------------------------------------------------------------
# Benchmark data loading (imported by search.py)
# ---------------------------------------------------------------------------

def load_corpus():
    """Load the document corpus."""
    with open(CORPUS_PATH) as f:
        return json.load(f)


def load_queries():
    """Load benchmark queries."""
    with open(QUERIES_PATH) as f:
        return json.load(f)


def load_ground_truth():
    """Load ground truth rankings."""
    with open(GROUND_TRUTH_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE - this is the fixed metric)
# ---------------------------------------------------------------------------

@dataclass
class SearchMetrics:
    """Results from a search benchmark evaluation."""
    qps: float              # queries per second
    recall_at_k: float      # recall@K (fraction of ground truth docs retrieved)
    avg_latency_ms: float   # average query latency in milliseconds
    p99_latency_ms: float   # 99th percentile query latency in ms
    index_build_s: float    # time to build the index in seconds
    index_memory_mb: float  # approximate index memory in MB
    composite_score: float  # combined metric (higher is better)


def evaluate_search(search_engine, corpus, queries, ground_truth, top_k=TOP_K):
    """
    Evaluate a search engine implementation.

    The search_engine must implement:
        - build_index(docs: list[str]) -> None
        - search(query: str, top_k: int) -> list[int]  (returns doc indices)

    Returns SearchMetrics with composite_score as the primary optimization target.
    Composite = QPS * recall@K (balances speed and quality).
    """
    # 1. Measure index build time
    t0 = time.time()
    search_engine.build_index(corpus)
    t1 = time.time()
    index_build_s = t1 - t0

    # 2. Estimate index memory (rough: measure process RSS delta isn't reliable,
    #    so we ask the engine if it has a memory_usage method, otherwise estimate)
    if hasattr(search_engine, "memory_usage_mb"):
        index_memory_mb = search_engine.memory_usage_mb()
    else:
        # Rough estimate: count stored data
        index_memory_mb = 0.0

    # 3. Run queries and measure latency
    latencies = []
    results = []
    for query in queries:
        t_q0 = time.perf_counter()
        result = search_engine.search(query, top_k)
        t_q1 = time.perf_counter()
        latencies.append((t_q1 - t_q0) * 1000)  # ms
        results.append(result)

    # 4. Compute recall@K
    total_recall = 0.0
    for result, truth in zip(results, ground_truth):
        if not truth:
            continue
        result_set = set(result[:top_k])
        truth_set = set(truth[:top_k])
        total_recall += len(result_set & truth_set) / len(truth_set)
    recall_at_k = total_recall / len(ground_truth) if ground_truth else 0.0

    # 5. Compute throughput and latency stats
    total_query_time = sum(latencies) / 1000  # seconds
    qps = len(queries) / total_query_time if total_query_time > 0 else 0
    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0
    sorted_lat = sorted(latencies)
    p99_idx = int(len(sorted_lat) * 0.99)
    p99_latency_ms = sorted_lat[min(p99_idx, len(sorted_lat) - 1)] if sorted_lat else 0

    # 6. Composite score: QPS * recall (higher = better)
    composite_score = qps * recall_at_k

    return SearchMetrics(
        qps=qps,
        recall_at_k=recall_at_k,
        avg_latency_ms=avg_latency_ms,
        p99_latency_ms=p99_latency_ms,
        index_build_s=index_build_s,
        index_memory_mb=index_memory_mb,
        composite_score=composite_score,
    )


# ---------------------------------------------------------------------------
# Reference BM25 baseline (Lucene-equivalent)
# ---------------------------------------------------------------------------

class BM25Baseline:
    """
    Standard BM25 implementation - the Lucene-equivalent baseline to beat.
    This is a clean, well-optimized pure Python BM25 with an inverted index.
    """

    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.inverted_index = {}   # term -> [(doc_id, tf), ...]
        self.doc_lens = []
        self.avgdl = 0.0
        self.N = 0
        self.df = {}               # term -> document frequency

    def build_index(self, docs):
        self.N = len(docs)
        self.doc_lens = []
        self.inverted_index = defaultdict(list)
        self.df = Counter()

        for doc_id, doc in enumerate(docs):
            tokens = tokenize(doc)
            self.doc_lens.append(len(tokens))
            tf = Counter(tokens)
            for term in tf:
                self.df[term] += 1
                self.inverted_index[term].append((doc_id, tf[term]))

        self.avgdl = sum(self.doc_lens) / self.N if self.N > 0 else 1.0
        # Convert to regular dict for faster lookups
        self.inverted_index = dict(self.inverted_index)
        self.df = dict(self.df)

    def search(self, query, top_k=10):
        query_terms = tokenize(query)
        scores = defaultdict(float)

        for term in query_terms:
            if term not in self.inverted_index:
                continue
            n = self.df.get(term, 0)
            idf = math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)
            for doc_id, tf in self.inverted_index[term]:
                dl = self.doc_lens[doc_id]
                tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                scores[doc_id] += idf * tf_norm

        # Return top-k doc IDs by score
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [doc_id for doc_id, _ in ranked[:top_k]]

    def memory_usage_mb(self):
        """Rough memory estimate of the index."""
        # Each posting: (doc_id int, tf int) ~ 56 bytes (Python overhead)
        total_postings = sum(len(v) for v in self.inverted_index.values())
        # Each term in index: ~100 bytes for string + list overhead
        total_terms = len(self.inverted_index)
        return (total_postings * 56 + total_terms * 100 + self.N * 8) / (1024 * 1024)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare search benchmark for autoresearch")
    parser.add_argument("--num-docs", type=int, default=DEFAULT_NUM_DOCS,
                        help="Number of documents in corpus")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print(f"Search benchmark directory: {SEARCH_DIR}")
    print()

    prepare_search_benchmark(args.num_docs)
    print()

    # Run baseline to show reference numbers
    print("Running BM25 baseline benchmark...")
    corpus = load_corpus()
    queries = load_queries()
    ground_truth = load_ground_truth()

    baseline = BM25Baseline()
    metrics = evaluate_search(baseline, corpus, queries, ground_truth)

    print()
    print("--- BM25 Baseline Results ---")
    print(f"composite_score:  {metrics.composite_score:.2f}")
    print(f"qps:              {metrics.qps:.1f}")
    print(f"recall_at_{TOP_K}:       {metrics.recall_at_k:.6f}")
    print(f"avg_latency_ms:   {metrics.avg_latency_ms:.3f}")
    print(f"p99_latency_ms:   {metrics.p99_latency_ms:.3f}")
    print(f"index_build_s:    {metrics.index_build_s:.2f}")
    print(f"index_memory_mb:  {metrics.index_memory_mb:.1f}")
    print()
    print("Done! This is the baseline to beat.")
