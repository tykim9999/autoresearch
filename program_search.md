# autoresearch — Faster Search Algorithm

This experiment uses the autoresearch framework to autonomously discover a search algorithm that outperforms Lucene-style BM25 on both **throughput (queries/sec)** and **retrieval quality (recall@K)**.

## Setup

1. **Agree on a run tag**: e.g. `search-mar18`.
2. **Create the branch**: `git checkout -b autoresearch/<tag>`.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `prepare_search.py` — fixed benchmark data, evaluation harness, BM25 baseline. **Do not modify.**
   - `search.py` — the file you modify. Search algorithm, index structure, query processing.
4. **Verify data exists**: Run `uv run prepare_search.py` to generate the benchmark corpus, queries, and ground truth.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go.**

## The Goal

**Beat the BM25 baseline** on `composite_score = QPS × recall@K`.

This means you can win by:
- Being **faster** (higher QPS) at equal recall
- Having **better recall** at equal speed
- Or ideally **both**

The BM25 baseline in `prepare_search.py` represents a clean Lucene-equivalent implementation. Your job is to find algorithmic improvements that outperform it.

## Experimentation

Each experiment runs `uv run search.py`. It builds an index, runs all queries, and compares against the baseline automatically.

**What you CAN do:**
- Modify `search.py` — everything is fair game: index structures, scoring functions, query processing, data layout, caching strategies, algorithmic shortcuts.

**What you CANNOT do:**
- Modify `prepare_search.py`. The evaluation harness, corpus, queries, and ground truth are fixed.
- Install new packages. Use only what's in `pyproject.toml`.
- Cheat the benchmark (e.g., memorizing ground truth, skipping queries).

## Ideas to Explore

Here are research directions that could beat BM25:

### Data Structure Optimizations
- **Compressed posting lists** — delta-encode doc IDs, use variable-byte or bit-packing
- **Skip lists** — skip pointers for faster posting list intersection
- **Block-max indexes** — store max scores per block for early termination (BMW/WAND)
- **Hash-based inverted index** — faster term lookups than dict

### Scoring Innovations
- **BM25+** or **BM25L** — improved BM25 variants that fix edge cases
- **TF-IDF with sublinear TF** — `1 + log(tf)` instead of raw TF
- **Language model scoring** — Dirichlet-smoothed LM instead of BM25
- **Learned sparse scoring** — term importance weights from corpus statistics

### Query Processing
- **WAND (Weak AND)** — skip documents that can't make top-K using score upper bounds
- **Block-Max WAND (BMW)** — block-level max scores for aggressive pruning
- **MaxScore** — partition terms into essential/non-essential for faster processing
- **Document-at-a-time (DAAT)** vs **Term-at-a-time (TAAT)** — try both
- **Early termination** — stop scoring when remaining terms can't change ranking

### Index Layout
- **Cache-friendly posting layouts** — arrays of structs vs struct of arrays
- **Frequency-sorted postings** — sort by TF for impact-ordered traversal
- **Tiered indexes** — separate high-TF and low-TF postings
- **Precomputed partial scores** — store IDF×TF at index time

### Hybrid Approaches
- **Two-phase retrieval** — fast candidate generation + precise re-ranking
- **Bloom filters** for negative term membership
- **Signature files** as a fast pre-filter

## Output Format

The script prints:
```
---
composite_score:  12345.67
qps:              5000.0
recall_at_10:     0.950000
avg_latency_ms:   0.200
p99_latency_ms:   0.500
index_build_s:    1.50
index_memory_mb:  45.2
total_seconds:    25.0

--- Comparison vs BM25 Baseline ---
baseline_composite: 10000.00
our_composite:      12345.67
qps_speedup:        1.25x
recall_delta:       -0.010000
composite_delta:    +2345.67

RESULT: BEATS BASELINE
```

## Logging Results

Log to `results.tsv` (tab-separated):

```
commit	composite_score	qps	recall	qps_speedup	status	description
```

1. git commit hash (short, 7 chars)
2. composite_score (QPS × recall)
3. qps achieved
4. recall@K
5. qps_speedup vs baseline (e.g. 1.25)
6. status: `keep`, `discard`, or `crash`
7. short description

Example:
```
commit	composite_score	qps	recall	qps_speedup	status	description
a1b2c3d	10000.00	5000.0	0.950000	1.00	keep	baseline (BM25 reference)
b2c3d4e	12345.67	6500.0	0.948000	1.30	keep	WAND early termination
c3d4e5f	8000.00	8000.0	0.500000	1.60	discard	too aggressive pruning killed recall
```

## The Experiment Loop

LOOP FOREVER:

1. Check git state
2. Edit `search.py` with an experimental idea
3. `git commit`
4. Run: `uv run search.py > run.log 2>&1`
5. Read results: `grep "^composite_score:\|^qps:\|^recall_at_\|^qps_speedup:" run.log`
6. If empty, it crashed — `tail -n 50 run.log` for traceback
7. Log to results.tsv
8. If composite_score improved → keep the commit
9. If equal or worse → `git reset` to previous best

**Timeout**: Each run should take < 2 minutes. Kill and discard if > 5 minutes.

**NEVER STOP**: Run indefinitely. The human may be away. Keep experimenting.
