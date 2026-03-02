# Index Building: How the FAISS HNSW Index Gets Wired Into the Pipeline

## The Story So Far (What Existed Before These Changes)

The compression pipeline did exactly two jobs:

1. **Train a compressor** (PCA + K-means + quantization) on sampled tokens
2. **Compress every clip** into `.widx` shard files, consolidate metadata, and return the metadata path

The pipeline ended the moment all shards were written and metadata was consolidated. The return value was `config.consolidated_metadata_path` — a single `clips.parquet` file mapping every clip to its shard file and offset.

Here's what the output directory looked like after a successful run:

```
compressed/
├── compression_model/       # Trained PCA, centroids, quantization levels
│   ├── pca_components.npy
│   ├── pca_mean.npy
│   ├── centroids.npy
│   ├── quantile_thresholds.npy
│   ├── quantization_levels.npy
│   ├── explained_variance_ratio.npy
│   └── input_dim.npy
├── shards/                  # Binary shard files
│   ├── shard_00000000.widx
│   ├── shard_00000001.widx
│   └── ...
├── metadata/                # Per-shard metadata
│   ├── shard_00000000.parquet
│   ├── shard_00000001.parquet
│   └── ...
└── clips.parquet            # Consolidated metadata (all shards merged)
```

Each `.widx` shard file stores compressed clips in a fixed-size binary format. Each clip record contains three things:

```
┌────────────────────────┬──────────────────────────────┬──────────────────────┐
│  centroid_ids (uint16) │  quantized_residuals (uint8) │  coarse_vector (f32) │
│  [8192 tokens]         │  [8192 × 32 packed bytes]    │  [128 floats]        │
└────────────────────────┴──────────────────────────────┴──────────────────────┘
```

That **coarse_vector** is the mean of all 8,192 projected token vectors in the clip. It's a single 128-dimensional float32 vector that acts as a rough fingerprint of the entire clip. It's been sitting inside every shard record this whole time, but nothing was *using* it for search. These changes fix that.

---

## The Problem: We Have Fingerprints But No Way to Search Them

Imagine you have a library of 76,000 books. Each book has a short summary (the coarse vector). The summaries are all filed away inside the books themselves (inside the `.widx` shard files). If someone walks in and says "find me books similar to this one," you'd have to:

1. Open every single shard file
2. Read every single coarse vector
3. Compare the query against all 76,000 vectors
4. Return the closest matches

That's a brute-force linear scan. For 76,000 clips at 128 dimensions, it takes about 10ms — fast enough for a demo, but it doesn't scale. When you have millions of clips, you need a proper search index.

**That's what these changes add:** a FAISS HNSW index that lets you find the nearest coarse vectors in ~1ms regardless of dataset size.

---

## What Changed: File by File

### 1. New File: `index/__init__.py`

```python
from index.builder import IndexBuilder

__all__ = ["IndexBuilder"]
```

This is just a package entry point. It lets other code write `from index import IndexBuilder` instead of `from index.builder import IndexBuilder`. Nothing happens here — it's pure convenience.

### 2. New File: `index/builder.py` — The Core Addition

This is the only file with real new logic. Let's walk through it method by method.

#### The Class: `IndexBuilder`

```python
class IndexBuilder:
    def __init__(
        self,
        hnsw_m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128,
    ) -> None:
```

Three hyperparameters control the HNSW index quality/speed tradeoff. To understand them, you need to understand what HNSW is.

**HNSW (Hierarchical Navigable Small World)** is a graph-based search structure. Picture it like this:

```
Layer 3 (sparse):    A ──────────────────── D
                     │                      │
Layer 2 (medium):    A ──── B ──── C ──── D
                     │      │      │      │
Layer 1 (dense):     A ─ B ─ C ─ D ─ E ─ F ─ G ─ H
                     │ │ │ │ │ │ │ │ │ │ │ │ │ │ │
Layer 0 (all nodes): A B C D E F G H I J K L M N O
```

Every vector is a node on layer 0. A random subset appears on layer 1, a smaller subset on layer 2, and so on. To find a query's nearest neighbor:
1. Start at the top layer (very few nodes, big jumps)
2. Greedily walk toward the query
3. Drop down to the next layer and repeat
4. At layer 0, you're very close to the answer

The three hyperparameters:

- **`hnsw_m = 32`**: Each node connects to up to 32 neighbors. More connections = better recall but more memory. Think of it as "how many friends each node has." With 32 connections, a node at layer 0 can reach its 32 nearest coarse vectors in one hop.

- **`ef_construction = 200`**: During index building, how many candidates to consider when deciding each node's neighbors. Higher = slower build, better graph quality. With 200, FAISS evaluates 200 potential neighbors before choosing the best 32. EF btw stands for Exploration Factor

- **`ef_search = 128`**: During search, how many candidates to track while traversing the graph. Higher = slower search, better recall. With 128, the search keeps a priority queue of 128 "best so far" candidates as it walks the graph.

#### `build_faiss_index()`: Collecting Vectors and Building the Graph

```python
def build_faiss_index(self, compressed_dir: Path, output_path: Path) -> np.ndarray:
    shard_paths = iter_shard_paths(compressed_dir)
```

First, it finds all `.widx` shard files in the directory. `iter_shard_paths` (from `compression/shards.py`) just does a sorted glob:

```python
def iter_shard_paths(directory: Path) -> list[Path]:
    return sorted(path for path in directory.glob("*.widx") if path.is_file())
```

Sorting matters for determinism — the global clip ordering must be consistent. If shard_00000000 has clips 0–99 and shard_00000001 has clips 100–199, the index needs to know that vector index 150 maps to shard 1, clip 50.

```python
coarse_vectors = np.ascontiguousarray(
    np.concatenate(
        [read_coarse_vectors_from_shard(shard_path) for shard_path in shard_paths],
        axis=0,
    ).astype(np.float32, copy=False)
)
```

This loops over every shard, reads just the coarse vectors (via memory-mapped structured dtype access), and concatenates them into one big matrix.

Let's trace `read_coarse_vectors_from_shard` to see exactly how it works:

```python
def read_coarse_vectors_from_shard(shard_path: Path) -> np.ndarray:
    header = read_shard_header(shard_path)        # Read 32-byte header
    records = np.memmap(                           # Memory-map the rest
        shard_path,
        mode="r",
        dtype=header.record_dtype(),               # Structured dtype with named fields
        offset=_SHARD_HEADER_STRUCT.size,           # Skip past the 32-byte header
        shape=(header.clip_count,),                 # One record per clip
    )
    return np.asarray(records["coarse_vector"], dtype=np.float32)
```

The structured dtype looks like this (for 8192 tokens, 32 residual bytes, 128-dim coarse):

```python
np.dtype([
    ("centroid_ids",         np.uint16,  (8192,)),          # 16,384 bytes
    ("quantized_residuals",  np.uint8,   (8192, 32)),       # 262,144 bytes
    ("coarse_vector",        np.float32, (128,)),           # 512 bytes
])
```

Using `records["coarse_vector"]` plucks out just the coarse_vector field from every record — numpy handles the byte-offset arithmetic. No need to read the centroid IDs or residuals. It's like accessing a single column from a database table.

**Concrete example:** If we have 2 shard files with 100 and 100 clips each, both using pca_dim=128:

```
Shard 0: read_coarse_vectors_from_shard → shape (100, 128)
Shard 1: read_coarse_vectors_from_shard → shape (100, 128)
np.concatenate → shape (200, 128)
```

The result is a (200, 128) float32 matrix — every clip in the dataset represented by its 128-dimensional fingerprint.

#### Normalization: Why and How

```python
normalized_vectors = self._normalize_rows(coarse_vectors)
```

```python
def _normalize_rows(self, vectors: np.ndarray) -> np.ndarray:
    row_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(row_norms == 0.0, 1.0, row_norms)
    return np.ascontiguousarray(vectors / safe_norms, dtype=np.float32)
```

Each row gets divided by its L2 norm, so every vector sits on the unit sphere. For example:

```
Before: [3.0, 4.0]  →  norm = 5.0  →  After: [0.6, 0.8]
Before: [1.0, 0.0]  →  norm = 1.0  →  After: [1.0, 0.0]
Before: [0.0, 0.0]  →  norm = 0.0  →  After: [0.0, 0.0]  (safe_norms avoids /0)
```

**Why normalize?** Because HNSW uses L2 distance by default (`IndexHNSWFlat`), but on normalized vectors, L2 distance is monotonically related to cosine similarity:

```
L2²(a, b) = ||a||² + ||b||² - 2·a·b = 1 + 1 - 2·cos(θ) = 2(1 - cos(θ))
```

When all vectors have norm 1, minimizing L2 distance is the same as maximizing cosine similarity. This means the index finds clips whose token distributions point in the most similar direction, regardless of magnitude.

The `safe_norms` guard handles the theoretical case where a clip's coarse vector is all zeros (every projected token averaged to zero). In practice this never happens, but the guard costs nothing and prevents a NaN that would silently corrupt the entire index.

#### Building and Writing the Index

```python
index = faiss.IndexHNSWFlat(normalized_vectors.shape[1], self.hnsw_m)
index.hnsw.efConstruction = self.ef_construction
index.hnsw.efSearch = self.ef_search
index.add(normalized_vectors)
```

`IndexHNSWFlat` stores the raw vectors (no additional compression). The `Flat` part means the vectors are stored as-is, not product-quantized further. For 128 dimensions × 4 bytes × 76,000 clips = ~37MB of vector data, plus the graph structure. Easily fits in memory, no need to add another layer of approximation error on top of HNSW graph traversal.

`index.add(normalized_vectors)` is where the actual graph is built. FAISS inserts each vector one at a time, connecting it to its nearest neighbors found via the current graph (using `efConstruction` candidates). This is the most expensive operation — O(N × log(N) × efConstruction).

```python
output_path.parent.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(output_path))
```

Serializes the entire index (vectors + graph structure) to a single binary file. For our 200-vector example, this would be a file like `coarse_hnsw.faiss` of maybe 100KB. For 76,000 real clips, roughly 40–50MB.

```python
return coarse_vectors
```

Returns the *unnormalized* coarse vectors. This is important — the validation step needs the original vectors to re-normalize them (to make sure the normalization is applied consistently). The raw vectors are returned rather than the normalized ones because downstream code might want the original magnitudes.

#### `validate_index()`: Trust But Verify

```python
def validate_index(self, index_path: Path, coarse_vectors: np.ndarray) -> float:
    vectors = self._normalize_rows(np.asarray(coarse_vectors, dtype=np.float32))
```

Re-normalizes the vectors. This must match what was done during `build_faiss_index` — if the normalization logic had a bug, validation would catch the mismatch.

```python
    query_count = min(1_000, vectors.shape[0])
    k = min(10, vectors.shape[0])
```

Randomly samples up to 1,000 queries from the dataset and searches for their 10 nearest neighbors.

```python
    hnsw_index = faiss.read_index(str(index_path))
    ...
    brute_force_index = faiss.IndexFlatIP(vectors.shape[1])
    brute_force_index.add(vectors)
```

Loads the HNSW index from disk, and builds a brute-force index (`IndexFlatIP` = inner product, which on unit vectors = cosine similarity). The brute-force index always returns the *exact* nearest neighbors.

```python
    rng = np.random.default_rng(0)
    query_indices = rng.choice(vectors.shape[0], size=query_count, replace=False)
    queries = np.ascontiguousarray(vectors[query_indices])

    _, hnsw_neighbors = hnsw_index.search(queries, k)
    _, brute_force_neighbors = brute_force_index.search(queries, k)
```

For each query, both indexes return the indices of the k nearest neighbors. The brute-force result is ground truth.

```python
    recall = float(
        np.mean([
            len(set(hnsw_row.tolist()) & set(brute_row.tolist())) / k
            for hnsw_row, brute_row in zip(hnsw_neighbors, brute_force_neighbors, strict=True)
        ])
    )
```

**Recall@10** measures what fraction of the true top-10 neighbors the HNSW index found. Concrete example:

```
Query 42:
  Brute-force top-10: {3, 7, 15, 22, 31, 45, 67, 88, 91, 100}
  HNSW top-10:        {3, 7, 15, 22, 31, 45, 67, 88, 91, 99}
                                                           ^^
  Intersection: 9 out of 10 → recall = 0.9

Query 99:
  Brute-force top-10: {1, 5, 12, 20, 33, 41, 55, 60, 78, 85}
  HNSW top-10:        {1, 5, 12, 20, 33, 41, 55, 60, 78, 85}

  Intersection: 10 out of 10 → recall = 1.0

Average recall = (0.9 + 1.0) / 2 = 0.95
```

```python
    assert recall > 0.95, f"recall@10 dropped below target: {recall:.4f}"
```

If the HNSW index misses more than 5% of true neighbors on average, the pipeline fails hard. This protects against bugs in normalization, incorrect hyperparameters, or corrupt index files.

---

### 3. Modified File: `compression/pipeline.py` — Plugging the Index Into the Pipeline

#### Change 1: New Import and Checkpoint Key

```python
from index import IndexBuilder               # NEW

_INDEXED_SHARD_KEY = "indexed_shard_id"       # NEW
```

A new checkpoint key tracks whether the index has been built. The pattern follows the existing `_LAST_COMPLETED_SHARD_KEY` — store the latest shard ID that's been indexed, so the pipeline knows whether it needs to rebuild.

#### Change 2: New Config Property

```python
@property
def faiss_index_path(self) -> Path:
    return self.output_dir / "coarse_hnsw.faiss"
```

The index file lives alongside the other outputs. After this change, the output directory looks like:

```
compressed/
├── compression_model/       # Unchanged
├── shards/                  # Unchanged
├── metadata/                # Unchanged
├── clips.parquet            # Unchanged
└── coarse_hnsw.faiss        # NEW — the HNSW search index
```

#### Change 3: Index Building Step in `run_compression_pipeline()`

Before the change, the end of `run_compression_pipeline` looked like:

```python
    _consolidate_metadata(config)

    logger.info(
        "compression_pipeline_complete",
        shard_count=len(raw_batches),
        metadata_path=str(config.consolidated_metadata_path),
    )
    return config.consolidated_metadata_path
```

After:

```python
    _consolidate_metadata(config)

    # ── NEW BLOCK: Build the search index ──
    indexed_shard_id = int(_read_checkpoint(config.checkpoint_db, _INDEXED_SHARD_KEY, "-1"))
    final_shard_id = len(raw_batches) - 1
    if indexed_shard_id < final_shard_id or not config.faiss_index_path.exists():
        index_builder = IndexBuilder()
        coarse_vectors = index_builder.build_faiss_index(config.shard_dir, config.faiss_index_path)
        index_builder.validate_index(config.faiss_index_path, coarse_vectors)
        _write_checkpoint(config.checkpoint_db, _INDEXED_SHARD_KEY, str(final_shard_id))
    # ── END NEW BLOCK ──

    logger.info(
        "compression_pipeline_complete",
        shard_count=len(raw_batches),
        metadata_path=str(config.consolidated_metadata_path),
        index_path=str(config.faiss_index_path),        # NEW log field
    )
    return config.faiss_index_path                       # CHANGED return value
```

Let's trace the resumability logic with a concrete scenario:

**Scenario: Fresh run with 3 raw batches (shard IDs 0, 1, 2)**

```
Step 1: Read checkpoint "indexed_shard_id" → default "-1"
Step 2: indexed_shard_id = -1
Step 3: final_shard_id = 3 - 1 = 2
Step 4: Is -1 < 2? YES → Build the index
Step 5: IndexBuilder reads all 3 shards, concatenates coarse vectors
Step 6: Normalizes vectors, builds HNSW graph, writes coarse_hnsw.faiss
Step 7: Validates recall@10 > 0.95
Step 8: Writes checkpoint: "indexed_shard_id" = "2"
```

**Scenario: Rerun after a successful run (nothing changed)**

```
Step 1: Read checkpoint "indexed_shard_id" → "2"
Step 2: indexed_shard_id = 2
Step 3: final_shard_id = 3 - 1 = 2
Step 4: Is 2 < 2? NO.
Step 5: Does coarse_hnsw.faiss exist? YES.
Step 6: Both conditions false → SKIP index building entirely
```

**Scenario: New data added (4 batches now, was 3)**

```
Step 1: Read checkpoint "indexed_shard_id" → "2"
Step 2: indexed_shard_id = 2
Step 3: final_shard_id = 4 - 1 = 3
Step 4: Is 2 < 3? YES → Rebuild the index
Step 5: IndexBuilder reads all 4 shards (rebuilds from scratch)
Step 6: Writes checkpoint: "indexed_shard_id" = "3"
```

**Scenario: Crash during index building, then rerun**

```
First run:
  Step 1-3: indexed_shard_id = -1, final_shard_id = 2
  Step 4: -1 < 2 → YES → Start building
  Step 5: *crash during HNSW construction*
  Step 6: Checkpoint never written (stays at -1)
  Step 7: coarse_hnsw.faiss either doesn't exist or is incomplete

Rerun:
  Step 1: Read checkpoint → "-1"
  Step 4: -1 < 2 → YES → Rebuild from scratch
  (The incomplete faiss file gets overwritten cleanly)
```

**Scenario: Index file manually deleted**

```
Step 1: Read checkpoint → "2"
Step 3: final_shard_id = 2
Step 4: Is 2 < 2? NO.
Step 5: Does coarse_hnsw.faiss exist? NO → Rebuild
```

The `or not config.faiss_index_path.exists()` guard catches this edge case. Even if the checkpoint says "done," if the file is missing, rebuild it.

#### Change 4: Updated Return Value and `main()`

```python
# Before:
return config.consolidated_metadata_path

# After:
return config.faiss_index_path
```

The pipeline's "final output" shifts from "here's the metadata" to "here's the search index." The metadata is still there and still consolidated — it's just not the primary return value anymore.

```python
# Before:
metadata_path = run_compression_pipeline(config)
print({"metadata_path": str(metadata_path)})

# After:
index_path = run_compression_pipeline(config)
print({"index_path": str(index_path), "metadata_path": str(config.consolidated_metadata_path)})
```

The CLI entrypoint now prints both paths, so the caller knows where to find both the index and the metadata.

---

### 4. Modified File: `tests/compression/test_pipeline.py`

The test `test_run_compression_pipeline_reuses_existing_outputs_on_rerun` verifies the end-to-end pipeline including resumability. Here's what changed and why:

#### Assertion changes: return value

```python
# Before:
metadata_path = run_compression_pipeline(config)
assert metadata_path == config.consolidated_metadata_path

# After:
index_path = run_compression_pipeline(config)
assert index_path == config.faiss_index_path
assert config.faiss_index_path.exists()
```

The return value changed, so the assertion changes to match. The additional `.exists()` check ensures the file was actually written to disk, not just that the path object was returned.

#### Assertion changes: checkpoint rows

```python
# Before:
assert checkpoint_rows == {
    "compression:training_complete": "1",
    "compression:last_completed_shard_id": "1",
}

# After:
assert checkpoint_rows == {
    "compression:training_complete": "1",
    "compression:last_completed_shard_id": "1",
    "compression:indexed_shard_id": "1",       # NEW
}
```

The test has 2 raw batches (shard IDs 0 and 1), so `final_shard_id = 1`. After the index is built, the checkpoint should read `"1"`.

#### Resumability assertion

```python
monkeypatch.setattr("compression.pipeline.write_compressed_shard", fail_if_called)
second_index_path = run_compression_pipeline(config)
assert second_index_path == config.faiss_index_path
```

This is the idempotency test. On the second run:
- `write_compressed_shard` is monkeypatched to raise an error — if the pipeline tries to re-compress any shard, the test fails
- The index checkpoint is already at `"1"` (= `final_shard_id`), so indexing is skipped too
- The pipeline returns the same index path without doing any work

### 5. New File: `tests/index/test_builder.py`

This tests the `IndexBuilder` in isolation (no compression pipeline, no checkpoint database).

```python
def test_build_faiss_index_and_validate_recall(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    rng = np.random.default_rng(0)

    # Create 20 clusters, each with 10 tightly-grouped coarse vectors
    centers = rng.normal(size=(20, 128)).astype(np.float32)
    coarse_vectors = np.concatenate([
        centers[i] + rng.normal(scale=0.01, size=(10, 128)).astype(np.float32)
        for i in range(centers.shape[0])
    ], axis=0)  # shape: (200, 128)
```

The test creates 200 coarse vectors clustered around 20 centers. The noise scale (0.01) is tiny relative to the center distances (~1.0 in each dimension), so the 10 vectors around each center are much closer to each other than to any other center. This makes the recall test meaningful — the true nearest neighbors are the vectors from the same cluster.

```python
    first_shard = [_make_dummy_clip(vector) for vector in coarse_vectors[:100]]
    second_shard = [_make_dummy_clip(vector) for vector in coarse_vectors[100:]]
    write_compressed_shard(first_shard, shard_dir / "shard_00000000.widx")
    write_compressed_shard(second_shard, shard_dir / "shard_00000001.widx")
```

Wraps each coarse vector in a `CompressedClip` with zeroed-out centroid IDs and residuals (the index only reads coarse vectors, so the other fields don't matter). Writes them into two shard files to verify multi-shard reading works.

```python
    builder = IndexBuilder(ef_search=256)
    recovered_vectors = builder.build_faiss_index(shard_dir, index_path)
    recall = builder.validate_index(index_path, recovered_vectors)

    assert index_path.exists()
    assert recovered_vectors.shape == (200, 128)
    assert recall > 0.95
```

Builds the index, validates it, and checks that:
1. The file exists on disk
2. All 200 vectors were recovered from the shards
3. The HNSW index achieves >95% recall vs brute force

---

## End-to-End: What Happens When You Run `poetry run python scripts/compress.py config.yaml`

Here's the complete timeline with the new changes, in order:

```
1. Parse config.yaml → CompressionPipelineConfig
2. Create output directories (shards/, metadata/)
3. Initialize SQLite checkpoint database
4. Discover raw extraction batches (tokens_*.npy + metadata_*.parquet)

5. TRAINING PHASE
   ├── Check checkpoint: "training_complete" == "1"?
   ├── If yes → Load existing compressor from disk
   └── If no  → Sample tokens → Train PCA + KMeans + Quantization → Save → Checkpoint

6. COMPRESSION PHASE (per batch)
   ├── Check checkpoint: shard_id <= "last_completed_shard_id"?
   ├── If yes → Skip this batch
   └── If no  → Load tokens → Compress each clip → Write .widx shard → Checkpoint

7. METADATA CONSOLIDATION
   └── Merge all shard_*.parquet files → clips.parquet

8. INDEX BUILDING PHASE (NEW)
   ├── Check checkpoint: "indexed_shard_id" < final_shard_id?
   ├── Also check: does coarse_hnsw.faiss exist?
   ├── If both checks pass → Skip
   └── Otherwise:
       ├── Read coarse vectors from every .widx shard (memory-mapped)
       ├── L2-normalize each vector to unit length
       ├── Build HNSW graph (M=32, efConstruction=200)
       ├── Write coarse_hnsw.faiss to disk
       ├── Validate recall@10 > 0.95 against brute-force search
       └── Write checkpoint: "indexed_shard_id" = final_shard_id

9. Log completion, return path to coarse_hnsw.faiss
```

After a successful run, the output directory now looks like:

```
compressed/
├── compression_model/
│   ├── pca_components.npy
│   ├── pca_mean.npy
│   ├── centroids.npy
│   ├── quantile_thresholds.npy
│   ├── quantization_levels.npy
│   ├── explained_variance_ratio.npy
│   └── input_dim.npy
├── shards/
│   ├── shard_00000000.widx
│   ├── shard_00000001.widx
│   └── ...
├── metadata/
│   ├── shard_00000000.parquet
│   ├── shard_00000001.parquet
│   └── ...
├── clips.parquet
└── coarse_hnsw.faiss              ← NEW
```

And the checkpoint database contains:

```
┌────────────────────────────────────────┬──────────────────┐
│ checkpoint_key                         │ checkpoint_value │
├────────────────────────────────────────┼──────────────────┤
│ compression:training_complete          │ 1                │
│ compression:last_completed_shard_id    │ 41               │
│ compression:indexed_shard_id           │ 41               │  ← NEW
└────────────────────────────────────────┴──────────────────┘
```
