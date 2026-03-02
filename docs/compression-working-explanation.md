# How WorldIndex Compression Works

## Start Here: Where Extraction Left Off

The extraction pipeline gave us token embeddings — for each video clip, V-JEPA 2 produced a matrix of numbers encoding *what's happening where* in the scene. A single clip turns into something like `(8192, 1280)`: 8,192 tokens, each a 1,280-dimensional vector. These sit on disk as numpy files alongside parquet metadata:

```
artifacts/extraction_output/
  tokens_00000000_00000099.npy        # Shape: (100, 8192, 1280), 100 clips stacked together
  metadata_00000000_00000099.parquet   # 100 rows of clip metadata
  tokens_00000100_00000199.npy
  metadata_00000100_00000199.parquet
  ...
```

These files are enormous. A single clip's tokens are `8192 tokens * 1280 dims * 4 bytes/float32 = 41,943,040 bytes ≈ 40 MB`. A batch of 100 clips is 4 GB. If you've extracted 76,000 clips, that's about 3 TB of raw token data just sitting on disk.

You can't build a search index over 3 TB of float32 arrays. Loading them all into memory is impossible on most machines. Scanning them linearly is glacially slow. Even with FAISS or another vector search library, the sheer volume of data means index construction takes forever and the index itself won't fit in memory.

The compression pipeline solves this. It takes those enormous raw token files and compresses them — lossy, but with provably high fidelity (>0.97 cosine similarity with the originals) — into a custom binary format that's roughly **90x smaller**. That 3 TB becomes ~33 GB. Now you can load the whole thing, build an index, and do real-time search. Index is the future step.

If extraction was "run the model on every clip and save the results", compression is "shrink those results down to a size that's actually usable for search, without destroying the information". It is lossy but works alright.

---

## High Level Architecture

Unlike the extraction pipeline (which used three OS processes communicating over queues), the compression pipeline is a single-process, sequential operation. There's no GPU involved, no streaming from the network, no shared memory tricks. It's pure CPU math — PCA (Principal Component Analysis, more about it later), K-means, quantization — applied batch by batch.

```
artifacts/extraction_output/                              artifacts/compressed_output/
┌─────────────────────────────────┐                       ┌──────────────────────────────────┐
│ tokens_00000000_00000099.npy    │──┐                    │ compression_model/               │
│ metadata_00000000_00000099.pq   │  │                    │   pca_components.npy             │
│ tokens_00000100_00000199.npy    │  │   ┌────────────┐   │   pca_mean.npy                   │
│ metadata_00000100_00000199.pq   │──┼──>│ COMPRESSION│──>│   centroids.npy                  │
│ tokens_00000200_00000299.npy    │  │   │  PIPELINE  │   │   quantile_thresholds.npy        │
│ metadata_00000200_00000299.pq   │  │   └────────────┘   │   quantization_levels.npy        │
│ ...                             │──┘                    │ shards/                          │
└─────────────────────────────────┘                       │   shard_00000000.widx            │
                                                          │   shard_00000001.widx            │
                                                          │   shard_00000002.widx            │
                                                          │ metadata/                        │
                                                          │   shard_00000000.parquet         │
                                                          │   shard_00000001.parquet         │
                                                          │ clips.parquet  (consolidated)    │
                                                          └──────────────────────────────────┘
```

The pipeline has two phases:

1. **Train** — Sample a subset of tokens from across all batches, learn a compression model (PCA projection + K-means centroids + quantization thresholds). This happens once.
2. **Compress** — Apply that learned model to every clip, writing compressed binary shards and enriched metadata.

Let's walk through each piece.

---

## Phase 1: Training the Compressor

### Why train at all?

The compression isn't a generic algorithm like gzip. It learns the statistical structure of *your specific token embeddings*. PCA learns which directions in 1,280-dimensional space carry the most variance (information). K-means learns what the typical token vectors look like so it can represent each one as "closest to cluster #4,271 plus a small residual". Without training on the actual data, these techniques would either preserve the wrong information or not compress well.

### Step 1: Discover raw batches

Before anything happens, the pipeline needs to find the extraction output. It scans `raw_dir` for files matching the pattern `tokens_*.npy` (remember, I mentioned in extraction that doing `glob` will become problematic if we have 1 clip per file pair):

```python
token_paths = sorted(path for path in raw_dir.glob("tokens_*.npy") if path.is_file())
```

For each token file, it expects a matching metadata file. If `tokens_00000000_00000099.npy` exists, there must be a `metadata_00000000_00000099.parquet` next to it. If the metadata file is missing, the pipeline raises an error — the two files are always produced together by extraction, so a missing one means something went wrong upstream.

Each batch is described by a `RawBatch` object:

```python
RawBatch(
    token_path=Path("artifacts/raw/tokens_00000000_00000099.npy"),
    metadata_path=Path("artifacts/raw/metadata_00000000_00000099.parquet"),
    clip_count=100,           # 100 clips in this file
    token_count=819200,       # 100 clips * 8192 tokens/clip = 819,200 individual tokens
    embedding_dim=1280,       # each token is a 1280-dim vector
)
```

The `token_count` here is the *total* number of individual token vectors in the batch (clips * tokens_per_clip), not the number of clips. This matters for sampling, which works at the individual token level.

Note that the token file is opened with `mmap_mode="r"` — memory-mapped, read-only. This means numpy doesn't load the entire 4 GB file into RAM. Instead, it creates a virtual mapping where the OS loads pages on demand as you access them. When we only need a random sample of tokens from this file, we read maybe 5% of the pages instead of all of them. This is how the pipeline handles datasets larger than available RAM.

### Step 2: Sample training tokens

You don't need to train on every single token. 76,000 clips * 8,192 tokens = 622 million tokens. Training PCA and K-means on 622 million 1,280-dimensional vectors would take days and hundreds of GB of RAM. Instead, we sample a manageable subset — by default, 500,000 tokens.

But we can't just grab the first 500K tokens from the first batch. That would give us tokens only from the earliest clips (probably all from one dataset, one robot, one scene). The compressor would learn a biased view of the data.

Instead, the sampling is **stratified across batches**. Each batch contributes tokens proportional to its share of the total:

```
Batch 0:  819,200 tokens out of 2,457,600 total → contributes 33.3% of samples → 166,667 tokens
Batch 1:  819,200 tokens out of 2,457,600 total → contributes 33.3% of samples → 166,667 tokens
Batch 2:  819,200 tokens out of 2,457,600 total → contributes 33.3% of samples → 166,666 tokens
                                                                           Total: 500,000 tokens
```

The actual allocation uses largest-remainder apportionment (the same algorithm used to allocate parliamentary seats to parties based on vote share). First, compute each batch's exact fractional share. Floor each to get a base count. Then distribute the leftover seats one at a time to the batches with the largest fractional remainders. This ensures the total is exactly `sample_size` without any batch getting zero samples unfairly.

Within each batch, the sampled token indices are chosen uniformly at random:

```python
sample_indices = rng.choice(batch.token_count, size=sample_count, replace=False)
```

The token file is memory-mapped, so `flattened[sample_indices]` only reads the specific rows we need from disk. For a batch with 819,200 tokens where we sample 166,667 of them, this reads ~20% of the file — scattered, but the OS readahead cache handles this reasonably well.

The result is a 2D array of shape `(500000, 1280)` — 500K randomly sampled token vectors from across all batches, covering the full diversity of the dataset. These are the training data for the compressor.

### Step 3: Train PCA (Stage 1 of compression)

**What PCA does:** PCA (Principal Component Analysis) finds the directions in the 1,280-dimensional space where the data varies the most. If 95% of the variance lives in just 128 directions (out of 1,280), we can project every token from 1,280 dimensions down to 128 dimensions and lose almost nothing. The remaining 1,152 dimensions were mostly noise or redundancy.

Think of it this way. Imagine you have a bunch of 3D points, but they all lie roughly on a flat plane — one of those three dimensions is nearly constant. PCA would discover that plane and let you describe each point with just 2 numbers (its position on the plane) instead of 3, dropping the "height off the plane" that's practically zero for all points anyway.

Same idea here, but in 1,280 dimensions instead of 3, projecting down to 128.

Here's how it works in the code:

```python
pca = PCA(n_components=128, svd_solver="randomized", random_state=0)
pca.fit(training_tokens)  # training_tokens shape: (500000, 1280)
```

The `svd_solver="randomized"` is important. Computing exact PCA on a 500,000 x 1,280 matrix requires a full Singular Value Decomposition, which is O(n * d^2) — expensive. Randomized PCA uses random projections to approximate the top-k singular vectors much faster, with negligible accuracy loss. For our use case (128 components out of 1,280), the approximation is essentially exact.

After fitting, `pca.explained_variance_ratio_` tells us what fraction of the total variance each component captures. For V-JEPA 2 embeddings, the first 128 components typically capture >95% of the total variance. That means our 128-dimensional projection preserves 95%+ of the information.

**The automatic fallback:** There's a safety check. If 128 dimensions capture less than 95% of variance, and the data has at least 256 dimensions available, the compressor automatically bumps `pca_dim` to 256:

```python
if self.pca_dim == 128 and explained_variance < 0.95 and min(n_samples, n_features) >= 256:
    pca = self._fit_pca(training_tokens, 256)
```

When would this trigger? If you used a different model that produces embeddings where variance is more evenly spread across dimensions. V-JEPA 2 embeddings are highly structured (they come from a learned model, so they have strong correlations), so 128 is almost always enough. But this fallback prevents silently destroying information if that assumption is wrong.

After PCA training, the compressor stores two arrays:

- `pca_components`: shape `(128, 1280)` — the 128 directions to project onto. Each row is a unit vector in 1,280-dimensional space.
- `pca_mean`: shape `(1280,)` — the mean of the training data. PCA requires centering (subtracting the mean) before projection.

To project a token: `projected = (token - pca_mean) @ pca_components.T`

That turns a 1,280-dim vector into a 128-dim vector. Let's visualize the size change:

```
Before PCA:  1 token = 1,280 floats * 4 bytes = 5,120 bytes
After PCA:   1 token = 128 floats * 4 bytes   = 512 bytes     (10x smaller)
```

Good start, but we can do much better.

### Step 4: Train K-means (Stage 2 of compression, part A)

After PCA, every token is a 128-dimensional float32 vector. We now want to compress these further. The key insight: in 128-dimensional space, tokens tend to cluster. Tokens from similar visual regions, similar objects, or similar motions end up near each other. If we can identify these clusters, we can represent each token as "which cluster it belongs to" rather than storing all 128 numbers.

This is **vector quantization** — replace each vector with the ID of its nearest codebook entry (centroid).

```python
kmeans = MiniBatchKMeans(
    n_clusters=32768,     # 2^15 = 32,768 centroids
    batch_size=10000,     # process 10K tokens at a time (K-means runs iteratively)
    n_init="auto",
    random_state=0,
)
kmeans.fit(projected)  # projected shape: (500000, 128)
```

**Why 32,768 centroids?** It's a balance between precision and storage. With 32,768 centroids, each token's cluster assignment fits in a uint16 (2 bytes, range 0–65,535). With more centroids, you get finer-grained representation (less quantization error), but each ID takes more bytes. 32,768 is the sweet spot — enough clusters to capture the diversity of token embeddings, small enough to store as 2 bytes.

**Why MiniBatchKMeans instead of regular KMeans?** Regular K-means loads all 500,000 training vectors into memory and iterates over all of them in each step. MiniBatchKMeans takes random mini-batches of 10,000 tokens at a time, which converges faster and uses less memory. The quality difference is negligible for our use case.

After fitting, `kmeans.cluster_centers_` gives us the 32,768 centroid vectors, each 128-dimensional. These are the "codebook" — the set of prototype vectors that every token will be approximated by.

To quantize a token: find the nearest centroid (using L2 distance) and record its index.

```
Original token:        [0.23, -0.41, 0.87, 0.12, ...]    128 floats = 512 bytes
Quantized to:          centroid #4271                      1 uint16  = 2 bytes
Centroid #4271 value:  [0.25, -0.39, 0.85, 0.14, ...]    (close but not exact)
```

But wait — going from 512 bytes to 2 bytes is a 256x compression, which sounds amazing but loses too much precision. The centroid is *close* to the original token but not *equal*. For WorldIndex's spatial search, we need high fidelity — each token encodes a specific spatial patch of a robot scene, and small differences matter.

That's where residual quantization comes in.

### Step 5: Compute residuals and quantization thresholds (Stage 2, part B)

The **residual** is the difference between the original token and its nearest centroid:

```
residual = token - centroid[assigned_cluster]
         = [0.23, -0.41, 0.87, 0.12, ...] - [0.25, -0.39, 0.85, 0.14, ...]
         = [-0.02, -0.02, 0.02, -0.02, ...]
```

Residuals are small — by definition, K-means minimizes the distance between points and their centroids, so the residuals cluster around zero. This matters because small, narrowly distributed values are easy to quantize coarsely without losing much.

The compressor quantizes each residual dimension to just **2 bits** (from 4 bytes = 32 bits!) — four possible levels (0, 1, 2, 3). To decide where to place the boundaries between levels, it uses percentiles of the training residuals:

```python
quantile_thresholds = np.percentile(residuals, [25, 50, 75], axis=0)
```

This computes, for each of the 128 dimensions, the 25th, 50th, and 75th percentile of residual values across all 500K training tokens. These three thresholds split each dimension's residual distribution into four quartile buckets.

Let's walk through a concrete example for one dimension (say dimension 0):

```
Training residuals for dim 0: [-0.05, -0.03, -0.01, 0.00, 0.01, 0.02, 0.04, 0.06, ...]

25th percentile (q1): -0.02
50th percentile (q2):  0.01    (the median)
75th percentile (q3):  0.03

Quantization rules:
  residual <= -0.02  →  level 0
  -0.02 < residual <=  0.01  →  level 1
  0.01 < residual <=  0.03  →  level 2
  residual > 0.03   →  level 3
```

The actual quantization code does this with broadcasting:

```python
quantized = (residuals > thresholds[0]).astype(np.uint8)   # 0 if below 25th percentile, 1 if above
quantized += (residuals > thresholds[1]).astype(np.uint8)  # +1 if above median
quantized += (residuals > thresholds[2]).astype(np.uint8)  # +1 if above 75th percentile
```

Each comparison adds 1 if the residual exceeds that threshold. If a value exceeds all three thresholds, it gets `1 + 1 + 1 = 3`. If it exceeds none, it gets `0`. This produces values in {0, 1, 2, 3} — exactly 2 bits' worth.

### Step 6: Compute quantization levels (for decompression later)

When we decompress, we need to go from level {0, 1, 2, 3} back to a float. The compressor pre-computes a representative value for each level in each dimension — the **median** of all training residuals that fell into that bucket:

```python
for dim in range(128):
    for level in range(4):
        bucket_values = residuals[:, dim][quantized[:, dim] == level]
        levels[level, dim] = np.median(bucket_values)
```

Continuing the example for dimension 0:

```
Level 0 (residuals <= -0.02): values = [-0.05, -0.03, -0.02], median = -0.03
Level 1 (-0.02 < r <= 0.01): values = [-0.01, 0.00, 0.01],   median =  0.00
Level 2 (0.01 < r <= 0.03):  values = [0.02, 0.03],           median =  0.025
Level 3 (r > 0.03):          values = [0.04, 0.06],           median =  0.05
```

So the `quantization_levels` array (shape `(4, 128)`) is a lookup table: given a quantized level (0–3) and a dimension (0–127), it returns the best float estimate for the original residual. Using the median (rather than the mean) makes this robust to outliers.

If a bucket is empty (no training tokens fell into that range for some dimension), a fallback rule uses midpoints between thresholds — this is rare but handled correctly.

### What the trained compressor stores

After training, the compressor has these learned parameters:

| Array                     | Shape          | Purpose                                            |
|---------------------------|----------------|-----------------------------------------------------|
| `pca_components`          | (128, 1280)    | Projection matrix (directions of max variance)      |
| `pca_mean`                | (1280,)        | Mean of training data (for centering)                |
| `centroids`               | (32768, 128)   | K-means cluster centers in PCA space                 |
| `quantile_thresholds`     | (3, 128)       | 25th/50th/75th percentile per dimension              |
| `quantization_levels`     | (4, 128)       | Representative value per quantization level/dim      |
| `explained_variance_ratio`| (128,)         | Variance captured by each PCA component              |
| `input_dim`               | scalar         | Original embedding dimension (1280)                  |

These are saved as individual `.npy` files in the `compression_model/` directory. They're loaded once and reused for every clip.

---

## Phase 2: Compressing Clips

Now the compressor is trained. For every raw batch, the pipeline:

1. Loads the token batch (memory-mapped)
2. Compresses each clip individually
3. Writes all compressed clips into a binary shard file
4. Writes enriched metadata (with shard ID and offset) to parquet

Let's trace the compression of a single clip through the code.

### Step 1: PCA projection

```python
projected = self._project(raw_tokens)
```

The raw tokens for one clip are shape `(8192, 1280)`. The projection is a matrix multiply:

```python
centered = raw_tokens - self.pca_mean             # (8192, 1280) - (1280,) = (8192, 1280)
projected = centered @ self.pca_components.T       # (8192, 1280) @ (1280, 128) = (8192, 128)
```

Each of the 8,192 tokens is independently projected from 1,280 dims to 128 dims. The matrix multiply is highly optimized by numpy/BLAS — it's essentially free compared to I/O.

```
Before: (8192, 1280) = 8192 * 1280 * 4 bytes = 41,943,040 bytes ≈ 40 MB
After:  (8192, 128)  = 8192 * 128 * 4 bytes  = 4,194,304 bytes  ≈ 4 MB
```

### Step 2: Find nearest centroids

```python
centroid_index = self._get_centroid_index()
_, nearest_centroid_ids = centroid_index.search(projected, 1)
```

For each of the 8,192 projected tokens, we need to find the nearest centroid (out of 32,768). This is a nearest-neighbor search in 128-dimensional space.

Naively, you'd compute the L2 distance from each token to all 32,768 centroids — that's `8192 * 32768 = 268 million` distance computations. Each distance computation involves 128 multiplications and additions. This is expensive.

FAISS (Facebook AI Similarity Search) handles this efficiently. The compressor builds a `faiss.IndexFlatL2` — a brute-force exact nearest-neighbor index. Despite being "brute force", FAISS uses SIMD instructions, cache-optimal memory layouts, and parallelism to make this blazingly fast on CPU. For 8,192 queries against 32,768 centroids in 128 dimensions, it takes milliseconds.

The index is created once (lazily, on first use) and reused:

```python
def _get_centroid_index(self):
    if self._centroid_index is None:
        centroid_index = faiss.IndexFlatL2(self.pca_dim)    # 128-dimensional index
        centroid_index.add(self.centroids)                   # add all 32,768 centroids
        self._centroid_index = centroid_index
    return self._centroid_index
```

The result is an array of centroid IDs, one per token:

```
Token 0: nearest centroid = 4271
Token 1: nearest centroid = 18023
Token 2: nearest centroid = 4271    (same cluster as token 0)
Token 3: nearest centroid = 902
...
Token 8191: nearest centroid = 7744

centroid_ids: shape (8192,), dtype uint16
```

Each ID is stored as uint16 (2 bytes). This is why `n_centroids` must be <= 65,535 — the maximum value of uint16.

### Step 3: Compute and quantize residuals

```python
centroids_for_tokens = self.centroids[centroid_ids.astype(np.intp)]  # (8192, 128)
residuals = projected - centroids_for_tokens                          # (8192, 128)
quantized_residuals = self._pack_2bit(self._quantize_residuals(residuals))
```

First, `_quantize_residuals` turns each float residual into a 2-bit level (0, 1, 2, or 3) using the learned thresholds. The result is shape `(8192, 128)` of uint8 values, each in {0, 1, 2, 3}.

Then, `_pack_2bit` packs four 2-bit values into a single byte. Since each value only uses 2 bits out of a byte's 8, we can fit 4 values per byte:

```
Four values:  [2, 0, 3, 1]
Binary:        10  00  11  01

Packed into one byte:
  value[0] occupies bits 0-1:  10       =  2
  value[1] occupies bits 2-3:  00 << 2  =  0
  value[2] occupies bits 4-5:  11 << 4  = 48
  value[3] occupies bits 6-7:  01 << 6  = 64

  Byte = 2 | 0 | 48 | 64 = 114 = 0b01110010
```

The code does this with vectorized bit shifting:

```python
grouped = quantized.reshape(n_tokens, -1, 4)        # group every 4 consecutive values
packed = (
    grouped[:, :, 0]            # bits 0-1
    | (grouped[:, :, 1] << 2)   # bits 2-3
    | (grouped[:, :, 2] << 4)   # bits 4-5
    | (grouped[:, :, 3] << 6)   # bits 6-7
)
```

After packing, 128 dimensions become 128/4 = 32 bytes per token:

```
Before packing: (8192, 128) uint8 = 8192 * 128 = 1,048,576 bytes = 1 MB
After packing:  (8192, 32) uint8  = 8192 * 32  = 262,144 bytes   = 256 KB
```

### Step 4: Compute coarse vector

```python
coarse_vector = projected.mean(axis=0, dtype=np.float32)  # shape (128,)
```

The coarse vector is simply the mean of all PCA-projected tokens in the clip. It's a single 128-dimensional vector that summarizes the entire clip. Think of it as a "thumbnail" of the clip's content in embedding space.

Why store this? For search, the first pass is a fast scan over coarse vectors to find candidate clips (like checking book titles before reading the full text). Only the promising candidates get their individual tokens decompressed for the expensive fine-grained comparison. Coarse vectors are small enough to load entirely into memory for the full dataset — 76,000 clips * 128 floats * 4 bytes = 39 MB, trivially manageable.

### Summary: What one compressed clip looks like

All three pieces are bundled into a `CompressedClip`:

```python
CompressedClip(
    centroid_ids=np.array([4271, 18023, 4271, 902, ...]),       # (8192,) uint16
    quantized_residuals=np.array([[114, 23, ...], ...]),         # (8192, 32) uint8
    coarse_vector=np.array([0.15, -0.22, 0.07, ...]),           # (128,) float32
)
```

Let's total up the bytes:

```
Component               Per-token         Total (8192 tokens)    Bytes
─────────────────────   ───────────       ───────────────────    ──────
centroid_ids            2 bytes           8192 * 2               16,384
quantized_residuals     32 bytes          8192 * 32              262,144
coarse_vector           (per-clip)        128 * 4                512
                                                          Total: 279,040 bytes ≈ 272 KB

Compare to original:
raw tokens              5,120 bytes       8192 * 5120            41,943,040 bytes ≈ 40 MB

Compression ratio: 40 MB / 272 KB ≈ 150x
```

The actual ratio depends on the token count and embedding dimension. With the coarse vector included, it's roughly 150x for V-JEPA 2 ViT-Huge output. Even with smaller models (fewer dimensions), the compression is still dramatic because the 2-bit quantization dominates the savings.

---

## Decompression: Getting Tokens Back

When search needs the fine-grained tokens for a clip (e.g., to compare spatial patches against a query), it decompresses:

```python
def decompress_clip(self, compressed: CompressedClip) -> np.ndarray:
    centroids = self.centroids[compressed.centroid_ids.astype(np.intp)]     # (8192, 128)
    quantized = self._unpack_2bit(compressed.quantized_residuals)           # (8192, 128) uint8
    dequantized = self.quantization_levels[quantized, np.arange(128)]      # (8192, 128) float32
    return centroids + dequantized                                          # (8192, 128) float32
```

Step by step:

1. **Look up centroids**: For each token, use its `centroid_id` to fetch the 128-dim centroid vector from the codebook. Shape goes from `(8192,)` indices to `(8192, 128)` vectors.

2. **Unpack 2-bit residuals**: Reverse the bit-packing — extract four 2-bit values from each byte:

   ```python
   unpacked[:, 0::4] = packed & 0b00000011        # bits 0-1
   unpacked[:, 1::4] = (packed >> 2) & 0b00000011  # bits 2-3
   unpacked[:, 2::4] = (packed >> 4) & 0b00000011  # bits 4-5
   unpacked[:, 3::4] = (packed >> 6) & 0b00000011  # bits 6-7
   ```

3. **Dequantize**: Convert levels {0, 1, 2, 3} back to floats using the `quantization_levels` lookup table. The expression `self.quantization_levels[quantized, np.arange(128)]` uses fancy indexing — for each token and each dimension, it looks up the float value corresponding to that token's quantization level in that dimension.

4. **Reconstruct**: Add centroid + dequantized residual = approximation of the original projected token.

The result is shape `(8192, 128)` — the tokens in PCA space, not the original 1,280-dimensional space. We don't invert the PCA projection because search operates in PCA space anyway (it's lower-dimensional, so distance computations are faster).

Note that this is **lossy** — the reconstructed tokens aren't identical to the originals. But the quality is high: >0.97 average cosine similarity in tests. For spatial search ("find clips where a robot arm is in this position"), this level of fidelity is more than sufficient — the spatial structure is preserved even if individual values drift by a few percent.

---

## The Shard Format: Custom Binary Files (.widx)

Compressed clips are stored in custom binary files called **shards**. Each shard corresponds to one extraction batch (one `tokens_*.npy` file), so if extraction produced 760 batch files, compression produces 760 shard files.

### Why a custom binary format?

Why not just use numpy `.npy` files or parquet or HDF5 or something standard?

1. **Random access.** The search index needs to fetch individual clips by index during queries. `.npy` files store dense arrays — to read clip #50 out of 100, you either load all 100 or do manual byte offset arithmetic. Parquet is columnar and doesn't support this kind of structured binary data well. Our shard format has fixed-size records, so reading clip #50 is a single seek to `header_size + 50 * record_size` — O(1).

2. **Heterogeneous data.** Each clip has three fields of different types (uint16 centroid IDs, uint8 packed residuals, float32 coarse vector). Standard formats would require either separate files per field (more file management) or awkward workarounds. Our format keeps everything contiguous per clip.

3. **Memory-mapped reads.** The format is designed so numpy can memory-map the entire record section as a structured array. This means the OS handles paging — only the clips you actually access get loaded into RAM.

### File layout

```
Byte offset   Content                                        Size
───────────   ────────────────────────────────────────────    ────
0             Magic bytes: "WIDXSHD1"                         8 bytes
8             clip_count (uint32)                              4 bytes
12            token_count (uint32)                             4 bytes
16            pca_dim (uint32)                                 4 bytes
20            residual_bytes_per_token (uint32)                4 bytes
24            coarse_dim (uint32)                              4 bytes
28            record_size (uint32)                             4 bytes
              ─────────── header total ───────────            32 bytes

32            Record 0: centroid_ids                          token_count * 2
              Record 0: quantized_residuals                  token_count * residual_bytes
              Record 0: coarse_vector                        coarse_dim * 4

32 + RS       Record 1: centroid_ids                          ...
              Record 1: quantized_residuals                  ...
              Record 1: coarse_vector                        ...

32 + 2*RS     Record 2: ...
              ...

              (RS = record_size)
```

The magic bytes `"WIDXSHD1"` are a sanity check — if you accidentally try to read a JPEG or a corrupted file as a shard, the magic won't match and you'll get a clear error instead of garbage data. The `1` in `WIDXSHD1` is a version number; if the format changes in the future, it can be bumped to `WIDXSHD2`.

All integer fields are little-endian (`<` in struct format), which is the native byte order on x86 and ARM (the platforms we care about). No byte-swapping needed.

### Concrete example

Say we have 100 clips, each with 8,192 tokens, PCA dim 128, and coarse dim 128:

```
token_count = 8192
residual_bytes_per_token = 128 / 4 = 32   (128 PCA dims, 4 values packed per byte)
coarse_dim = 128

record_size = 8192 * 2          (centroid_ids: uint16)
            + 8192 * 32         (quantized_residuals: packed uint8)
            + 128 * 4           (coarse_vector: float32)
            = 16,384 + 262,144 + 512
            = 279,040 bytes per record

Total file size = 32 (header) + 100 * 279,040 = 27,904,032 bytes ≈ 26.6 MB
```

Compare to the raw token batch: 100 clips * 40 MB/clip = 4 GB. The shard is 26.6 MB. That's the compression in action.

### Writing shards atomically

Shards are written using the same write-then-rename pattern as extraction:

```python
temp_output_path = output_path.with_suffix(".widx.tmp")

with temp_output_path.open("wb") as shard_file:
    shard_file.write(header_bytes)
    for clip in clips:
        shard_file.write(clip.centroid_ids.tobytes())
        shard_file.write(clip.quantized_residuals.tobytes())
        shard_file.write(clip.coarse_vector.tobytes())

temp_output_path.replace(output_path)  # atomic rename
```

If the process crashes mid-write, you get a `.widx.tmp` file (incomplete) but no `.widx` file. The pipeline never sees a half-written shard. On retry, it re-compresses the batch and writes a fresh temp file.

Before writing, the code validates that all clips in the shard have the same shape — same token count, same packed residual width, same coarse vector dimension. A shard is homogeneous; you can't mix clips with different token counts in the same shard. This is what makes fixed-size records (and therefore random access) possible.

### Reading clips from shards

Reading a single clip by index uses memory mapping:

```python
records = np.memmap(
    shard_path,
    mode="r",
    dtype=header.record_dtype(),   # structured numpy dtype matching the record layout
    offset=32,                     # skip the 32-byte header
    shape=(header.clip_count,),    # number of records
)
record = records[clip_index]       # O(1) access — just a pointer offset
```

The `record_dtype()` method constructs a numpy structured dtype that matches the binary layout:

```python
np.dtype([
    ("centroid_ids", np.uint16, (8192,)),
    ("quantized_residuals", np.uint8, (8192, 32)),
    ("coarse_vector", np.float32, (128,)),
])
```

This tells numpy: "each record starts with 8,192 uint16 values, followed by 8,192 * 32 uint8 values, followed by 128 float32 values". When you index `records[50]`, numpy calculates `32 + 50 * 279040` and reads exactly those bytes from disk. Nothing else gets loaded.

For batch reading of coarse vectors (used during the first search pass), `read_coarse_vectors_from_shard` memory-maps the whole shard but only accesses the `coarse_vector` field:

```python
records = np.memmap(shard_path, ...)
return np.asarray(records["coarse_vector"], dtype=np.float32)
```

Numpy is smart enough to stride through the records, reading only the coarse_vector portion of each one, skipping the centroid IDs and residuals. For 100 clips, this reads 100 * 512 bytes = 50 KB out of the 26.6 MB shard file.

---

## The Pipeline Orchestrator: `run_compression_pipeline`

Let's trace a full run from the command line to the final output.

### Entry point

```bash
poetry run python scripts/compress.py config/compression.yaml
```

The script calls `main()`, which parses the YAML config and delegates:

```python
config = CompressionPipelineConfig.from_yaml(Path(argv[0]))
metadata_path = run_compression_pipeline(config)
```

The YAML config specifies where to find raw extraction output, where to write compressed output, and compression parameters:

```yaml
raw_dir: artifacts/extraction_output
output_dir: artifacts/compressed_output
sample_size: 500000
pca_dim: 128
n_centroids: 32768
n_bits: 2
random_seed: 0
```

### Step 1: Create output directories

```python
config.output_dir.mkdir(parents=True, exist_ok=True)
config.shard_dir.mkdir(parents=True, exist_ok=True)
config.shard_metadata_dir.mkdir(parents=True, exist_ok=True)
```

This creates:
```
artifacts/compressed_output/
├── shards/
└── metadata/
```

The `compression_model/` directory is created later by `compressor.save()`.

### Step 2: Train or load the compressor

```python
if config.compressor_dir.exists():
    compressor = TokenCompressor.load(config.compressor_dir)
else:
    sampled_tokens = sample_training_tokens(raw_batches, config.sample_size, config.random_seed)
    compressor = TokenCompressor(pca_dim=config.pca_dim, n_centroids=config.n_centroids, n_bits=config.n_bits)
    compressor.train(sampled_tokens)
    compressor.save(config.compressor_dir)
```

If the `compression_model/` directory already exists (from a previous run), the compressor is loaded from disk — no retraining. This makes the pipeline resumable. If you ran compression yesterday and it crashed halfway through the shards, re-running today skips the training step (which might take 10+ minutes) and jumps straight to shard writing.

This also means you can tune compression (change `n_centroids`, `pca_dim`) by deleting the `compression_model/` directory and re-running. The pipeline will retrain from scratch with the new parameters.

### Step 3: Compress each batch into a shard

```python
for shard_id, raw_batch in enumerate(raw_batches):
    shard_path = config.shard_dir / f"shard_{shard_id:08d}.widx"
    shard_metadata_path = config.shard_metadata_dir / f"shard_{shard_id:08d}.parquet"

    token_batch = np.load(raw_batch.token_path, mmap_mode="r")
    metadata_frame = pl.read_parquet(raw_batch.metadata_path)

    compressed_clips = [
        compressor.compress_clip(np.asarray(token_batch[clip_index], dtype=np.float32))
        for clip_index in range(raw_batch.clip_count)
    ]
    write_compressed_shard(compressed_clips, shard_path)
```

Each raw batch becomes one shard. The mapping is 1:1 — `tokens_00000000_00000099.npy` (batch 0) becomes `shard_00000000.widx` (shard 0). This preserves the natural grouping from extraction.

The token batch is loaded with `mmap_mode="r"` again. When we access `token_batch[clip_index]`, numpy reads just that clip's slice from disk (~40 MB). The loop processes clips sequentially, so memory usage is bounded to ~40 MB of raw tokens + ~272 KB of compressed output per clip, regardless of how many clips are in the batch.

### Step 4: Write enriched metadata

```python
shard_offsets = pl.Series("shard_offset", np.arange(raw_batch.clip_count, dtype=np.int64))
enriched_metadata = metadata_frame.with_columns(
    pl.lit(shard_id, dtype=pl.Int64).alias("shard_id"),
    shard_offsets,
)
_write_parquet_atomically(enriched_metadata, shard_metadata_path)
```

The original metadata from extraction (clip_index, episode_id, dataset_name, robot_type, timestamps, etc.) is preserved and extended with two new columns:

- `shard_id`: which shard file contains this clip's compressed data
- `shard_offset`: the clip's index within that shard (0, 1, 2, ..., clip_count-1)

These two values are everything you need to load a clip's compressed tokens: `read_clip_from_shard(shard_dir / f"shard_{shard_id:08d}.widx", shard_offset)`.

The metadata is written atomically (write to `.tmp`, then rename) just like shards and extraction output.

### Step 5: Consolidate metadata

```python
def _consolidate_metadata(config):
    metadata_paths = sorted(config.shard_metadata_dir.glob("shard_*.parquet"))
    consolidated = pl.concat([pl.read_parquet(path) for path in metadata_paths], how="vertical")
    _write_parquet_atomically(consolidated, config.consolidated_metadata_path)
```

After all shards are written, the per-shard metadata files are concatenated into a single `clips.parquet` file. This is the master catalog — one row per clip across all datasets, with every field from extraction plus the new `shard_id` and `shard_offset` columns:

```
┌────────────┬───────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ clip_index │ episode_id            │ dataset_name │ robot_type   │ shard_id     │ shard_offset │
│ i64        │ str                   │ str          │ str          │ i64          │ i64          │
╞════════════╪═══════════════════════╪══════════════╪══════════════╪══════════════╪══════════════╡
│ 0          │ lerobot/droid_0       │ droid        │ franka       │ 0            │ 0            │
│ 1          │ lerobot/droid_1       │ droid        │ franka       │ 0            │ 1            │
│ ...        │ ...                   │ ...          │ ...          │ ...          │ ...          │
│ 99         │ lerobot/droid_99      │ droid        │ franka       │ 0            │ 99           │
│ 100        │ lerobot/bridge_0      │ bridge       │ widowx       │ 1            │ 0            │
│ 101        │ lerobot/bridge_1      │ bridge       │ widowx       │ 1            │ 1            │
│ ...        │ ...                   │ ...          │ ...          │ ...          │ ...          │
└────────────┴───────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

This file is what the search index will use as its source of truth. "Give me all clips from the DROID dataset" is a column filter. "Give me the compressed tokens for clip #101" is a lookup of `shard_id=1, shard_offset=1`, followed by `read_clip_from_shard`.

---

## End-to-End Example: From 3 Clips to Compressed Shards

Let's trace a tiny concrete run to make everything tangible. We'll use the numbers from the pipeline test.

### Setup: Two raw batches

```
raw/
  tokens_00000000_00000001.npy     Shape: (2, 12, 32)    → 2 clips, 12 tokens/clip, 32-dim embeddings
  metadata_00000000_00000001.pq    2 rows
  tokens_00000002_00000002.npy     Shape: (1, 12, 32)    → 1 clip, 12 tokens/clip, 32-dim embeddings
  metadata_00000002_00000002.pq    1 row
```

Config: `pca_dim=8, n_centroids=4, sample_size=24`

### 1. Discover batches

```
Batch 0: token_path=tokens_00000000_00000001.npy, clip_count=2, token_count=24, embedding_dim=32
Batch 1: token_path=tokens_00000002_00000002.npy, clip_count=1, token_count=12, embedding_dim=32
```

### 2. Sample training tokens

Total tokens: 24 + 12 = 36. Sample size: 24 (minimum of 24 and 36).

Proportional allocation:
- Batch 0: 24 * (24/36) = 16 tokens
- Batch 1: 24 * (12/36) = 8 tokens

Random indices are drawn from each batch. Result: `(24, 32)` — 24 tokens, each 32-dimensional.

### 3. Train compressor

**PCA**: Learns 8 principal components from 24 samples of 32 dimensions. `pca_components` shape: `(8, 32)`. `pca_mean` shape: `(32,)`. Now every 32-dim token becomes an 8-dim token.

**K-means**: Clusters the 24 projected tokens (each 8-dim) into 4 centroids. `centroids` shape: `(4, 8)`.

**Quantization thresholds**: Computed from residuals of 24 training tokens. `quantile_thresholds` shape: `(3, 8)`. `quantization_levels` shape: `(4, 8)`.

### 4. Compress batch 0 → shard 0

**Clip 0** (12 tokens, each 32-dim):
- PCA project: `(12, 32)` → `(12, 8)`
- Find nearest centroids: `(12,)` uint16 — e.g., `[2, 0, 2, 1, 3, 0, 2, 1, 0, 3, 2, 1]`
- Compute residuals: `(12, 8)` float32
- Quantize to 2-bit: `(12, 8)` uint8 values in {0,1,2,3}
- Pack: `(12, 2)` uint8 — 8 dims / 4 per byte = 2 bytes per token
- Coarse vector: mean of projected tokens, `(8,)` float32

**Clip 1**: Same process, different values.

**Write shard_00000000.widx:**
```
Header (32 bytes):
  magic = "WIDXSHD1"
  clip_count = 2
  token_count = 12
  pca_dim = 8
  residual_bytes_per_token = 2
  coarse_dim = 8
  record_size = 12*2 + 12*2 + 8*4 = 24 + 24 + 32 = 80

Record 0 (80 bytes): clip 0's centroid_ids + residuals + coarse_vector
Record 1 (80 bytes): clip 1's centroid_ids + residuals + coarse_vector

Total file size: 32 + 2*80 = 192 bytes
```

Compare to raw: 2 clips * 12 tokens * 32 dims * 4 bytes = 3,072 bytes. Compressed to 192 bytes — 16x compression even on this toy example.

### 5. Compress batch 1 → shard 1

Same process for the single clip. Produces `shard_00000001.widx` with 1 record.

### 6. Write enriched metadata

`shard_00000000.parquet` — 2 rows, original metadata + `shard_id=0`, `shard_offset={0,1}`
`shard_00000001.parquet` — 1 row, original metadata + `shard_id=1`, `shard_offset=0`

### 7. Consolidate

`clips.parquet` — 3 rows total:

```
clip_index  episode_id  dataset_name  robot_type  shard_id  shard_offset
0           episode_0   demo          testbot     0         0
1           episode_1   demo          testbot     0         1
2           episode_2   demo          testbot     1         0
```

### 8. Verify random access

```python
recovered = read_clip_from_shard("shard_00000001.widx", clip_index=0)
assert recovered.centroid_ids.shape == (12,)          # 12 tokens
assert recovered.quantized_residuals.shape == (12, 2) # 12 tokens, 2 packed bytes each
assert recovered.coarse_vector.shape == (8,)          # 8-dim PCA coarse vector
```

This is exactly what the pipeline test validates.

---

## Data Model: The Pydantic Classes

The codebase uses Pydantic models with `frozen=True` for all data structures. This means once created, they can't be modified. This is intentional — a compressed clip's data should never change after compression. Immutability prevents a whole class of bugs (accidentally overwriting centroid IDs, modifying shared arrays, etc.).

### CompressedClip

```python
class CompressedClip(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    centroid_ids: np.ndarray         # (token_count,) uint16
    quantized_residuals: np.ndarray  # (token_count, residual_bytes_per_token) uint8
    coarse_vector: np.ndarray        # (coarse_dim,) float32
```

The validators ensure correct dtypes and shapes. `centroid_ids` must be 1D uint16. `quantized_residuals` must be 2D uint8. `coarse_vector` must be 1D float32. The `model_post_init` check ensures `centroid_ids` and `quantized_residuals` have matching token counts.

`arbitrary_types_allowed=True` tells Pydantic to accept numpy arrays (normally it only handles JSON-serializable types).

### ShardHeader

```python
class ShardHeader(BaseModel):
    model_config = ConfigDict(frozen=True)

    clip_count: int       # Number of clips in the shard
    token_count: int      # Tokens per clip (constant within a shard)
    pca_dim: int          # PCA dimensionality
    residual_bytes_per_token: int  # Packed residual bytes per token
    coarse_dim: int       # Dimension of coarse vector
    record_size: int      # Total bytes per clip record
```

The `model_post_init` validates that `record_size` equals the sum of its parts. This catches file corruption — if any header field is wrong, the record_size won't match and you get an immediate error instead of silently reading garbage.

The `record_dtype()` method returns a numpy structured dtype that mirrors the binary layout. This is the key that enables memory-mapped reading — numpy knows exactly how to interpret each byte range within a record.

---

## How Everything Connects

Here's the full data lineage, from raw robot video to searchable compressed index:

```
Robot video (HuggingFace LeRobot dataset)
    │
    ▼
[Ingestion: ClipFormer]
    │  Splits video into fixed-length clips, preprocesses frames
    │  Output: (64, 3, 256, 256) pixel tensors + ClipMetadata
    │
    ▼
[Extraction: V-JEPA 2 encoder]
    │  Runs vision transformer, extracts spatial token embeddings
    │  Output: tokens_*.npy  (clip_count, 8192, 1280) float32
    │          metadata_*.parquet
    │
    ▼
[Compression: TokenCompressor]           ◄── YOU ARE HERE
    │
    │  Phase 1: Train
    │    Sample 500K tokens across all batches
    │    PCA: 1280 dims → 128 dims (10x)
    │    K-means: 32,768 centroids in 128-dim space
    │    Residual quantization: 2-bit per dimension
    │
    │  Phase 2: Compress
    │    For each clip:
    │      Project tokens to 128 dims
    │      Assign nearest centroid (uint16 ID)
    │      Compute & pack residuals (2 bits/dim → 32 bytes/token)
    │      Compute coarse vector (mean of projected tokens)
    │    Write binary shard (.widx)
    │    Write enriched metadata (.parquet)
    │
    │  Output: shard_*.widx     (custom binary, ~150x smaller than raw)
    │          clips.parquet    (consolidated metadata with shard pointers)
    │          compression_model/  (learned PCA + centroids + quantization)
    │
    ▼
[Future: Index]
    Load coarse vectors for fast first-pass search
    Decompress individual clip tokens for fine-grained matching
```

Each stage is independent — you can re-run compression without re-running extraction, and re-run indexing without re-running compression. The only coupling is the file format: compression reads extraction's output and produces files the index will read.
