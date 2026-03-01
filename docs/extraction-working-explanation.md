# How WorldIndex Extraction Works

## Start Here: Where Ingestion Left Off

The ingestion pipeline gave us clips — 64-frame chunks of preprocessed video, each as a `(64, 3, 256, 256)` tensor with metadata saying which dataset, episode, robot, and timestamp it came from. Think of these as "raw materials".

But a tensor of pixel values is useless for search. You can't ask "find me a clip where a robot arm reaches toward the top-right corner" by comparing raw pixels. Pixel values are noisy, lighting-dependent and don't encode spatial structure in a searchable way.

The extraction pipeline takes those clips and runs them through V-JEPA 2 — Meta's video world model — to produce **token embeddings**. Each clip goes in as pixel values and comes out as a compact matrix of numbers that encodes *what's happening where* in the scene. These embeddings are what the search index actually stores and compares against.

If ingestion was "chew through raw videos and prepare them", extraction is "run the model on every prepared clip and save the results to disk".

---

## High Level Architecture

The extraction pipeline runs as **three separate OS processes** communicating through **two queues**:

```
                    q1                          q2
  ┌──────────┐  ─────────>   ┌──────────┐  ─────────>   ┌──────────┐
  │ PRODUCER │               │ ENCODER  │               │  WRITER  │
  │          │  SharedMemory │          │  (tokens,     │          │
  │ Makes    │  descriptors  │ Runs     │   metadata)   │ Saves    │
  │ clips    │               │ V-JEPA 2 │               │ to disk  │
  └──────────┘               └──────────┘               └──────────┘
```

**Why three separate processes?** Because they do fundamentally different work that benefits from parallelism:

- The **producer** is I/O-bound — it streams video data from HuggingFace over the network and preprocesses frames. It spends most of its time waiting for data to download and for the video processor to resize/normalize images.
- The **encoder** is compute-bound — it runs the V-JEPA 2 model forward pass. On a GPU, this saturates the GPU. On CPU, it saturates all cores.
- The **writer** is I/O-bound again — it writes numpy arrays and parquet files to disk, plus updates an SQLite checkpoint.

If these ran sequentially in one process, the GPU would sit idle while the next clip downloads, and the CPU would sit idle while the disk flushes. With three processes, the producer is preparing clip N+1 while the encoder processes clip N while the writer saves clip N-1. The queues act as buffers so each process can work at its own pace.

**Why processes instead of threads?** Python's Global Interpreter Lock (GIL) prevents true parallel execution of Python code in threads (we allow Python 3.11+ but starting Python 3.14, we can use free threaded (no GIL) build). Processes sidestep the GIL entirely — each gets its own Python interpreter. For CPU-intensive model inference and numpy operations, this is essential.

---

## Producer: Reading Clips and Sharing Them via Shared Memory

The producer's job: iterate through all dataset configs, create clips using the ingestion pipeline's `ClipFormer`, and put them on queue `q1` for the encoder to consume.

Here's what happens for each clip:

### Step 1: Get a clip from ClipFormer

```python
clip_former = ClipFormer(dataset_config)
for processed, metadata in clip_former.iter_clips():
    # processed["pixel_values"] is shape (64, 3, 256, 256)
    # metadata is a ClipMetadata with episode_id, timestamps, etc.
```

This is the ingestion pipeline doing its thing — streaming data, subsampling, windowing, preprocessing. The producer doesn't care about the details; it just consumes the iterator.

### Step 2: Convert to numpy

```python
pixel_values = _to_numpy_array(processed["pixel_values"])
```

The video processor might return a PyTorch tensor or a numpy array depending on the backend. We normalize to a contiguous numpy array because shared memory works with raw bytes, not PyTorch tensors.

For a single clip, this array is `(64, 3, 256, 256)` of `float32` values. That's `64 * 3 * 256 * 256 * 4 bytes = 50,331,648 bytes ≈ 48 MB`.

### Step 3: Copy into shared memory

This is the most interesting part of the producer, so let's walk through it carefully.

**The problem:** We need to send a 48 MB array from the producer process to the encoder process. The standard way to pass data through a `multiprocessing.Queue` is pickling — Python serializes the object to bytes, sends the bytes through a pipe, and the receiver deserializes them. For a 48 MB numpy array, that means:
1. Serialize 48 MB in the producer (CPU time + memory allocation)
2. Copy 48 MB through the pipe (kernel copy)
3. Deserialize 48 MB in the encoder (CPU time + memory allocation)

That's three copies of 48 MB, plus serialization overhead. At thousands of clips, this adds up.

**The solution: shared memory.** Instead of sending the data through the pipe, we:
1. Allocate a block of shared memory visible to all processes
2. Copy the array into it (one copy)
3. Send only a tiny *descriptor* through the queue (the shared memory name, shape and dtype — maybe 200 bytes)
4. The encoder attaches to the same shared memory block and reads the array directly (zero additional copies)

Here's the actual code flow:

```python
# 1. Create a shared memory block big enough for the array
shared_memory = SharedMemory(create=True, size=pixel_values.nbytes)

# 2. Create a numpy array that views the shared memory buffer (no copy!)
shared_array = np.ndarray(
    pixel_values.shape,
    dtype=pixel_values.dtype,
    buffer=shared_memory.buf,
)

# 3. Copy pixel values into the shared memory
shared_array[...] = pixel_values

# 4. Send a lightweight descriptor through the queue
descriptor = SharedClipDescriptor(
    shm_name=shared_memory.name,    # e.g. "/psm_12345"
    shape=(64, 3, 256, 256),
    dtype="float32",
    metadata=metadata_dict,
)
q1.put(descriptor)  # This sends ~200 bytes, not 48 MB
```

The `SharedClipDescriptor` is a `NamedTuple` with four fields — the shared memory segment name (a string like `"/psm_12345"` that the OS uses to identify the shared block), the array shape, the dtype, and the clip metadata. This is all the encoder needs to reconstruct the array on its end.

### The resource tracker dance

There's a subtle but important detail here:

```python
resource_tracker.unregister(shared_memory._name, "shared_memory")
```

Python's `multiprocessing` module has a "resource tracker" — a background process that automatically cleans up shared memory blocks when the process that created them exits. This sounds helpful, but it's actually dangerous here. The producer creates the shared memory and immediately closes its handle (it doesn't need the data anymore). The *encoder* is the one who still needs it. If the resource tracker cleans it up because the producer's reference is gone, the encoder would try to read freed memory.

By unregistering from the tracker, we're saying: "I know what I'm doing. The encoder will clean this up when it's done." The encoder does exactly that — after reading the data, it calls `shared_memory.unlink()` to free the block.

The `try/finally` block ensures that if anything goes wrong *before* the descriptor is queued (e.g., the queue is full and we time out), we clean up the shared memory ourselves rather than leaking it:

```python
enqueued = False
try:
    # ... set up shared array, put descriptor on queue ...
    enqueued = True
finally:
    shared_memory.close()       # Always close our handle
    if not enqueued:
        shared_memory.unlink()  # Only unlink if encoder never got the descriptor
```

### Step 4: Attach a global clip index

The metadata gets a `clip_index` field that counts clips globally across all datasets:

```python
metadata_dict["clip_index"] = clip_index
```

If you have two datasets — DROID producing 4 clips and Bridge producing 3 clips — the indices are 0, 1, 2, 3 (DROID) then 4, 5, 6 (Bridge). This gives every clip a unique sequential ID used for checkpointing and file naming.

### Step 5: Signal completion

When all clips from all datasets are exhausted, the producer puts `None` (a sentinel) on the queue:

```python
q1.put(None)
```

This tells the encoder: "There are no more clips. Finish up."

### Resume support

The producer accepts a `start_clip_index` parameter. If we're resuming from a checkpoint (say clip 150 was the last successfully written one), we pass `start_clip_index=151`. The producer simply counts clips and skips any with index < 151:

```python
if clip_index < start_clip_index:
    clip_index += 1
    continue
```

This is not perfectly efficient — it still iterates through clips 0-150 in the ClipFormer, it just doesn't put them on the queue. But it's simple and correct, which matters more for a pipeline that will mostly run to completion anyway. That said, I might make this more efficient later on.

---

## The Encoder: Running V-JEPA 2

The encoder's job: take clip descriptors from `q1`, load each clip from shared memory, run it through the model, and put the resulting token embeddings on `q2` for the writer.

### Step 1: Load the model

```python
model = _load_encoder_model(model_id=model_id, device=device, model_loader=model_loader)
```

By default, this loads V-JEPA 2 from HuggingFace using `AutoModel.from_pretrained("facebook/vjepa2-vith-fpc64-256")`. On GPU, it uses `float16` for speed; on CPU, `float32` for compatibility (CPUs are generally reliable with float32, not float16).

The model is loaded once when the encoder process starts and reused for every clip. Loading a ViT-Huge model takes several seconds and several GB of memory, so you definitely don't want to do this per-clip.

In tests, the `model_loader` parameter lets you inject a fake model that doesn't require a GPU or downloading weights. This is how the test suite runs extraction tests without a real model. Essentially, mocking & stubbing.

### Step 2: Read from shared memory

```python
item = q1.get()                                          # Blocks until a descriptor arrives
descriptor = SharedClipDescriptor(*item)
shared_memory = SharedMemory(name=descriptor.shm_name)   # Attach to the block the producer created
shared_array = np.ndarray(
    descriptor.shape,
    dtype=np.dtype(descriptor.dtype),
    buffer=shared_memory.buf,
)
```

The encoder attaches to the same shared memory block by name. It creates a numpy array that *views* this memory (no copy). Now both processes can see the same bytes.

The encoder then converts this to a PyTorch tensor:

```python
pending_tensors.append(_to_tensor(shared_array))
pending_metadata.append(descriptor.metadata)
```

Note that `_to_tensor` does `np.array(value, copy=True)` — it copies the data out of shared memory into a regular numpy array, then wraps it as a torch tensor. This copy is necessary because we're about to unlink the shared memory:

```python
shared_memory.close()
shared_memory.unlink()  # Free the shared memory block
```

After unlinking, the shared memory block is gone from the OS. Any reference to it would segfault. That's why we copied first.

### Step 3: Batch and encode

The encoder accumulates clips into a batch before running the model:

```python
if len(pending_tensors) >= batch_size:
    _encode_batch(model, pending_tensors, pending_metadata, q2, device)
    pending_tensors = []
    pending_metadata = []
```

**Why batch?** GPUs are massively parallel. Running one clip at a time through V-JEPA 2 uses maybe 5% of the GPU's compute capacity. Running 8 clips at once uses 40%. Batching amortizes the overhead of GPU kernel launches and memory transfers.

A batch size of 8 means: accumulate 8 clips, stack them into a single tensor of shape `(8, 64, 3, 256, 256)`, and run the model once on the whole batch. This is dramatically faster than 8 individual forward passes.

The `_encode_batch` function:

```python
def _encode_batch(model, clips, metadata, q2, device):
    batch = torch.stack(clips, dim=0).to(device)   # Shape: (batch_size, 64, 3, 256, 256)
    with torch.no_grad():
        raw_output = model.get_vision_features(batch)

    encoded = _extract_tensor(raw_output)
    encoded_numpy = encoded.to(device="cpu", dtype=torch.float32).numpy()

    for clip_tokens, clip_metadata in zip(encoded_numpy, metadata, strict=True):
        q2.put((np.ascontiguousarray(clip_tokens), clip_metadata))
```

Let's trace the shapes for a concrete example. Say `batch_size=2` and the model is V-JEPA 2 ViT-Huge:

```
Input:  batch shape (2, 64, 3, 256, 256)
        2 clips, each 64 frames, 3 RGB channels, 256x256 pixels

Model:  V-JEPA 2 produces 256 spatial patches per frame (V-JEPA doesnt see pixels, it splits image into patches i.e. a grid of 16*16), each patch represented as a 1280-dimensional vector (ViT-Huge hidden dimension = 1280)

Output: encoded shape (2, 8192, 1280)
        2 clips, each with 8192 tokens (64 frames x ~128 patches (not 256) because the model did some internal temporal/spatial processing that reduced tokens by ~2x),
        each token is a 1280-dimensional embedding

Per-clip: (8192, 1280) — this goes on q2 along with metadata
```

The exact output shape depends on the model architecture (how it pools across frames, patch size, etc.), but the key point is: a 48 MB pixel tensor (input) becomes a much smaller embedding matrix that encodes spatial structure (output).

**Why `torch.no_grad()`?** We're doing inference, not training. Disabling gradient computation saves memory (no storing intermediate activations) and speeds things up.

**Why convert back to float32 and numpy?** The model may run in float16 on GPU. We convert to float32 on CPU for consistent storage format, then to numpy so it can be saved with `np.save` without a torch dependency at read time.

After encoding, each clip's tokens and metadata are put on `q2` individually (not as a batch). The writer consumes one clip at a time.

### Step 4: Flush remaining clips and signal completion

After the sentinel `None` arrives from the producer, there might be leftover clips in the pending batch (fewer than `batch_size`). The encoder runs one final batch on those:

```python
if pending_tensors:
    _encode_batch(model, pending_tensors, pending_metadata, q2, device)
```

Then sends its own sentinel to the writer:

```python
q2.put(None)
```

---

## The Writer: Saving to Disk with Atomic Writes and Checkpointing

The writer's job: take encoded tokens from `q2` and save them to disk in a format that's fast to load and search later. It also maintains a checkpoint so the pipeline can resume if interrupted (start_clip_index we talked about earlier).

### The output format

Each flush produces two files:

```
output/
  tokens_00000000_00000002.npy        # numpy array of shape (3, 8192, 1280) -> 3 because it contains 0,1,2
  metadata_00000000_00000002.parquet   # polars DataFrame with 3 rows
  tokens_00000003_00000005.npy
  metadata_00000003_00000005.parquet
  tokens_00000006_00000006.npy
  metadata_00000006_00000006.parquet
```

The filenames encode the clip index range: `tokens_00000000_00000002.npy` contains clips 0, 1, and 2. The zero-padded 8-digit format supports up to 99,999,999 clips before filenames get wider (plenty for any realistic dataset).

**Token files** (`.npy`) store the raw embeddings. Shape is `(num_clips_in_batch, token_dim_1, token_dim_2)` — for example, `(3, 8192, 1280)` if flushing 3 clips of V-JEPA 2 ViT-Huge output. These are float32 numpy arrays. Using `.npy` format means they can be loaded with a single `np.load()` call — no parsing, no deserialization, just an mmap of raw floats.

**Metadata files** (`.parquet`) store the clip metadata as a columnar table. Each row is one clip:

```
┌────────────┬───────────────────────┬────────────┬──────────────────┬──────────────┬─────────────────┬───────────────┬──────────────────────┬──────────────────────┐
│ clip_index │ episode_id            │ dataset_na │ robot_type       │ clip_start_f │ clip_end_frame  │ timestamp_sta │ language_instruction │ num_original_frames  │
│ ---        │ ---                   │ me         │ ---              │ rame         │ ---             │ rt            │ ---                  │ ---                  │
│ i64        │ str                   │ ---        │ str              │ ---          │ i64             │ ---           │ str                  │ i64                  │
│            │                       │ str        │                  │ i64          │                 │ f64           │                      │                      │
╞════════════╪═══════════════════════╪════════════╪══════════════════╪══════════════╪═════════════════╪═══════════════╪══════════════════════╪══════════════════════╡
│ 0          │ lerobot/droid_1.0.1_0 │ droid_1.0. │ franka           │ 0            │ 98              │ 0.0           │ pick up the mug      │ 27                   │
│ 1          │ lerobot/droid_1.0.1_1 │ 1          │ franka           │ 0            │ 2998            │ 0.0           │ open the drawer      │ 64                   │
│ 2          │ lerobot/droid_1.0.1_1 │ droid_1.0. │ franka           │ 1496         │ 2998            │ 99.73         │ open the drawer      │ 64                   │
│            │                       │ 1          │                  │              │                 │               │                      │                      │
└────────────┴───────────────────────┴────────────┴──────────────────┴──────────────┴─────────────────┴───────────────┴──────────────────────┴──────────────────────┘
```

Parquet is a columnar binary format. It's compact (compression built in), fast to query (you can read just the `clip_index` column without loading all columns), and supported by both Polars and Pandas. It's the standard for analytical data.

### Buffering and flushing

The writer doesn't write every clip individually. It accumulates clips in a buffer and flushes when the buffer reaches `flush_size`:

```python
token_buffer.append(np.asarray(tokens, dtype=np.float32))
metadata_buffer.append(dict(metadata))

if len(token_buffer) >= flush_size:
    _flush_buffers(token_buffer, metadata_buffer, output_dir, checkpoint_db_path)
    token_buffer = []
    metadata_buffer = []
```

**Why buffer?** Two reasons:

1. **Fewer files.** If you flush every clip, processing 76,000 clips produces 76,000 file pairs (76K for .npy and 76K for .parquet) — 152,000 files. Filesystems hate this. Directory listings slow down, metadata operations become expensive, and any downstream code that globbes `tokens_*.npy` gets sluggish. With `flush_size=100`, you get ~760 file pairs instead.

2. **Larger writes.** A single clip's token array might be 40 MB. Writing one 40 MB file is slower than writing one 4 GB file (100 clips). That does not mean 40 MB takes longer than 4 GB (obviously 4 GB is more bytes). However, Disk I/O has fixed overhead per operation (open file, create file, allocate metadata, write, flush, close), so fewer, larger writes are more efficient. Imagine doing this for 100 clips, each 40MB vs 1 thats 4GB. This cannot be turned into a much larger write because we might lose data, making the backend recompute everything, causing inefficiency.

After the sentinel arrives, the writer flushes whatever remains in the buffer (even if it's less than `flush_size`). That's why you might see a final file with fewer clips than the others.

### Atomic writes

The writer uses a **write-then-rename** pattern:

```python
temp_token_path = token_path.with_suffix(".npy.tmp")     # tokens_00000000_00000002.npy.tmp
temp_metadata_path = metadata_path.with_suffix(".parquet.tmp")

np.save(temp_token_path, stacked_tokens)                  # Write to temp file
metadata_frame.write_parquet(temp_metadata_path)

temp_token_path.replace(token_path)                       # Atomic rename
temp_metadata_path.replace(metadata_path)
```

**Why?** Imagine the process crashes halfway through writing `tokens_00000000_00000002.npy`. You'd have a corrupted file on disk — half-written, invalid numpy format. When you resume, the pipeline would see this file exists, assume those clips are done, and skip them. But the data is garbage. You've lost clips 0-2 forever.

With write-then-rename: the temporary file might be corrupted, but the final filename never exists in a half-written state. `Path.replace()` is an atomic operation on POSIX filesystems — it either fully succeeds or doesn't happen at all. If the process crashes during the `np.save()`, the `.tmp` file is left behind but the final file doesn't exist, so the checkpoint won't advance past those clips. On resume, they get re-processed.

### The checkpoint: an SQLite database

After each successful flush, the writer updates a checkpoint:

```python
_write_checkpoint(checkpoint_db_path, last_clip_index)
```

The checkpoint is a single row in an SQLite table:

```sql
CREATE TABLE extraction_checkpoint (
    checkpoint_key TEXT PRIMARY KEY,
    clip_index     INTEGER NOT NULL,
    updated_at     TEXT NOT NULL
);
```

There's exactly one row with `checkpoint_key = "single_node_extraction"`. After flushing clips 0-2, the row says `clip_index = 2`. After flushing clips 3-5, it says `clip_index = 5`. It always holds the index of the last clip that was *fully written to disk*.

**Why SQLite?** It's a single-file database with ACID transactions. The checkpoint write happens within a transaction, so if the process crashes mid-update, the database rolls back to its previous state. No corruption. This is much safer than writing to a plain text file (which can be truncated on crash) and simpler than a full database server.

**Why a database for a single integer?** It's future-proofed for distributed extraction where multiple workers need to coordinate. For now on a single node, it's overkill — but it works perfectly and the overhead is negligible (one SQLite write per flush, not per clip).

### How the checkpoint is used on resume

When `run_extraction` starts, it reads the checkpoint:

```python
resume_from = _read_checkpoint(config.checkpoint_db) + 1
```

If the checkpoint says `clip_index = 4`, that means clips 0-4 are safely on disk. So `resume_from = 5`. This value is passed to the producer as `start_clip_index=5`, and the producer skips clips 0-4.

If this is a fresh run (no checkpoint DB exists), `_initialize_checkpoint_db` creates the table with `clip_index = -1`. So `resume_from = -1 + 1 = 0` — start from the beginning.

---

## The Orchestrator: run_extraction()

`run_extraction` is the function that ties everything together. It creates the processes, starts them, monitors them, and handles shutdown. Let's walk through it.

### Step 1: Set up

```python
config.output_dir.mkdir(parents=True, exist_ok=True)
_initialize_checkpoint_db(config.checkpoint_db)
resume_from = _read_checkpoint(config.checkpoint_db) + 1
```

Create the output directory if it doesn't exist. Initialize the checkpoint database. Figure out where to resume from.

### Step 2: Select the multiprocessing context

```python
context = _select_context(config.start_method)
```

Python's `multiprocessing` module supports three ways to create child processes:

- **`spawn`** (default on macOS/Windows): Starts a fresh Python interpreter and imports the necessary modules. Safest — child has a clean slate. But slower because of the import overhead.
- **`fork`** (default on Linux): Copies the parent process's memory space. Fast — no reimporting. But can be dangerous if the parent has threads, open file descriptors, or CUDA contexts (fork + CUDA = instant crash).
- **`forkserver`**: A compromise. Forks a clean server process early, then forks children from that server.

The pipeline defaults to `spawn` if available, which is the safest option. You can override this via `config.start_method`.

### Step 3: Create queues and processes

```python
q1 = context.Queue(maxsize=config.queue_depth)
q2 = context.Queue(maxsize=config.queue_depth)
stop_event = context.Event()
```

Both queues have a bounded size (`queue_depth`, default 8). This creates **backpressure**: if the producer is faster than the encoder, `q1` fills up to 8 items, and then `q1.put()` blocks the producer until the encoder consumes something. This prevents the producer from loading thousands of clips into memory while the encoder chugs along. Memory usage stays bounded.

Similarly, if the encoder is faster than the writer, `q2` fills up and the encoder blocks. In practice, the encoder (running the model) is almost always the bottleneck, so `q1` fills up and `q2` stays nearly empty.

The `stop_event` is a cross-process flag used for graceful shutdown.

Then three processes are created:

```python
producer = context.Process(target=producer_fn, args=(...))
encoder  = context.Process(target=encoder_fn,  args=(...))
writer   = context.Process(target=writer_fn,   args=(...))
```

Each process gets a human-readable name (`"worldindex-producer"`, etc.) for debugging — these show up in `ps aux` or activity monitors.

### Step 4: Monitor the processes

After starting all three processes, the orchestrator enters a monitoring loop:

```python
while True:
    for process in processes:
        process.join(timeout=_JOIN_POLL_SECONDS)   # 0.5 seconds

    # Check for failures
    failed = [p for p in processes if p.exitcode not in (None, 0)]
    if failed:
        # A process crashed — shut everything down
        ...

    # Log progress periodically
    if now - last_progress_log >= 5.0:
        checkpoint = _read_checkpoint(config.checkpoint_db)
        logger.info("extraction_progress", processed_clips=checkpoint + 1)

    # Check if all processes finished
    if all(not p.is_alive() for p in processes):
        break
```

The loop polls every 0.5 seconds. It does three things:

1. **Detects failures.** If any process exits with a non-zero code (crash, unhandled exception), the orchestrator sets the stop event and pushes sentinel values to unblock any process stuck on a queue.

2. **Logs progress.** Every 5 seconds, it reads the checkpoint database to see how many clips have been written. This gives you log lines like:
   ```
   extraction_progress resume_from=0 processed_clips=347 last_clip_index=346
   ```

3. **Detects completion.** When all three processes finish (exit code 0), the loop breaks.

### Step 5: Handle interrupts

If you press Ctrl+C:

```python
except KeyboardInterrupt:
    interrupted = True
    _signal_stop(stop_event=stop_event, q1=q1, q2=q2, force_writer=False)
```

The stop event gets set. The producer and encoder check this event periodically and exit their loops early. The writer is NOT force-stopped (`force_writer=False`) — it's allowed to finish flushing its current buffer so that the data already computed isn't lost. The checkpoint will reflect whatever was fully written.

This means you can safely Ctrl+C a long extraction run, and when you restart, it picks up exactly where it left off with zero duplicate or missing clips.

### Step 6: Clean up

```python
finally:
    _join_or_terminate(processes, q1=q1, q2=q2)
```

`_join_or_terminate` gives processes a grace period to finish (about 10 seconds). If a process is still alive after that, it sends another sentinel to try unblocking it, waits one more second, then `terminate()`s it (SIGTERM). Finally, it closes and joins the queue threads to flush any remaining pipe buffers.

This escalation ladder (stop event → sentinel → wait → terminate) handles every case:
- Normal completion: processes already exited, join is immediate.
- Graceful interrupt: stop event triggers, processes exit within seconds.
- Hung process: after the timeout, force-terminated. You don't wait forever.

---

## A Complete Walkthrough: Two Datasets, Seven Clips

Let's trace a full extraction run with concrete numbers. We have two dataset configs:

```python
ExtractionConfig(
    dataset_configs=[alpha_config, bravo_config],
    model_id="facebook/vjepa2-vith-fpc64-256",
    device="cuda:0",
    batch_size=2,
    queue_depth=2,
    flush_size=3,
    output_dir=Path("output/"),
    checkpoint_db=Path("output/checkpoint.sqlite3"),
)
```

The "alpha" dataset produces 4 clips. The "bravo" dataset produces 3 clips. Total: 7 clips (indices 0 through 6).

### Phase 1: Startup

```
Orchestrator:
  - Creates output/ directory
  - Initializes checkpoint DB with clip_index = -1
  - resume_from = -1 + 1 = 0 (fresh run)
  - Creates q1 (maxsize=2), q2 (maxsize=2), stop_event
  - Starts producer, encoder, writer processes
```

### Phase 2: Producer fills q1

```
Producer:
  - Creates ClipFormer for alpha dataset
  - clip 0 from alpha → shared memory "psm_001" → descriptor on q1
  - clip 1 from alpha → shared memory "psm_002" → descriptor on q1
  - q1 is now full (maxsize=2), producer BLOCKS on clip 2
```

### Phase 3: Encoder starts consuming

```
Encoder:
  - Loads V-JEPA 2 model onto cuda:0 (takes ~5 seconds. For me, on CPU, it took 60 seconds+ and my laptop got cooked)
  - Reads clip 0 descriptor from q1 → attaches to psm_001 → copies to tensor → unlinks psm_001
    pending_tensors = [tensor_0]
  - Reads clip 1 descriptor from q1 → attaches to psm_002 → copies to tensor → unlinks psm_002
    pending_tensors = [tensor_0, tensor_1]
  - len(pending_tensors) >= batch_size (2 >= 2), so: ENCODE BATCH
    - Stack: shape (2, 64, 3, 256, 256)
    - model.get_vision_features(batch) → shape (2, 8192, 1280)
    - Put (tokens_0, metadata_0) on q2
    - Put (tokens_1, metadata_1) on q2
    - q2 is now full (maxsize=2), encoder BLOCKS

Meanwhile, q1 has space again, so:
Producer:
  - clip 2 from alpha → shared memory "psm_003" → descriptor on q1
  - clip 3 from alpha → shared memory "psm_004" → descriptor on q1
  - q1 full again, producer blocks
```

### Phase 4: Writer starts consuming

```
Writer:
  - Reads (tokens_0, metadata_0) from q2 → buffer = [clip_0]
  - Reads (tokens_1, metadata_1) from q2 → buffer = [clip_0, clip_1]
  - len(buffer) < flush_size (2 < 3), so: keep accumulating

q2 has space, so encoder unblocks and processes clips 2-3...

Encoder:
  - Reads clips 2, 3 from q1 → batch → encode → puts on q2

Writer:
  - Reads (tokens_2, metadata_2) from q2 → buffer = [clip_0, clip_1, clip_2]
  - len(buffer) >= flush_size (3 >= 3), so: FLUSH
    - Writes tokens_00000000_00000002.npy.tmp → renames to tokens_00000000_00000002.npy
    - Writes metadata_00000000_00000002.parquet.tmp → renames to metadata_00000000_00000002.parquet
    - UPDATE checkpoint SET clip_index = 2
    - Clear buffer
```

### Phase 5: Second dataset

```
Producer:
  - Alpha exhausted, moves to bravo dataset
  - Creates ClipFormer for bravo
  - clip 4 from bravo → q1
  - clip 5 from bravo → q1
  - clip 6 from bravo → q1 (blocks until space)

Encoder processes clips 4-5 (batch of 2), then clip 6 (batch of 1 — the remainder):
  - Batch 1: clips 4, 5 → encode → q2
  - Batch 2: clip 6 → encode → q2 (final, partial batch)
  - Sends sentinel (None) to q2

Writer:
  - Accumulates clips 3, 4, 5 → flush_size reached → FLUSH
    - tokens_00000003_00000005.npy
    - metadata_00000003_00000005.parquet
    - checkpoint = 5
  - Accumulates clip 6 → reads sentinel (None) → FLUSH REMAINING
    - tokens_00000006_00000006.npy
    - metadata_00000006_00000006.parquet
    - checkpoint = 6
```

### Phase 6: Completion

```
All processes exit with code 0.
Orchestrator returns final_checkpoint = 6.

On disk:
  output/
    tokens_00000000_00000002.npy      shape (3, 8192, 1280)
    metadata_00000000_00000002.parquet 3 rows
    tokens_00000003_00000005.npy      shape (3, 8192, 1280)
    metadata_00000003_00000005.parquet 3 rows
    tokens_00000006_00000006.npy      shape (1, 8192, 1280)
    metadata_00000006_00000006.parquet 1 row
    checkpoint.sqlite3                clip_index = 6
```

---

## Resume: What Happens When You Restart After an Interruption

Say you Ctrl+C'd after the first flush (checkpoint = 2). On restart:

```
1. _read_checkpoint() returns 2
2. resume_from = 2 + 1 = 3
3. Producer starts with start_clip_index=3
4. Producer skips clips 0, 1, 2 from alpha (they're already on disk)
5. Producer starts yielding from clip 3 onward
6. Encoder and writer proceed normally
7. New files appear alongside the old ones:
     tokens_00000003_00000005.npy   (new)
     tokens_00000006_00000006.npy   (new)
   The existing tokens_00000000_00000002.npy is untouched
```

No clips are duplicated. No clips are missing. The global clip indices stay consistent because the producer always counts from 0 — it just skips putting the first N on the queue.

---

## The Test Fakes: How Tests Work Without a Real Model or Real Data

The test suite needs to test the full pipeline — producer, encoder, writer, and their interactions — without downloading GB of model weights or streaming GB of video data. It does this with three fakes in `tests/extraction/spawn_fakes.py`:

### FakeClipFormer

Replaces the real `ClipFormer`. Instead of streaming from HuggingFace:

```python
class FakeClipFormer:
    def iter_clips(self):
        for clip_number in range(self._clip_count):
            pixel_values = np.full(self._clip_shape, fill_value=float(clip_number), dtype=np.float32)
            yield ({"pixel_values": pixel_values}, FakeMetadata(...))
```

It yields a fixed number of clips where every pixel value is the clip number (all 0.0s for clip 0, all 1.0s for clip 1, etc.). This makes it easy to verify in tests that the right clips ended up in the right places.

The clip count is encoded in the repo_id string: `"tests/alpha__4"` means "dataset named alpha, 4 clips." The `build_fake_clip_former` factory function parses this:

```python
def build_fake_clip_former(config):
    # config.repo_id = "tests/alpha__4"
    # _parse_clip_count("tests/alpha__4") → 4
    # _parse_dataset_label("tests/alpha__4") → "alpha"
    return FakeClipFormer(config, clip_count=4, dataset_name="alpha")
```

### FakeVisionModel

Replaces V-JEPA 2. Instead of running a real transformer:

```python
class FakeVisionModel:
    def get_vision_features(self, batch):
        # Output shape: (batch_size, *token_shape)
        # Fill each clip's tokens with a deterministic value based on input
        outputs = torch.empty((batch_size, *self._token_shape), ...)
        base_values = batch.float().mean(dim=tuple(range(1, batch.ndim)))
        for i in range(batch_size):
            outputs[i].fill_(float(base_values[i].item() + i))
        return outputs
```

The token shape is encoded in the model_id: `"fake:6x5"` means token output of shape `(6, 5)`. This lets tests control the output dimensions. The values are deterministic — derived from the mean of the input — so tests can verify that the right input produced the right output.

### How tests use the fakes

The fakes are injected via the `clip_former_factory` and `model_loader` config fields:

```python
ExtractionConfig(
    dataset_configs=[DatasetConfig(repo_id="tests/alpha__4", ...)],
    model_id="fake:6x5",
    clip_former_factory="tests.extraction.spawn_fakes:build_fake_clip_former",
    model_loader="tests.extraction.spawn_fakes:load_fake_encoder_model",
    ...
)
```

These are Python import paths. The pipeline uses `_load_callable` to dynamically import and call them:

```python
def _load_callable(import_path):
    # "tests.extraction.spawn_fakes:build_fake_clip_former"
    module_path, attribute = import_path.split(":", maxsplit=1)
    module = importlib.import_module(module_path)
    return getattr(module, attribute)
```

This means the pipeline code has zero test-specific logic. It doesn't know or care whether it's running a real model or a fake one — it just calls whatever factory function the config points to. Clean separation.

---

## The Dependency Injection Points

The pipeline has two extension points where you can swap in custom implementations:

### clip_former_factory

A callable with signature `(DatasetConfig) -> ClipFormer-like object`. The returned object must have an `iter_clips()` method that yields `(processed_dict, metadata)` pairs.

**Default (None):** Uses the real `ClipFormer` from the ingestion module.

**Test usage:** `build_fake_clip_former` returns a `FakeClipFormer`.

**Real integration test usage:** `build_limited_real_clip_former` wraps the real `ClipFormer` but stops after N clips (controlled by the `WORLDINDEX_REAL_EXTRACTION_CLIP_LIMIT` env var, default 20). This lets you run the full pipeline against real data but cap it so the test finishes in minutes, not hours.

### model_loader

A callable with signature `(model_id: str, device: str) -> model`. The returned model must have a `get_vision_features(batch)` method.

**Default (None):** Loads via `AutoModel.from_pretrained(model_id)` with the appropriate dtype.

**Test usage:** `load_fake_encoder_model` returns a `FakeVisionModel`.

---

## Glossary

**Token embeddings**: The output of running a clip through V-JEPA 2. A matrix where each row is a "token" — a learned representation of a spatial-temporal region of the video. These tokens are what gets indexed and compared during search.

**Shared memory**: An OS feature where multiple processes can access the same block of physical memory. Used here to transfer 48 MB clip tensors from the producer to the encoder without copying through a pipe.

**Sentinel (None)**: A special value put on a queue to signal "no more data." The producer sends None to the encoder; the encoder sends None to the writer.

**Backpressure**: When a queue is full, the putting process blocks. This prevents fast processes from overwhelming slow ones and keeps memory usage bounded.

**Checkpoint**: A persistently stored integer saying "clips 0 through N are safely on disk." Used to resume after interruptions without reprocessing.

**Atomic write**: Writing to a temporary file then renaming it to the final name. Ensures the final file is either complete or doesn't exist — never half-written.

**Flush size**: How many clips to accumulate before writing to disk. Larger values = fewer, bigger files = more efficient I/O, but more data at risk of being lost on crash (the unwritten buffer).

**Batch size**: How many clips to run through the model at once. Larger values = better GPU utilization, but more GPU memory needed. A ViT-Huge at batch size 8 needs roughly 16 GB of VRAM.

**Queue depth**: Maximum number of items a queue can hold before `put()` blocks. Controls how far ahead the producer can get before the encoder catches up.

**Multiprocessing start method**: How child processes are created. `spawn` (safe, slow), `fork` (fast, risky with CUDA), `forkserver` (compromise).
