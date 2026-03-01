# How WorldIndex Ingestion Works

## Start Here: What Problem Are We Solving?

You have 76,000+ robot demonstration videos scattered across HuggingFace. A researcher wants to upload a photo of their lab and find demonstrations where robots did something similar in a similar-looking workspace.

To make that search possible, you first need to **process every single video** into a format that a model (V-JEPA 2) can understand and that you can later search against. That processing is what this ingestion pipeline does. It is the part that chews through raw robot videos and spits out searchable tensors.

Think of it like building a Google search index, except instead of crawling web pages, you are crawling robot videos, and instead of extracting text tokens, you are extracting spatial visual patches from a world model.

This is purely the "read every video and prepare it" step, nothing in the ingestion pipeline does the actual searching (as the name suggests).

---

## What Do Robot Datasets Generally Look Like?

Robot datasets on HuggingFace (via the LeRobot library) are organized into **episodes**. An episode is one continuous recording of a robot doing a task like picking up a mug, opening a drawer or inserting a peg.

Each episode is a sequence of **samples**. Each sample is one moment in time and looks something like this (not exactly but you get the point):

```python
{
    "episode_index": 0,
    "frame_index": 42,
    "observation": {
        "images": {
            "exterior_1_left": <a PIL Image of what the external camera sees>,
            "wrist_cam": <a PIL Image of what the wrist camera sees>,
        },
        "state": [0.1, 0.2, 0.3, ...],   # joint angles, gripper state, etc.
    },
    "action": [0.01, -0.02, ...],          # what the robot did next
    "language_instruction": "pick up the mug",
}
```

We only care about three things from each sample:
1. **The image** — one specific camera view (e.g. the external left camera)
2. **The frame number** — so we know the ordering
3. **The language instruction** — what the robot was told to do (if available)

We ignore the robot state, actions, other camera views or anything else thats available. We are building a visual index.

---

## Telling the System About a Dataset

Each dataset is different. They use different camera names, different frame rates, different robots. So we describe each one in a YAML config file. Here is the config we use for the DROID dataset:

```yaml
# config/datasets/droid.yaml
repo_id: "lerobot/droid_1.0.1"
image_key: "observation.images.exterior_1_left"
source_fps: 15
target_fps: 4
robot_type: "franka"
language_key: "language_instruction"
clip_length: 64
clip_stride: 32
```

Let's go through every field:

**`repo_id`** is the HuggingFace repository. This is where the data lives. When we say `lerobot/droid_1.0.1`, it means "go to huggingface.co, find the dataset called `droid_1.0.1` under the `lerobot` organization, and stream it". BTW actual URL is here - [HuggingFace LeRobot Droid Dataset](https://huggingface.co/datasets/lerobot/droid_1.0.1)

**`image_key`** is the path to the image inside each sample's nested dictionary. DROID stores its external camera image at `sample["observation"]["images"]["exterior_1_left"]`. Different datasets put images in different places — ALOHA puts it at `sample["observation"]["images"]["top"]`, Bridge puts it at `sample["observation"]["images"]["image_0"]`. The dot notation (`observation.images.exterior_1_left`) tells our code how to dig into the nested dict to find the image.

**`source_fps`** is how fast the original video was recorded. DROID records at 15 frames per second. ALOHA records at 50 fps (very fast, lots of redundant frames). Bridge records at 5 fps (very slow, already sparse).

**`target_fps`** is what we downsample to. V-JEPA 2 works best at 4 fps (their paper mentions they used 16 frames per 4 seconds). We always downsample to 4, regardless of the source rate. This means DROID gets reduced by 3.75x, ALOHA by 12.5x, and Bridge by only 1.25x. More about the conversion later.

**`robot_type`** is metadata — "franka", "aloha", "widowx". We store it alongside each clip so that when search results come back, you know what kind of robot was in the video. You should actually go and check out all of these robots, very cool.

**`language_key`** tells us where to find the human-written task description. DROID has these ("pick up the mug", "open the drawer"). ALOHA and Bridge don't (I think its because each task episodes are stored in a different repo for example all episodes related to static cups open are stored here [HF](https://huggingface.co/datasets/lerobot/aloha_static_cups_open), I might be wrong though), so their configs omit this field. When it's missing, we just store `None`.

**`clip_length`** is how many frames V-JEPA 2 expects in one chunk. It wants exactly 64 frames. We cannot give it 63 or 65.

**`clip_stride`** is how far the sliding window moves between clips. At 32, each consecutive clip overlaps the previous one by 50%. More on what and why later.

These configs are loaded by a Pydantic model (`DatasetConfig`) which validates that fps values are positive, that required fields are present, etc. The `from_yaml` classmethod reads a YAML file and returns a validated config object. Nothing exotic.

---

## What Happens When You Create a ClipFormer

```python
config = DatasetConfig.from_yaml(Path("config/datasets/droid.yaml"))
clip_former = ClipFormer(config)
```

The constructor does three things:

**1. Opens the dataset.** It creates a `StreamingLeRobotDataset` — this is a LeRobot class that streams data from HuggingFace over the network. That means you dont have to download terabytes to disk. Samples arrive one at a time over HTTP as you iterate. The `streaming=True, shuffle=False, buffer_size=1, max_num_shards=1` settings mean: stream in order, don't randomize, don't buffer ahead, use a single data shard. We want deterministic sequential access. Also, simplicity.

In tests, you pass a fake dataset object instead, so no network calls happen.

**2. Loads the video processor.** This is V-JEPA 2's preprocessor from HuggingFace Transformers. It knows how to resize images to 256x256, normalize pixel values, and stack frames into tensors. The model identifier is `facebook/vjepa2-vith-fpc64-256` — that's the ViT-Huge (Vision Transformer) V-JEPA 2 model trained on 64-frame clips at 256x256 resolution.

Again, in tests, you pass a fake processor.

**3. Figures out episode boundaries.** Some datasets have an **episode index** — a precomputed lookup table that tells you "episode 0 is rows 0-449, episode 1 is rows 450-1199, episode 2 is rows 1200-1799, ..." This looks like:

```python
dataset.episode_data_index = {
    "from": [0,    450,  1200],
    "to":   [450, 1200,  1800],
}
```

If this exists, we parse it into a list of `(start, stop)` tuples: `[(0, 450), (450, 1200), (1200, 1800)]`. This is the **indexed path** — we can jump directly to any episode by row number.

If this doesn't exist (some datasets don't have it), we fall back to the **streaming path** — we read every sample sequentially and group them by their `episode_index` field. Slower ofcourse but works for any dataset.

The constructor also tries to figure out the total number of episodes (either from the index length or from `dataset.num_episodes`) so we can validate episode IDs later.

---

## The Main Loop: iter_clips()

This is the entry point. You call it, and it yields `(processed_clip, metadata)` pairs one at a time:

```python
for processed_clip, metadata in clip_former.iter_clips():
    # processed_clip["pixel_values"] is a tensor of shape (64, 3, 256, 256) where theres 64 time steps, 3 rgb channels, 256*256 frame resolution
    # metadata tells you where this clip came from
    do_something(processed_clip, metadata)
```

Internally, `iter_clips()` picks the right strategy based on what the constructor discovered:

**If we have an episode index (indexed path):**
```
for each episode_index in 0, 1, 2, ...:
    load all frames for this episode (random access by row number)
    process this episode into clips
    yield each clip
```

**If we don't (streaming and slower path):**
```
iterate every sample in order
group samples into episodes (detect when episode_index changes)
when an episode is complete, process it into clips
yield each clip
```

Both paths converge on the same function: `_emit_episode_clips()`. By the time we get there, we have the same thing regardless of path: a list of image frames, a list of frame numbers, and an optional language instruction. The rest of the pipeline is identical.

---

## Loading an Episode (Indexed Path): _load_episode()

When we have an episode index, loading episode `i` is straightforward. We look up `(start, stop) = self._episode_ranges[i]`, then read rows `start` through `stop - 1` from the dataset.

For each row, we extract three things:

**The image frame.** We use `_get_nested_value(sample, "observation.images.exterior_1_left")` to dig into the nested dict. This method first checks if the full dotted string exists as a literal key in the sample (some streaming datasets flatten their keys, for example Droid does this). If not, it splits on dots and walks the dict: `sample["observation"]["images"]["exterior_1_left"]`.

**The frame number.** We read `sample["frame_index"]`. If a dataset doesn't provide this field, we fall back to the loop counter (0, 1, 2, ...). Frame numbers matter because they tell us the temporal ordering (what frame came first according to time) and let us compute timestamps.

**The language instruction.** If the config has a `language_key`, we grab it from the first sample that has one. We only need it once per episode since it's the same instruction for every frame ("pick up the mug" doesn't change mid-episode).

The result is a tuple: `(frames, frame_numbers, language_instruction)`.

---

## Streaming Episodes: _iter_streamed_episodes()

When there is no episode index, we can't jump to a specific episode. We have to iterate through every sample and figure out where episodes begin and end.

The logic works like a state machine. We track the `current_episode_index`. As we read each sample:

- If the sample's `episode_index` matches the current one, we add its frame to the current episode's accumulator.
- If it's a new episode index, the previous episode is complete. We yield the accumulated episode and start a new accumulator.
- When the stream ends, we yield whatever is left in the accumulator.

There's one optimization: if we're looking for a specific episode (via `target_episode_index`), we skip samples from earlier episodes and stop as soon as we've passed the target. This avoids reading the entire dataset when you only want one episode.

### Where do samples come from? _iter_stream_samples()

Most of the time, iterating the dataset is simple: `yield from self._dataset`. You just iterate and get samples.

But some LeRobot datasets have a special `make_frame()` method. These datasets store raw data in `dataset.hf_dataset` (the underlying HuggingFace dataset) and provide `make_frame()` to transform raw rows into properly formatted samples. Think of it as a decode step — the raw rows might have encoded video frames or compressed data, and `make_frame()` turns them into the nice nested dict with PIL images.

When a dataset has both `hf_dataset` and `make_frame`, we:
1. Iterate the raw `hf_dataset`
2. Feed rows into `make_frame()`, which is a generator that yields decoded samples
3. Yield those decoded samples

There is a `RuntimeError` catch here that deserves explanation. In Python 3.7+, if a generator internally calls `next()` on an exhausted iterator without catching `StopIteration`, Python converts that `StopIteration` into a `RuntimeError` with the message `"generator raised StopIteration"`. This happens when `make_frame()` is a generator that consumes from our iterator and the iterator runs dry. It's not an error — it means "done." We catch it and return cleanly.

---

## Processing an Episode: _emit_episode_clips()

This is where the real transformation happens. It takes raw frames from an episode and produces searchable clips. There are three stages: subsample, window, preprocess.

### Stage 1: Sort by frame number

First, we sort frames by their frame number. This is a safety measure. Frames should already be in order from how we loaded them, but sorting guarantees it. On already-sorted data, Python's Timsort runs in O(n), so this costs nothing in practice.

### Stage 2: Subsample — _subsample_episode()

This is the FPS conversion step. The goal: convert from `source_fps` (e.g. 15) to `target_fps` (4).

**Why not just take every Nth frame?** Because the ratio isn't always a clean integer. DROID is 15 fps source, 4 fps target. That's a ratio of 3.75. You can't take "every 3.75th frame." If you take every 3rd, you're sampling at 5 fps (too fast). If you take every 4th, you're sampling at 3.75 fps (too slow, and the drift accumulates over long episodes).

Instead, we use **nearest-neighbor timestamp matching**.

For a 100-frame DROID episode:

```
The episode spans 100 frames at 15 fps = 6.6 seconds.
At 4 fps, we need frames at: 0.00s, 0.25s, 0.50s, 0.75s, 1.00s, ...

Source frame timestamps (at 15 fps):
  frame 0:  0.000s
  frame 1:  0.067s
  frame 2:  0.133s
  frame 3:  0.200s
  frame 4:  0.267s    <-- closest to target 0.25s
  frame 5:  0.333s
  frame 6:  0.400s
  frame 7:  0.467s
  frame 8:  0.533s    <-- closest to target 0.50s
  ...

Target 0 (0.000s) → pick source frame 0  (0.000s)  — exact match
Target 1 (0.250s) → pick source frame 4  (0.267s)  — 0.017s off, best available
Target 2 (0.500s) → pick source frame 8  (0.533s)  — 0.033s off, best available
Target 3 (0.750s) → pick source frame 11 (0.733s)  — 0.017s off, best available
...
Target 26 (6.500s) → pick source frame 98 (6.533s) — last meaningful frame

Result: 27 subsampled frames from 100 originals
```

The algorithm is efficient — it walks a single cursor forward through the source frames, never going backwards. For each target timestamp, it advances the cursor until it finds the closest source frame. This is O(source_frames + target_frames), not O(source * target).

Let's compare how different datasets get subsampled:

- **DROID** (15 fps → 4 fps): 100 source frames → 27 target frames. Moderate reduction.
- **ALOHA** (50 fps → 4 fps): 100 source frames → 8 target frames. Aggressive reduction. At 50 fps, consecutive frames are nearly identical (the robot barely moves in 20ms), so throwing away 92% of frames loses almost nothing.
- **Bridge** (5 fps → 4 fps): 100 source frames → 80 target frames. Gentle reduction. Already close to the target rate.

### Stage 3: Sliding window — _clip_windows()

V-JEPA 2 needs exactly 64 frames per clip. Episodes are rarely exactly 64 frames after subsampling. So we cut them into 64-frame windows with a sliding window.

The window slides forward by `clip_stride` (32) frames each step. This means consecutive clips overlap by 32 frames (50%). Let's walk through two scenarios:

**Short episode (27 subsampled frames from our DROID example):**

```
We need 64 frames but only have 27.

Window 1: Take frames 0-26 (27 frames).
           We're short by 37 frames.
           Pad by repeating frame 26 thirty-seven times.
           Now we have 64 frames. original_count = 27.
           This is the only window — we stop after padding.

Total: 1 clip
```

**Long episode (say 400 source frames → 107 subsampled frames):**

```
Window 1: frames 0 through 63.  That's 64 frames. Full window, no padding.
           Advance by stride of 32.

Window 2: frames 32 through 95. That's 64 frames. Full window, no padding.
           Note: frames 32-63 appear in BOTH window 1 and window 2. That's the overlap.
           Advance by stride of 32.

Window 3: frames 64 through 106. That's only 43 frames.
           Pad by repeating frame 106 twenty-one times to reach 64.
           original_count = 43. Stop after padding.

Total: 3 clips
```

**Why overlap?** Consider frame 63 in the long episode above. Without overlap, it would be the very last frame of clip 1 — right at the edge, with no future context. With 50% overlap, that same frame also appears in the middle of clip 2, where it has context on both sides. Overlap means every frame (except near the very start and end of the episode) gets represented with good surrounding context in at least one clip.

**Why pad the last window?** Because V-JEPA 2 requires exactly 64 frames. We can't give it 43. We repeat the last real frame to fill the gap. This is better than zero-padding (which would introduce artificial black frames) or interpolation (which would hallucinate motion that didn't happen). The metadata records `num_original_frames = 43` so downstream code knows where padding starts and can ignore or downweight those positions.

**Why does padding always come last?** The code is structured so that if a window needs padding, it's always the final window. Once you emit a padded window, you return immediately. This makes sense — if you have fewer frames than `clip_length`, there's no room for another window. And if the previous window was full, padding only happens at the tail end.

### Stage 4: Preprocess — _preprocess_clip()

Each 64-frame window gets fed through V-JEPA 2's video processor:

```python
processed = self._video_processor(clip_frames, return_tensors="pt")
```

The video processor (from HuggingFace Transformers) does the standard computer vision preprocessing: resize each frame to 256x256 pixels, normalize pixel values to the range the model expects, and stack everything into a PyTorch tensor.

The output shape is `(1, 64, 3, 256, 256)` — that's `(batch, frames, channels, height, width)`. We always have batch size 1, so we squeeze it off to get `(64, 3, 256, 256)`.

One quirk: depending on the version of the HuggingFace Transformers library, the processor might return the tensor under the key `"pixel_values"` or `"pixel_values_videos"`. We normalize this — if `pixel_values` is missing, we look for `pixel_values_videos` and rename it.

If the input frames are already PyTorch tensors (as opposed to PIL images), we stack them with `torch.stack` before passing to the processor. This handles both PIL and tensor inputs.

### Stage 5: Attach metadata

Each clip gets a `ClipMetadata` object that records where it came from:

```python
ClipMetadata(
    episode_id="lerobot/droid_1.0.1_0",     # dataset + episode number
    dataset_name="droid_1.0.1",               # just the dataset part
    robot_type="franka",                      # what robot was used
    clip_start_frame=0,                       # first real frame number in this clip
    clip_end_frame=98,                        # last real frame number (before padding)
    timestamp_start=0.0,                      # seconds from episode start
    timestamp_end=6.533,                      # seconds from episode start
    language_instruction="pick up the mug",   # what the robot was told (or None)
    num_original_frames=27,                   # how many frames are real (rest is padding)
)
```

The timestamps are computed from frame numbers using the **source fps** (not target fps). `clip_start_frame` is the frame number of the first frame, `clip_end_frame` is the frame number of the last real (non-padded) frame. The timestamps are relative to the first frame of the episode: `(frame_number - first_frame_of_episode) / source_fps`.

This metadata is what makes search results useful. When a query returns a match, you can say "this match came from the DROID dataset, episode 0, a Franka arm, between timestamps 0.0 and 6.5 seconds, and the robot was told to pick up the mug."

---

## Putting It All Together: A Complete Walkthrough

Let's trace a single episode from the DROID dataset all the way through. Episode 0 has 100 frames.

```
1. CONFIG
   repo_id = "lerobot/droid_1.0.1"
   image_key = "observation.images.exterior_1_left"
   source_fps = 15, target_fps = 4
   clip_length = 64, clip_stride = 32

2. LOAD EPISODE
   Episode 0 spans rows 0-99 in the dataset.
   For each row, extract the image at sample["observation"]["images"]["exterior_1_left"].
   Also grab frame_index (0, 1, 2, ..., 99) and language_instruction ("pick up the mug").
   Result: 100 PIL images, frame numbers [0..99], instruction "pick up the mug"

3. SORT
   Frames are already in order. Sort is a no-op (O(n) verification by Timsort).

4. SUBSAMPLE (15 fps → 4 fps)
   100 frames spanning 6.6 seconds → 27 frames at 4 fps.
   Picked frames: 0, 4, 8, 11, 15, 19, 23, 26, 30, 34, 38, 41, 45, 49,
                   53, 56, 60, 64, 68, 71, 75, 79, 83, 86, 90, 94, 98

5. WINDOW (clip_length=64, clip_stride=32)
   27 frames < 64, so: one window of 27 frames, padded to 64 by repeating frame 98.
   original_count = 27.

6. PREPROCESS
   Feed 64 images to V-JEPA 2 video processor.
   Out comes pixel_values of shape (64, 3, 256, 256).

7. METADATA
   episode_id = "lerobot/droid_1.0.1_0"
   clip_start_frame = 0, clip_end_frame = 98
   timestamp_start = 0.0, timestamp_end = 98/15 = 6.533s
   language_instruction = "pick up the mug"
   num_original_frames = 27

8. YIELD
   Yield ({"pixel_values": tensor(64, 3, 256, 256)}, metadata).
   Move on to episode 1.
```

Now imagine episode 1 is much longer — 2000 frames:

```
2000 frames at 15 fps = 133 seconds.
At 4 fps → 533 subsampled frames.

Clip windows:
  Clip 1: frames 0-63     (64 real frames)
  Clip 2: frames 32-95    (64 real frames, overlaps clip 1 by 32)
  Clip 3: frames 64-127   (64 real frames)
  Clip 4: frames 96-159   (64 real frames)
  ...continuing...
  Clip 15: frames 448-511 (64 real frames)
  Clip 16: frames 480-532 (53 real frames, padded to 64)

Total: 16 clips from one long episode.
Each clip becomes a (64, 3, 256, 256) tensor with its own metadata.
```

---

## Glossary

**Episode**: One continuous recording of a robot doing a task. Like a single YouTube video.

**Sample/Frame**: One moment within an episode. One image, one set of sensor readings, one timestep.

**Subsampling**: Reducing the frame rate. Going from 15 fps to 4 fps by picking the nearest source frame for each target timestamp.

**Clip**: A fixed-length chunk of 64 consecutive (subsampled) frames. What V-JEPA 2 actually processes.

**Clip stride**: How far the sliding window moves between clips. At stride 32 with length 64, clips overlap by 50%.

**Padding**: Repeating the last real frame to fill a clip that has fewer than 64 frames. Always happens at the end of an episode.

**Episode index**: A precomputed lookup table mapping episode numbers to row ranges in the dataset. Enables random access instead of sequential scanning.

**V-JEPA 2**: Meta's video world model. It takes a 64-frame clip and produces 256 spatial patch tokens per frame, capturing the spatial structure of the scene. This is the model whose preprocessor we use here. The actual encoding (running the model forward) happens downstream, not in this ingestion code.

**Spatial patches**: V-JEPA 2 divides each frame into a 16x16 grid of patches (256 total). Each patch corresponds to a region of the image. This is what makes WorldIndex different from simple visual similarity — it preserves *where* things are in the frame, not just *what* things look like.

**LeRobot**: A library by HuggingFace for working with robot learning datasets. Provides `StreamingLeRobotDataset` which streams data from HuggingFace Hub without downloading the full dataset.
