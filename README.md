# WorldIndex

ColBERT-style late interaction Token level spatiotemporal retrieval using VJEPA v2. 

## What it is & Why this is useful

Imagine you are a robotics researcher. You have a robot arm in your lab and you want to teach it to pick up a mug from a cluttered table. Normally, you would spend days collecting hundreds of demonstrations yourself.

With WorldIndex, you take a single photo of your workspace and upload it. WorldIndex searches through 76,000+ real robot demonstrations collected across dozens of labs worldwide (thanks to open source datasets) and multiple robot types.

It can be thought of as a better Google Image Search. It does not just find "visually similar" scenes. It understands spatial structure (world models!). It knows that the mug in the top-right corner of your photo corresponds to an object in the top-right of a demonstration even if the objects look completely different. It knows that a robot approaching from the left in one demonstration is doing the same thing as a robot approaching from below in another. This is because WorldIndex preserves the 256 spatial patches that V-JEPA 2 world model produces for each video frame, not a dumbed-down single-number summary.

## What you can do (theoretically, atleast):

* "Find scenes like this" — Upload a photo. Get the 10 most spatially similar robot demonstrations, ranked by how well the spatial layout matches. You see thumbnails, timestamps, which robot, which lab, what the robot was doing.
* "Find trajectories like this" — Upload a short video or pick an existing episode. Find demonstrations where the robot follows a similar motion path, even if one robot moves faster or slower than the other (the system handles speed variation via Dynamic Time Warping).
* "Find where this happens, then that happens" — Upload two images (a before state and after state). Find demonstrations that contain both states in sequence. These are transition queries.
* "What is happening in this region?" — Draw a box on an image. Search only those spatial patches. Find demonstrations with similar activity in that specific region of the workspace.


Note: I say "theoretically" here because I dont know if I will implement the frontend part. Maybe I'll vibe code the UI. I do plan to include the API endpoints, for sure!


## Tests

### All tests
* Unit: `poetry run pytest tests/ -v`
* Unit + End to End (Included): `WORLDINDEX_RUN_REAL_INGESTION=1 WORLDINDEX_REAL_INGESTION_DATASET=aloha poetry run pytest tests/ -v`

### Ingestion
* Unit: `poetry run pytest tests/ingestion -v`
* Unit + End to End (Included): `WORLDINDEX_RUN_REAL_INGESTION=1 WORLDINDEX_REAL_INGESTION_DATASET=aloha poetry run pytest tests/ingestion -v`

## Explanations

Explanation about how each component works can be found within `docs/`.