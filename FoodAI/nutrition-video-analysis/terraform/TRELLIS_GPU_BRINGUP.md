# TRELLIS GPU Bring-Up

This is the Option A path for v2:

- keep the existing CPU worker untouched
- use a separate GPU path for TRELLIS only
- first milestone: generate `.glb` files for a few dish images

## Why this path

TRELLIS.2 is a very different runtime from the current CPU-oriented nutrition worker:

- CUDA-only
- custom compiled extensions like `o-voxel`, `CuMesh`, and `FlexGEMM`
- Hugging Face model downloads

Because of that, the safest first step is a dedicated GPU image and runner.

## Instance choice

Start with:

- `gpu_instance_type = "g6.2xlarge"`
- `use_gpu = true`
- `device_type = "cuda"`
- `gpu_min_capacity = 0`
- `gpu_max_capacity = 1`

That keeps the GPU off when unused and gives us a single bring-up lane.

## New files

- `docker/Dockerfile.trellis-gpu`
- `docker/trellis_glb_runner.py`

## First success criteria

We are done with milestone 1 when all of these are true:

1. A GPU ECS task starts on `g6.2xlarge`.
2. `torch.cuda.is_available()` is true inside the task.
3. TRELLIS loads.
4. A test dish image produces a `.glb`.
5. The output is written to the task output directory.

No volume-estimation changes are included in this step.

## Suggested first test images

Good initial candidates already present in this repo:

- `/Users/rakshith911/Downloads/food_detection/chicken-curry-recipe.jpg`
- `/Users/rakshith911/Downloads/food_detection/paratha.jpeg`
- `/Users/rakshith911/Downloads/food_detection/IMG_3717.jpg`

## Local smoke command

If you have a Linux GPU box with the image built, the runner entrypoint is:

```bash
python3 /app/trellis_glb_runner.py \
  --images /app/test-images/chicken-curry-recipe.jpg /app/test-images/paratha.jpeg \
  --output-dir /app/data/trellis_outputs
```

If you want to skip MP4 preview rendering for the very first smoke test:

```bash
python3 /app/trellis_glb_runner.py \
  --images /app/test-images/chicken-curry-recipe.jpg \
  --output-dir /app/data/trellis_outputs \
  --skip-video
```

## Bring-up sequence

1. Build the dedicated TRELLIS image with `Dockerfile.trellis-gpu`.
2. Push it to ECR with a separate tag such as `trellis-gpu-test`.
3. Set Terraform GPU variables to `g6.2xlarge` and scale-to-zero.
4. Start one GPU task manually for the first smoke test.
5. Confirm `.glb` generation before wiring any S3 or queue flow.

## Important note

`Dockerfile.trellis-gpu` currently clones the upstream TRELLIS.2 repo during build.
That is the fastest clean-room path for bring-up.

If your local `Desktop/TRELLIS.2` folder contains changes we need to preserve, the
next step after smoke success should be vendoring or syncing that exact TRELLIS
copy into the runtime instead of relying on upstream `main`.
