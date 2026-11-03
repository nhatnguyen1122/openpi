# Reinforcement Learning for Vision-Language-Action Models


We evaluates the `pi05_libero` checkpoint on the four LIBERO suites, runs a small ablation matrix.
We design an RL actor (SAC) that chooses the flow-matching noise and the replanning horizon, policy is frozen.

## Setup

Requirements: Linux, an NVIDIA GPU with at least 8 GB of memory per policy server, and uv.

```bash
# Model environment (Python 3.11)
GIT_LFS_SKIP_SMUDGE=1 uv sync

# LIBERO simulator in the same environment
uv pip install robosuite==1.4.1 bddl==1.0.1 gym==0.25.2 easydict==1.9 future==0.18.2 -e third_party/libero

# LIBERO path config (no demonstration datasets needed for evaluation)
mkdir -p data/libero/config
L=$PWD/third_party/libero/libero/libero
cat > data/libero/config/config.yaml <<EOF
assets: $L/assets
bddl_files: $L/bddl_files
benchmark_root: $L
datasets: $PWD/data/libero/datasets
init_states: $L/init_files
EOF

# Checkpoint (about 12 GB) to checkpoints/pi05_libero
.venv/bin/python examples/libero/download_checkpoint.py
```

Client environment, set in every shell that runs LIBERO:

```bash
export PYTHONPATH=$PWD:$PWD/third_party/libero LIBERO_CONFIG_PATH=$PWD/data/libero/config
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl JAX_PLATFORMS=cpu TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

<!-- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` lets torch 2.7 load LIBERO's initial-state files. For EGL rendering on GPU `g`,
`CUDA_VISIBLE_DEVICES` must include `g` and `MUJOCO_EGL_DEVICE_ID=g`. -->

## Evaluate π0.5

Start a policy server, then run the client:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/serve_policy.py --port 8000 \
    policy:checkpoint --policy.config pi05_libero --policy.dir checkpoints/pi05_libero

.venv/bin/python examples/libero/main.py --args.port 8000 --args.task-suite-name libero_10 \
    --args.num-trials-per-task 50 --args.results-path data/libero/eval/libero_10.jsonl \
    --args.video-out-path data/libero/eval/videos
```

<!-- Each of the 50 trials of a task uses one of LIBERO's fixed initial states. Useful options: `--args.task-ids 8`,
`--args.start-episode-idx 10 --args.num-trials-per-task 40` (extend states 0–9 to 0–49 in the same results file),
`--args.replan-steps {N|full}`, `--args.no-save-video`. Results are appended, one JSON record per episode, so
use a fresh path for each run. -->

## Ablations

`examples/libero/ablations.py` starts one server per GPU and several concurrent clients. It runs 13 conditions, each with 10 tasks × initial states 0–9:

- instruction: empty, paraphrase, wrong task (LIBERO-Goal)
- flow steps: 5, 2, 1 (LIBERO-Goal)
- camera blackout: wrist, third-person (LIBERO-Object)
- replanning every 1, 5, or 10 (full chunk) actions (LIBERO-10)

<!-- The noise seed of every request is derived from (seed, suite, task, episode, call), so conditions are paired. -->

```bash
.venv/bin/python examples/libero/ablations.py --output-dir data/libero/ablations_run1          # print the plan
.venv/bin/python examples/libero/ablations.py --output-dir data/libero/ablations_run1 \
    --gpus 1 2 --ports 8001 8002 --clients-per-server 2 --execute
```

## Noise and horizon steering


```bash
# Train on randomized resets (add --horizons 10 for noise only, --no-learn-noise for k only)
.venv/bin/python -m examples.libero.steering.train --output-dir data/steering/t8 --ports 8000 \
    --workers-per-server 2 --suites libero_10 --task-ids 8 --total-episodes 300 \
    --target-entropy-continuous 90 --target-entropy-discrete 1.5

# Evaluate a snapshot (actor_u<updates>.pt, saved every 1000 updates) on the fixed initial states
.venv/bin/python examples/libero/main.py --args.port 8000 --args.task-suite-name libero_10 --args.task-ids 8 \
    --args.num-trials-per-task 10 --args.steering-checkpoint data/steering/t8/actor_u00006000.pt --args.no-save-video
```
