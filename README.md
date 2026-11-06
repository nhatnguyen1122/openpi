# Reinforcement Learning for Vision-Language-Action Models


We evaluate the `pi05_libero` checkpoint on the four LIBERO suites and run a small ablation matrix. With the policy
frozen, we then steer it with RL: first a SAC actor that chooses the flow-matching noise and the replanning horizon
(negative result), then a critic that selects among π0.5's own sampled action chunks.

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

Results on LIBERO-10 task 8, initial states 0–9 (success out of 10). Unsteered π0.5: 8 (k = 10), 4 (k = 5).

| SAC snapshot (updates) | 2k | 4k | 6k | 8k | 10k |
|---|---|---|---|---|---|
| noise only (k = 10) | 7 | 7 | 8 | 6 | 7 |
| noise + k | 4 | 7 | 6 | 4 | 0 |

Steering never beats the unsteered policy: the noise actor gives no gain, and with k learned the actor drifts to
k = 5 and the critic diverges.

## Value-guided candidate selection

π0.5 stays frozen; a learned critic picks among its own samples (`examples/libero/steering/qselect.py`).
Every k = 10 steps:

1. draw N = 8 noises ε_i ~ N(0, I); π0.5 denoises them into chunks a_i in one batched call;
2. score each chunk with an ensemble of E = 5 critics Q_j(s, a_i), where s = [P e; q; t / T] (e: π0.5 prefix
   embedding, P: fixed random projection to 256 dims, q: robot state) and a_i: the 10 actions to execute;
3. execute the chunk with the highest mean_j Q_j − 0.5 std_j Q_j.

The critic is trained off-policy by TD (γ = 0.99) on randomized LIBERO resets with reward = success +
potential-based shaping from the task's goal predicates. Exploration (probability 0.2) picks a random candidate, which
is exactly π0.5, so the control is the same pipeline with `--mode random`.

```bash
# Train one critic per suite (all 10 tasks)
.venv/bin/python -m examples.libero.steering.qselect_train train --output-dir data/steering/qselA_libero10 \
    --ports 8000 --workers-per-server 6 --suites libero_10 --total-episodes 800 --warmup-episodes-per-task 20

# Evaluate on the fixed initial states with videos: greedy critic, or --mode random for unsteered π0.5
.venv/bin/python -m examples.libero.steering.qselect_train eval --port 8000 --suite libero_10 --task-id 8 \
    --checkpoint data/steering/qselA_libero10/critic_u00065060_final.pt --start-state 0 --num-states 50 \
    --results-path data/steering/qeval_full/qselA_libero10/libero_10_t8.jsonl --video-dir data/steering/qeval_full/videos
```

A server keeps at most 16 steering sessions, so use at most 16 concurrent clients per server.

Results: 10 tasks × 50 initial states per suite, final checkpoint, paired with the control on the same states and
noise (sign test). Training used 500 episodes per suite (800 for LIBERO-10), one seed.

| Suite | Selection | π0.5 (k = 10) | wins / losses | p |
|---|---|---|---|---|
| Spatial | **497/500** | 486/500 | 13 / 2 | 0.007 |
| Object | 496/500 | 495/500 | 5 / 4 | 1.0 |
| Goal | 487/500 | 495/500 | 5 / 13 | 0.10 |
| LIBERO-10 | **479/500** | 468/500 | 22 / 11 | 0.08 |
| All | 1959/2000 | 1944/2000 | 45 / 30 | 0.11 |

The gain on LIBERO-10 comes from task 8 (43/50 vs 34/50), the task with the most headroom. Ablations on LIBERO-10:

| Variant | Success | vs full (wins / losses, p) | Task 8 |
|---|---|---|---|
| full | 479 | – | 43 |
| critic without π0.5 embedding (robot state, time) | 465 | 13 / 27, 0.04 | 35 |
| no reward shaping | 466 | 13 / 26, 0.05 | 39 |
| no pessimism (β = 0) | 468 | 13 / 24, 0.10 | 36 |
| one critic (E = 1) | 472 | 14 / 21, 0.31 | 41 |
| N = 4 candidates | 475 | 14 / 18, 0.60 | 40 |
| N = 16 candidates | 471 | 11 / 19, 0.20 | 44 |
| π0.5, no selection | 468 | 11 / 22, 0.08 | 34 |

The π0.5 embedding and reward shaping are clearly needed; pessimism and the ensemble help a little; N hardly matters
(one seed per variant). Flags: `--no-use-embedding`,
`--shaping-scale 0`, `--ensemble 1 --pessimism 0` (train); `--pessimism 0`, `--num-candidates N` (eval).
