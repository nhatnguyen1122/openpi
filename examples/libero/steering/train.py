"""Train a noise-and-horizon steering actor for a frozen pi0.5 LIBERO policy with SAC.

Processes:
    policy servers (started separately, one per GPU): `scripts/serve_policy.py ... policy:checkpoint ...`
    rollout workers (this script, W per server): LIBERO + CPU copy of the actor; send transitions to the learner
    learner (this script's main process): replay buffer + SAC; writes `actor.pt` that workers reload

Example (one server on port 8000, two workers, one task):
    uv run python -m examples.libero.steering.train --output-dir data/steering/libero10_t0 \
        --ports 8000 --workers-per-server 2 --suites libero_10 --task-ids 0
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import multiprocessing as mp
import os
import pathlib
import queue as queue_lib
import sys
import time
from typing import Any, Literal

import numpy as np
import tyro

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
    __package__ = "examples.libero.steering"

from . import common

LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    output_dir: str
    host: str = "127.0.0.1"
    ports: list[int] = dataclasses.field(default_factory=lambda: [8000])
    workers_per_server: int = 2

    # Training tasks: every task id in `task_ids` (all tasks when omitted) of every suite.
    suites: list[str] = dataclasses.field(default_factory=lambda: ["libero_10"])
    task_ids: list[int] | None = None

    total_episodes: int = 2000  # Budget, including warmup episodes.
    warmup_episodes_per_task: int = 20  # Unsteered VLA (eps ~ N(0, I), uniform k) before updates start.
    updates_per_transition: int = 1
    buffer_size: int = 200_000
    save_every_updates: int = 500
    # Also keep numbered copies actor_u<updates>.pt so that an early, good actor survives later collapse.
    snapshot_every_updates: int = 1000
    log_every_updates: int = 100

    # Steering specification.
    control_dims: int = 7
    noise_bound: float = 2.5
    horizons: list[int] = dataclasses.field(default_factory=lambda: [5, 6, 7, 8, 9, 10])
    projection_dim: int = 256
    gamma: float = 0.99
    shaping_scale: float = 1.0
    num_flow_steps: int | None = None
    pad_noise: Literal["gaussian", "zero"] = "gaussian"
    # Ablations: --no-learn-noise learns only k (eps ~ N(0, I)); --horizons 10 learns only the noise.
    learn_noise: bool = True

    # SAC.
    hidden_dims: list[int] = dataclasses.field(default_factory=lambda: [256, 256])
    lr: float = 3e-4
    batch_size: int = 256
    tau: float = 0.005
    init_alpha_continuous: float = 0.1
    init_alpha_discrete: float = 0.1
    # Entropy targets act as a trust region around the unsteered policy. The initial entropy of u is ~112 nats for the
    # default 70 dims (log |K| = 1.79 for k); with the SAC default (-dim/2) alpha decays to ~0, the actor collapses
    # within ~2000 updates and chases noisy Q. None keeps the SAC defaults.
    target_entropy_continuous: float | None = None
    target_entropy_discrete: float | None = None
    # Abort when |mean Q| exceeds this: rewards are bounded by ~2 per episode, so larger values mean divergence.
    max_abs_q: float = 20.0
    backup_entropy: bool = False  # Entropy bonus in the critic target; off by default (diverges with a 70-dim u).
    learner_device: str = "cpu"

    seed: int = 0
    resize_size: int = 224
    num_steps_wait: int = 10
    render_gpu_device_id: int = -1
    max_cached_envs_per_worker: int = 4


# --------------------------------------------------------------------------------------------------------------------
# Setup helpers


def training_tasks(suites: list[str], task_ids: list[int] | None) -> list[tuple[str, int]]:
    from libero.libero import benchmark

    tasks = []
    for suite_name in suites:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        ids = range(suite.n_tasks) if task_ids is None else task_ids
        for task_id in ids:
            if not 0 <= task_id < suite.n_tasks:
                raise ValueError(f"{suite_name} has no task {task_id}")
            tasks.append((suite_name, int(task_id)))
    if not tasks:
        raise ValueError("No training tasks")
    return tasks


def make_spec(args: Args, server_metadata: dict, embedding_dim: int) -> common.SteeringSpec:
    steering = server_metadata["steering"]
    return common.SteeringSpec(
        action_horizon=int(steering["action_horizon"]),
        model_action_dim=int(steering["action_dim"]),
        control_dims=args.control_dims,
        noise_bound=args.noise_bound,
        horizons=tuple(args.horizons),
        embedding_dim=embedding_dim,
        projection_dim=args.projection_dim,
        projection_seed=args.seed,
        gamma=args.gamma,
        shaping_scale=args.shaping_scale,
        num_flow_steps=args.num_flow_steps,
        pad_noise=args.pad_noise,
        learn_noise=args.learn_noise,
    )


def sac_config(args: Args):
    from . import sac

    return sac.SACConfig(
        hidden_dims=tuple(args.hidden_dims),
        actor_lr=args.lr,
        critic_lr=args.lr,
        alpha_lr=args.lr,
        batch_size=args.batch_size,
        tau=args.tau,
        init_alpha_continuous=args.init_alpha_continuous,
        init_alpha_discrete=args.init_alpha_discrete,
        backup_entropy=args.backup_entropy,
        target_entropy_continuous=args.target_entropy_continuous,
        target_entropy_discrete=args.target_entropy_discrete,
    )


class LiberoTaskFactory:
    """Creates (and caches a few) LIBERO environments in a worker process."""

    def __init__(self, args: Args, client: common.SteeringClient, spec: common.SteeringSpec):
        self.args = args
        self.client = client
        self.spec = spec
        self._cache: collections.OrderedDict[tuple[str, int], Any] = collections.OrderedDict()

    def get(self, suite_name: str, task_id: int):
        from examples.libero import main as libero_main

        from . import env as steering_env

        key = (suite_name, task_id)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        from libero.libero import benchmark
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        task = benchmark.get_benchmark_dict()[suite_name]().get_task(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        # No env.seed(): LIBERO seeds the global numpy RNG, which each worker seeds once so that
        # recreating an evicted environment does not replay the same initial placements.
        raw = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
            render_gpu_device_id=self.args.render_gpu_device_id,
        )
        wrapped = steering_env.SteeringEnv(
            raw,
            str(task.language),
            libero_main.MAX_STEPS[suite_name],
            self.client,
            self.spec,
            make_element=lambda obs, prompt: libero_main.make_policy_element(obs, prompt, self.args.resize_size),
            robot_state=libero_main.robot_state,
            num_steps_wait=self.args.num_steps_wait,
        )
        self._cache[key] = wrapped
        while len(self._cache) > self.args.max_cached_envs_per_worker:
            _, evicted = self._cache.popitem(last=False)
            evicted.env.close()
        return wrapped

    def drop(self, suite_name: str, task_id: int) -> None:
        """Closes and forgets one environment so that the next `get` builds a fresh one."""
        evicted = self._cache.pop((suite_name, task_id), None)
        if evicted is not None:
            evicted.env.close()

    def close(self):
        for wrapped in self._cache.values():
            wrapped.env.close()
        self._cache.clear()


# --------------------------------------------------------------------------------------------------------------------
# Worker


def worker_main(
    worker_id: int,
    args_dict: dict,
    port: int,
    run_dir: str,
    tasks: list[tuple[str, int]],
    messages: mp.Queue,
    episode_counter,
    stop_event,
) -> None:
    import torch

    from . import sac

    logging.basicConfig(level=logging.INFO, format=f"[worker {worker_id}] %(asctime)s %(message)s", force=True)
    torch.set_num_threads(1)
    args = Args(**args_dict)
    worker_seed = args.seed * 1000 + worker_id + 1
    np.random.seed(worker_seed)  # LIBERO's placement sampler uses the global numpy RNG.
    rng = np.random.default_rng(worker_seed)
    generator = torch.Generator().manual_seed(worker_seed)
    run_dir = pathlib.Path(run_dir)
    warmup_total = args.warmup_episodes_per_task * len(tasks)

    try:
        client = common.SteeringClient(args.host, port)
        policy = sac.SteeringPolicy.load(run_dir)
        factory = LiberoTaskFactory(args, client, policy.spec)

        def claim_episode() -> int | None:
            with episode_counter.get_lock():
                index = episode_counter.value
                if index >= args.total_episodes or stop_event.is_set():
                    return None
                episode_counter.value += 1
                return index

        try:
            run_worker_episodes(
                worker_id,
                policy,
                factory.get,
                tasks,
                warmup_total,
                claim_episode,
                lambda kind, payload: messages.put((kind, payload)),
                run_dir / "actor.pt",
                rng,
                generator,
            )
        finally:
            factory.close()
    except Exception as error:
        logging.exception("Worker failed")
        messages.put(("error", f"worker {worker_id}: {error!r}"))
        raise
    finally:
        messages.put(("worker_done", worker_id))


def run_worker_episodes(
    worker_id: int,
    policy,
    get_env,
    tasks: list[tuple[str, int]],
    warmup_total: int,
    claim_episode,
    send,
    actor_path: pathlib.Path,
    rng: np.random.Generator,
    generator,
) -> None:
    """Runs episodes until `claim_episode` returns None; sends ("transition", t) and ("episode", summary)."""
    import torch

    from . import env as steering_env

    loaded_mtime = None
    while (index := claim_episode()) is not None:
        warmup = index < warmup_total
        # Reload the actor at episode boundaries when the learner has written a newer one.
        if not warmup and actor_path.stat().st_mtime_ns != loaded_mtime:
            loaded_mtime = actor_path.stat().st_mtime_ns
            policy.load_weights(torch.load(actor_path, map_location="cpu", weights_only=False))

        suite_name, task_id = tasks[index % len(tasks)]

        def choose(s, warmup=warmup):
            if warmup:  # Unsteered VLA: eps ~ N(0, I) and a uniformly random k.
                noise, u = policy.mapper.base_policy_noise(rng)
                return noise, u, int(rng.integers(policy.spec.num_horizons))
            return policy.decide(s, rng, deterministic=False, generator=generator)

        summary = steering_env.run_episode(
            get_env(suite_name, task_id),
            choose,
            state_fn=policy.state,
            emit=lambda transition: send("transition", transition),
        )
        summary.update(
            episode=index,
            worker=worker_id,
            suite=suite_name,
            task_id=task_id,
            warmup=warmup,
            actor_version=None if warmup else policy.version,
        )
        send("episode", summary)


# --------------------------------------------------------------------------------------------------------------------
# Learner


class Learner:
    """Consumes transitions and episode summaries; independent of processes so it can be tested directly."""

    def __init__(self, args: Args, spec: common.SteeringSpec, run_dir: pathlib.Path, warmup_total: int):
        from . import sac

        self.sac = sac
        self.args = args
        self.run_dir = run_dir
        self.agent = sac.SACAgent(spec, sac_config(args), args.learner_device)
        self.buffer = sac.ReplayBuffer(spec, args.buffer_size, seed=args.seed)
        self.warmup_total = warmup_total
        self.warmup_finished = 0
        self.episodes = 0
        self.pending_updates = 0
        self.recent = collections.deque(maxlen=50)
        self.metrics_file = (run_dir / "metrics.jsonl").open("a")
        self.start = time.monotonic()
        self.save_actor()

    def log(self, record: dict[str, Any]) -> None:
        record = {"time": time.monotonic() - self.start, "updates": self.agent.num_updates, **record}
        self.metrics_file.write(json.dumps(record) + "\n")
        self.metrics_file.flush()

    def save_actor(self) -> None:
        self.sac.save_atomic(self.agent.actor_payload(self.agent.num_updates), self.run_dir / "actor.pt")

    def save_full(self) -> None:
        self.sac.save_atomic(self.agent.state_dict(), self.run_dir / "agent.pt")

    @property
    def ready(self) -> bool:
        return self.warmup_finished >= self.warmup_total and len(self.buffer) >= self.args.batch_size

    def on_transition(self, transition: dict[str, Any]) -> None:
        self.buffer.add(transition)
        self.pending_updates += self.args.updates_per_transition
        if not self.ready:
            self.pending_updates = 0
            return
        while self.pending_updates > 0:
            self.pending_updates -= 1
            info = self.agent.update(self.buffer.sample(self.args.batch_size, self.agent.device))
            if abs(info["q_mean"]) > self.args.max_abs_q:
                self.log({"kind": "abort", "reason": "Q diverged", **info})
                raise RuntimeError(f"Q diverged: mean Q {info['q_mean']:.1f} after {self.agent.num_updates} updates")
            if self.agent.num_updates % self.args.log_every_updates == 0:
                self.log({"kind": "update", "buffer": len(self.buffer), **info})
            if self.agent.num_updates % self.args.save_every_updates == 0:
                self.save_actor()
            if self.args.snapshot_every_updates > 0 and self.agent.num_updates % self.args.snapshot_every_updates == 0:
                self.sac.save_atomic(
                    self.agent.actor_payload(self.agent.num_updates),
                    self.run_dir / f"actor_u{self.agent.num_updates:08d}.pt",
                )

    def on_episode(self, summary: dict[str, Any]) -> None:
        self.episodes += 1
        self.warmup_finished += bool(summary["warmup"])
        if not summary["warmup"]:
            self.recent.append(summary["success"])
        self.log({"kind": "episode", **summary})
        logging.info(
            "episode %d %s:%d %s success=%s steps=%d decisions=%d mean_k=%.2f recent_success=%s",
            summary["episode"],
            summary["suite"],
            summary["task_id"],
            "warmup" if summary["warmup"] else f"actor v{summary['actor_version']}",
            summary["success"],
            summary["policy_steps"],
            summary["decisions"],
            summary["mean_k"],
            f"{np.mean(self.recent):.2f}" if self.recent else "-",
        )

    def close(self) -> None:
        self.save_actor()
        self.save_full()
        self.metrics_file.close()


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="[learner] %(asctime)s %(message)s", force=True)
    import torch

    from . import sac

    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
    if args.workers_per_server < 1 or not args.ports:
        raise ValueError("Need at least one server port and one worker per server")
    if args.total_episodes < 1 or args.warmup_episodes_per_task < 0:
        raise ValueError("Invalid episode budget")
    run_dir = pathlib.Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "actor.pt").exists():
        raise FileExistsError(f"{run_dir} already contains a run; use a new --output-dir")

    tasks = training_tasks(args.suites, args.task_ids)
    warmup_total = min(args.warmup_episodes_per_task * len(tasks), args.total_episodes)

    # Probe the first server for the noise shape and the embedding width.
    from examples.libero import main as libero_main

    probe = common.SteeringClient(args.host, args.ports[0])
    for port in args.ports[1:]:
        other = common.SteeringClient(args.host, port).server_metadata
        for key in ("policy_config", "checkpoint_dir", "steering"):
            if other.get(key) != probe.server_metadata.get(key):
                raise ValueError(f"Server on port {port} differs in {key}")
    dummy_obs = {
        "agentview_image": np.zeros((LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, 3), np.uint8),
        "robot0_eye_in_hand_image": np.zeros((LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, 3), np.uint8),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
    }
    embedding_dim = probe.embed(libero_main.make_policy_element(dummy_obs, "probe", args.resize_size)).shape[0]
    spec = make_spec(args, probe.server_metadata, embedding_dim)
    np.save(run_dir / "projection.npy", common.make_projection(spec))
    common.write_json(
        run_dir / "config.json",
        {
            "args": dataclasses.asdict(args),
            "spec": spec.to_dict(),
            "tasks": tasks,
            "server": {k: probe.server_metadata.get(k) for k in ("policy_config", "checkpoint_dir", "steering")},
        },
    )
    logging.info("Spec: %s", spec)
    logging.info("%d training tasks, %d warmup episodes, budget %d", len(tasks), warmup_total, args.total_episodes)

    learner = Learner(args, spec, run_dir, warmup_total)
    sac.SteeringPolicy.load(run_dir)  # Fail early if the saved actor cannot be loaded.

    context = mp.get_context("spawn")
    messages = context.Queue(maxsize=10_000)
    episode_counter = context.Value("i", 0)
    stop_event = context.Event()
    workers = []
    for server_index, port in enumerate(args.ports):
        for slot in range(args.workers_per_server):
            worker_id = server_index * args.workers_per_server + slot
            process = context.Process(
                target=worker_main,
                args=(
                    worker_id,
                    dataclasses.asdict(args),
                    port,
                    str(run_dir),
                    tasks,
                    messages,
                    episode_counter,
                    stop_event,
                ),
                daemon=True,
            )
            process.start()
            workers.append(process)

    finished = 0
    try:
        while finished < len(workers):
            try:
                kind, payload = messages.get(timeout=5)
            except queue_lib.Empty:
                dead = [p for p in workers if p.exitcode not in (None, 0)]
                if dead:
                    raise RuntimeError(f"Worker exited with code {dead[0].exitcode}") from None
                continue
            if kind == "transition":
                learner.on_transition(payload)
            elif kind == "episode":
                learner.on_episode(payload)
            elif kind == "worker_done":
                finished += 1
            elif kind == "error":
                raise RuntimeError(payload)
    finally:
        stop_event.set()
        learner.close()
        for process in workers:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
    logging.info("Done: %d episodes, %d updates. Actor: %s", learner.episodes, learner.agent.num_updates, run_dir)


if __name__ == "__main__":
    main(tyro.cli(Args))
