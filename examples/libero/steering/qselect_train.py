"""Train and evaluate value-guided candidate selection (see qselect.py).

    # train: one server per port, W workers per server; randomized LIBERO resets
    python -m examples.libero.steering.qselect_train train --output-dir data/steering/qsel_t8 --ports 8100 8101 \
        --suites libero_10 --task-ids 8
    # evaluate a snapshot on the fixed benchmark initial states (greedy), or the base policy (--mode random)
    python -m examples.libero.steering.qselect_train eval --checkpoint data/steering/qsel_t8/critic_u00004000.pt \
        --port 8100 --suite libero_10 --task-id 8 --start-state 0 --num-states 10 --results-path out.jsonl
"""

from __future__ import annotations

import collections
import dataclasses
import faulthandler
import json
import logging
import multiprocessing as mp
import os
import pathlib
import queue as queue_lib
import sys
import time

import numpy as np
import tyro

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
    __package__ = "examples.libero.steering"

from . import common


@dataclasses.dataclass
class TrainArgs:
    output_dir: str
    host: str = "127.0.0.1"
    ports: list[int] = dataclasses.field(default_factory=lambda: [8000])
    workers_per_server: int = 2
    suites: list[str] = dataclasses.field(default_factory=lambda: ["libero_10"])
    task_ids: list[int] | None = None
    total_episodes: int = 300
    warmup_episodes_per_task: int = 20  # Base policy (exploratory choices only) before updates start.
    updates_per_transition: int = 4
    buffer_size: int = 200_000
    save_every_updates: int = 250
    snapshot_every_updates: int = 2000
    log_every_updates: int = 200
    episode_timeout: int = 900  # seconds; a stuck episode is logged with its stack, abandoned and counted as failed
    video_every: int = 10  # each worker records every n-th of its episodes to <output_dir>/videos (0: none)

    # Selection.
    num_candidates: int = 8
    segment: int = 10
    allow_continue: bool = False
    ensemble: int = 5
    pessimism: float = 0.5
    explore_eps: float = 0.2
    continue_prob: float = 0.5
    use_embedding: bool = True  # False: ablation without the VLA prefix embedding in the critic state
    hidden_dims: list[int] = dataclasses.field(default_factory=lambda: [256, 256])
    lr: float = 3e-4
    batch_size: int = 256
    tau: float = 0.005

    # Shared with the SAC steering spec.
    projection_dim: int = 256
    gamma: float = 0.99
    shaping_scale: float = 1.0
    num_flow_steps: int | None = None

    seed: int = 0
    resize_size: int = 224
    num_steps_wait: int = 10
    render_gpu_device_id: int = -1
    max_cached_envs_per_worker: int = 4


@dataclasses.dataclass
class EvalArgs:
    checkpoint: str  # critic*.pt (projection.npy must be next to it)
    results_path: str
    host: str = "127.0.0.1"
    port: int = 8000
    suite: str = "libero_10"
    task_id: int = 8
    start_state: int = 0
    num_states: int = 10
    mode: str = "greedy"  # greedy | random (base policy with the same candidate machinery)
    seed: int = 7
    num_candidates: int | None = None  # override the trained N (the critic scores executed actions, not N)
    pessimism: float | None = None  # override beta in mean - beta * std
    allow_continue: bool | None = None  # override; False never offers the continue option (ablation of B)
    resize_size: int = 224
    num_steps_wait: int = 10
    render_gpu_device_id: int = -1
    max_cached_envs_per_worker: int = 1
    video_dir: str | None = None  # records every episode, named like main.py's videos


def _config(args: TrainArgs):
    from . import qselect

    return qselect.QSelectConfig(
        num_candidates=args.num_candidates,
        segment=args.segment,
        allow_continue=args.allow_continue,
        ensemble=args.ensemble,
        hidden_dims=tuple(args.hidden_dims),
        lr=args.lr,
        batch_size=args.batch_size,
        tau=args.tau,
        pessimism=args.pessimism,
        q_min=-1.5 * max(args.shaping_scale, 1.0),
        q_max=1.0 + 1.5 * max(args.shaping_scale, 1.0),
        explore_eps=args.explore_eps,
        continue_prob=args.continue_prob,
        use_embedding=args.use_embedding,
    )


def _spec(args: TrainArgs, metadata: dict, embedding_dim: int) -> common.SteeringSpec:
    steering = metadata["steering"]
    return common.SteeringSpec(
        action_horizon=int(steering["action_horizon"]),
        model_action_dim=int(steering["action_dim"]),
        horizons=(args.segment,),
        embedding_dim=embedding_dim,
        projection_dim=args.projection_dim,
        projection_seed=args.seed,
        gamma=args.gamma,
        shaping_scale=args.shaping_scale,
        num_flow_steps=args.num_flow_steps,
    )


def worker_main(worker_id, generation, args_dict, port, run_dir, tasks, messages, episode_counter, stop_event) -> None:
    import torch

    from . import env as steering_env
    from . import qselect
    from .train import LiberoTaskFactory

    logging.basicConfig(level=logging.INFO, format=f"[worker {worker_id}] %(asctime)s %(message)s", force=True)
    torch.set_num_threads(1)
    args = TrainArgs(**args_dict)
    # A replacement worker (generation > 0) must not replay its predecessor's initial placements.
    worker_seed = args.seed * 1000 + worker_id + 1 + 100_000 * generation
    np.random.seed(worker_seed)  # LIBERO's placement sampler uses the global numpy RNG.
    rng = np.random.default_rng(worker_seed)
    run_dir = pathlib.Path(run_dir)
    critic_path = run_dir / "critic.pt"
    warmup_total = args.warmup_episodes_per_task * len(tasks)
    try:
        client = common.SteeringClient(args.host, port)
        selector = qselect.Selector.load(critic_path)
        factory = LiberoTaskFactory(args, client, selector.spec)
        loaded = None
        local_episodes = 0
        try:
            while True:
                with episode_counter.get_lock():
                    index = episode_counter.value
                    if index >= args.total_episodes or stop_event.is_set():
                        break
                    episode_counter.value += 1
                warmup = index < warmup_total
                if not warmup and critic_path.stat().st_mtime_ns != loaded:
                    loaded = critic_path.stat().st_mtime_ns
                    selector.load_weights(torch.load(critic_path, map_location="cpu", weights_only=False))
                suite, task_id = tasks[index % len(tasks)]
                record = args.video_every > 0 and local_episodes % args.video_every == 0
                local_episodes += 1
                messages.put(("start", (worker_id, index, warmup, suite, task_id)))
                # The learner kills a worker whose episode exceeds episode_timeout (some episodes spin forever
                # inside native LIBERO/MuJoCo code); dump the stack shortly before so the log shows where.
                faulthandler.dump_traceback_later(max(1, args.episode_timeout - 60))
                for attempt in range(3):
                    env = factory.get(suite, task_id)
                    env.record = record
                    try:
                        summary = qselect.run_episode(
                            env,
                            selector,
                            rng,
                            mode="random" if warmup else "train",
                            emit=lambda t: messages.put(("transition", t)),
                        )
                        break
                    except steering_env.ResetError:  # raised before any transition is emitted
                        logging.warning("Reset failed for %s/%d (attempt %d); rebuilding", suite, task_id, attempt)
                        factory.drop(suite, task_id)
                else:
                    raise RuntimeError(f"Could not reset {suite}/{task_id}")
                faulthandler.cancel_dump_traceback_later()
                summary.update(
                    episode=index,
                    worker=worker_id,
                    suite=suite,
                    task_id=task_id,
                    warmup=warmup,
                    critic_version=None if warmup else selector.version,
                )
                if env.record:
                    name = common.video_name(suite, env.prompt, index, success=summary["success"])
                    common.save_video(run_dir / "videos" / f"w{worker_id}_{name}", env.frames)
                messages.put(("episode", summary))
        finally:
            factory.close()
    except Exception as error:
        logging.exception("Worker failed")
        messages.put(("error", f"worker {worker_id}: {error!r}"))
        raise
    finally:
        messages.put(("worker_done", worker_id))


class Learner:
    def __init__(self, args: TrainArgs, spec: common.SteeringSpec, run_dir: pathlib.Path, warmup_total: int):
        from . import qselect

        self.qselect = qselect
        self.args, self.run_dir, self.warmup_total = args, run_dir, warmup_total
        config = _config(args)
        self.agent = qselect.QSelectAgent(spec, config, seed=args.seed)
        self.buffer = qselect.ReplayBuffer(spec, config, args.buffer_size, seed=args.seed)
        self.warmup_finished = 0
        self.episodes = 0
        self.pending = 0
        self.recent = collections.deque(maxlen=40)
        self.start = time.monotonic()
        self.metrics = (run_dir / "metrics.jsonl").open("a")
        self.save()

    def log(self, record: dict) -> None:
        record = {"time": time.monotonic() - self.start, "updates": self.agent.num_updates, **record}
        self.metrics.write(json.dumps(record) + "\n")
        self.metrics.flush()

    def save(self, name: str = "critic.pt") -> None:
        self.qselect.save_atomic(self.agent.payload(), self.run_dir / name)

    def on_transition(self, transition: dict) -> None:
        self.buffer.add(transition)
        if self.warmup_finished < self.warmup_total or len(self.buffer) < self.args.batch_size:
            return
        self.pending += self.args.updates_per_transition
        while self.pending > 0:
            self.pending -= 1
            info = self.agent.update(self.buffer.sample(self.args.batch_size))
            n = self.agent.num_updates
            if n % self.args.log_every_updates == 0:
                self.log({"kind": "update", "buffer": len(self.buffer), **info})
            if n % self.args.save_every_updates == 0:
                self.save()
            if n % self.args.snapshot_every_updates == 0:
                self.save(f"critic_u{n:08d}.pt")

    def on_episode(self, summary: dict) -> None:
        self.episodes += 1
        self.warmup_finished += bool(summary["warmup"])
        if not summary["warmup"]:
            self.recent.append(summary["success"])
        self.log({"kind": "episode", **summary})
        logging.info(
            "episode %d %s success=%s steps=%d continue=%.2f recent=%s updates=%d",
            summary["episode"],
            "warmup" if summary["warmup"] else f"critic v{summary['critic_version']}",
            summary["success"],
            summary["policy_steps"],
            summary["continue_fraction"],
            f"{np.mean(self.recent):.2f}" if self.recent else "-",
            self.agent.num_updates,
        )

    def close(self) -> None:
        self.save()
        self.save(f"critic_u{self.agent.num_updates:08d}_final.pt")
        self.metrics.close()


def train(args: TrainArgs) -> None:
    import torch

    from examples.libero import main as libero_main

    from .train import LIBERO_ENV_RESOLUTION
    from .train import training_tasks

    logging.basicConfig(level=logging.INFO, format="[learner] %(asctime)s %(message)s", force=True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    if args.allow_continue and 2 * args.segment > 10:
        raise ValueError("--allow-continue needs segment <= action_horizon / 2")
    run_dir = pathlib.Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "critic.pt").exists():
        raise FileExistsError(f"{run_dir} already contains a run; use a new --output-dir")
    tasks = training_tasks(args.suites, args.task_ids)
    warmup_total = min(args.warmup_episodes_per_task * len(tasks), args.total_episodes)

    probe = common.SteeringClient(args.host, args.ports[0])
    for port in args.ports[1:]:
        other = common.SteeringClient(args.host, port).server_metadata
        for key in ("policy_config", "checkpoint_dir", "steering"):
            if other.get(key) != probe.server_metadata.get(key):
                raise ValueError(f"Server on port {port} differs in {key}")
    blank = np.zeros((LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, 3), np.uint8)
    dummy = {
        "agentview_image": blank,
        "robot0_eye_in_hand_image": blank,
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
    }
    embedding_dim = probe.embed(libero_main.make_policy_element(dummy, "probe", args.resize_size)).shape[0]
    spec = _spec(args, probe.server_metadata, embedding_dim)
    np.save(run_dir / "projection.npy", common.make_projection(spec))
    common.write_json(
        run_dir / "config.json",
        {"args": dataclasses.asdict(args), "spec": spec.to_dict(), "config": _config(args).to_dict(), "tasks": tasks},
    )
    learner = Learner(args, spec, run_dir, warmup_total)

    context = mp.get_context("spawn")
    messages = context.Queue(maxsize=10_000)
    counter = context.Value("i", 0)
    stop = context.Event()

    def spawn(worker_id: int, generation: int):
        port = args.ports[worker_id // args.workers_per_server]
        process = context.Process(
            target=worker_main,
            args=(worker_id, generation, dataclasses.asdict(args), port, str(run_dir), tasks, messages, counter, stop),
            daemon=True,
        )
        process.start()
        return process

    num_workers = len(args.ports) * args.workers_per_server
    workers = [spawn(worker_id, 0) for worker_id in range(num_workers)]
    generations = [0] * num_workers
    active: dict[int, tuple[float, dict]] = {}  # worker -> (start time, summary to log if it gets stuck)
    finished, last_check = 0, time.monotonic()
    try:
        while finished < num_workers:
            if time.monotonic() - last_check > 10:
                last_check = time.monotonic()
                for worker_id, (started, stuck) in list(active.items()):
                    if last_check - started < args.episode_timeout:
                        continue
                    logging.warning("Worker %d stuck on episode %d; replacing it", worker_id, stuck["episode"])
                    workers[worker_id].kill()
                    workers[worker_id].join(timeout=30)
                    del active[worker_id]
                    learner.on_episode(stuck)  # counted as a failure so that warmup accounting still completes
                    generations[worker_id] += 1
                    workers[worker_id] = spawn(worker_id, generations[worker_id])
            try:
                kind, payload = messages.get(timeout=5)
            except queue_lib.Empty:
                dead = [p for p in workers if p.exitcode not in (None, 0, -9)]
                if dead:
                    raise RuntimeError(f"Worker exited with code {dead[0].exitcode}") from None
                continue
            if kind == "start":
                worker_id, index, warmup, suite, task_id = payload
                stuck = {"success": False, "policy_steps": 0, "continue_fraction": 0.0, "timed_out": True}
                stuck.update(episode=index, worker=worker_id, suite=suite, task_id=task_id, warmup=warmup)
                stuck.update(critic_version=None if warmup else -1)
                active[worker_id] = (time.monotonic(), stuck)
            elif kind == "transition":
                learner.on_transition(payload)
            elif kind == "episode":
                active.pop(payload["worker"], None)
                learner.on_episode(payload)
            elif kind == "worker_done":
                finished += 1
            elif kind == "error":
                raise RuntimeError(payload)
    finally:
        stop.set()
        learner.close()
        for process in workers:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
    logging.info("Done: %d episodes, %d updates in %s", learner.episodes, learner.agent.num_updates, run_dir)


def evaluate(args: EvalArgs) -> None:
    from libero.libero import benchmark

    from . import qselect
    from .train import LiberoTaskFactory

    logging.basicConfig(level=logging.INFO, format="[eval] %(asctime)s %(message)s", force=True)
    if args.mode not in ("greedy", "random"):
        raise ValueError("mode must be greedy or random")
    selector = qselect.Selector.load(args.checkpoint)
    overrides = {
        "num_candidates": args.num_candidates,
        "pessimism": args.pessimism,
        "allow_continue": args.allow_continue,
    }
    selector.config = dataclasses.replace(selector.config, **{k: v for k, v in overrides.items() if v is not None})
    client = common.SteeringClient(args.host, args.port)
    factory = LiberoTaskFactory(args, client, selector.spec)
    env = factory.get(args.suite, args.task_id)
    env.record = args.video_dir is not None
    init_states = benchmark.get_benchmark_dict()[args.suite]().get_task_init_states(args.task_id)
    path = pathlib.Path(args.results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["episode_idx"] for line in path.open()} if path.exists() else set()
    try:
        for state in range(args.start_state, args.start_state + args.num_states):
            if state in done:  # resume after a killed run
                continue
            faulthandler.dump_traceback_later(600)  # diagnostics if an episode spins in native code
            rng = np.random.default_rng([args.seed, args.task_id, state])
            summary = qselect.run_episode(env, selector, rng, mode=args.mode, init_state=init_states[state])
            summary.update(
                suite=args.suite,
                task_id=args.task_id,
                episode_idx=state,
                mode=args.mode,
                checkpoint=args.checkpoint,
                critic_version=selector.version,
                num_candidates=selector.config.num_candidates,
                pessimism=selector.config.pessimism,
                allow_continue=selector.config.allow_continue,
            )
            if args.video_dir is not None:
                name = common.video_name(args.suite, env.prompt, state, success=summary["success"])
                common.save_video(pathlib.Path(args.video_dir) / name, env.frames)
            with path.open("a") as stream:
                stream.write(json.dumps(summary) + "\n")
            faulthandler.cancel_dump_traceback_later()
            logging.info("state %d success=%s steps=%d", state, summary["success"], summary["policy_steps"])
    finally:
        factory.close()


if __name__ == "__main__":
    parsed = tyro.extras.subcommand_cli_from_dict({"train": TrainArgs, "eval": EvalArgs})
    train(parsed) if isinstance(parsed, TrainArgs) else evaluate(parsed)
