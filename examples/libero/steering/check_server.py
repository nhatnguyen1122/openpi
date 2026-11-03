"""Pre-training checks for a steering-enabled policy server.

1. Determinism: the same observation and noise give identical actions (twice through `act`, and through the
   stateless `infer` op), and different noise gives different actions.
2. Equivalence with the original sampler: a regular request with `_openpi_inference = {seed: s}` samples its noise
   as `jax.random.normal(jax.random.key(s), (1, H, D))`; sending that same noise through the steering path must
   give the same actions. This is checked on real observations from several initial states.
3. Optional (--episodes-per-task > 0): episodes through the steering path with eps ~ N(0, I) and a fixed k on the
   benchmark's initial states.

    python -m examples.libero.steering.check_server --port 8000 --suite libero_10 --task-ids 8
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import sys

import numpy as np
import tyro

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
    __package__ = "examples.libero.steering"

from . import common
from . import env as steering_env


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    suite: str = "libero_spatial"
    task_ids: list[int] | None = None
    episodes_per_task: int = 0  # 0 runs only the determinism and equivalence checks
    equivalence_states: int = 3  # initial states per task used for the equivalence check
    equivalence_tolerance: float = 1e-3
    replan_steps: int = 5
    seed: int = 7  # env seed, as in main.py
    noise_seed: int = 0
    results_path: str | None = None
    render_gpu_device_id: int = -1


def check_determinism(client: common.SteeringClient, element: dict, rng: np.random.Generator) -> dict:
    shape = (client.action_horizon, client.model_action_dim)
    noise = rng.standard_normal(shape).astype(np.float32)
    client.embed(element)
    first = client.act(noise)
    second = client.act(noise)
    other = client.act(rng.standard_normal(shape).astype(np.float32))
    stateless = client._policy.infer({**element, common.STEERING_KEY: {"op": "infer", "noise": noise}})  # noqa: SLF001
    report = {
        "repeat_identical": bool(np.array_equal(first, second)),
        "stateless_max_abs_diff": float(np.max(np.abs(np.asarray(stateless["actions"]) - first))),
        "other_noise_max_abs_diff": float(np.max(np.abs(other - first))),
        "chunk_shape": list(first.shape),
    }
    if not report["repeat_identical"] or report["stateless_max_abs_diff"] > 1e-4:
        raise AssertionError(f"Steering is not deterministic: {report}")
    if report["other_noise_max_abs_diff"] == 0:
        raise AssertionError("Different noise produced identical actions")
    return report


def check_equivalence(client: common.SteeringClient, element: dict, seed: int, num_steps: int) -> float:
    """Max |actions(original sampler, seed) - actions(steering path, same noise)|."""
    import jax

    noise = np.asarray(
        jax.random.normal(jax.random.key(seed), (1, client.action_horizon, client.model_action_dim)), np.float32
    )[0]
    policy = client._policy  # noqa: SLF001
    original = policy.infer({**element, "_openpi_inference": {"num_steps": num_steps, "seed": seed}})
    client.embed(element)
    steered = client.act(noise, num_steps)
    return float(np.max(np.abs(np.asarray(original["actions"]) - steered)))


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    from libero.libero import benchmark

    from examples.libero import main as libero_main

    client = common.SteeringClient(args.host, args.port)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = list(range(suite.n_tasks)) if args.task_ids is None else args.task_ids
    spec = common.SteeringSpec(
        action_horizon=client.action_horizon,
        model_action_dim=client.model_action_dim,
        horizons=(args.replan_steps,),
        embedding_dim=1,
        projection_dim=1,
    )
    rng = np.random.default_rng(args.noise_seed)
    results = []
    for task_id in task_ids:
        task = suite.get_task(task_id)
        raw, prompt = libero_main._get_libero_env(  # noqa: SLF001
            task, libero_main.LIBERO_ENV_RESOLUTION, args.seed, args.render_gpu_device_id
        )
        wrapped = steering_env.SteeringEnv(
            raw,
            str(prompt),
            libero_main.MAX_STEPS[args.suite],
            client,
            spec,
            make_element=libero_main.make_policy_element,
            robot_state=libero_main.robot_state,
        )
        initial_states = suite.get_task_init_states(task_id)
        if task_id == task_ids[0]:
            wrapped.reset(initial_states[0])
            report = check_determinism(client, libero_main.make_policy_element(wrapped.obs, prompt), rng)
            logging.info("Determinism check passed: %s", report)
        num_steps = int(client.metadata["default_num_steps"])
        for state_idx in range(args.equivalence_states):
            wrapped.reset(initial_states[state_idx])
            element = libero_main.make_policy_element(wrapped.obs, prompt)
            diffs = [check_equivalence(client, element, seed, num_steps) for seed in (state_idx, 1000 + state_idx)]
            logging.info("Equivalence task %d state %d: max |diff| = %s", task_id, state_idx, diffs)
            if max(diffs) > args.equivalence_tolerance:
                raise AssertionError(f"Steering path differs from the original sampler: {diffs}")
        for episode_idx in range(args.episodes_per_task):
            summary = steering_env.run_episode(
                wrapped,
                lambda s: (
                    rng.standard_normal((spec.action_horizon, spec.model_action_dim)).astype(np.float32),
                    None,
                    0,
                ),
                init_state=initial_states[episode_idx],
                state_fn=lambda *_: None,
            )
            summary.update(suite=args.suite, task_id=task_id, episode_idx=episode_idx, replan_steps=args.replan_steps)
            results.append(summary)
            logging.info(
                "task %d episode %d success=%s steps=%d",
                task_id,
                episode_idx,
                *[summary[k] for k in ("success", "policy_steps")],
            )
        raw.close()
    if results:
        rate = np.mean([r["success"] for r in results])
        logging.info(
            "Steering path with N(0, I) noise, k=%d: success %.3f over %d episodes",
            args.replan_steps,
            rate,
            len(results),
        )
    if args.results_path:
        path = pathlib.Path(args.results_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in results))


if __name__ == "__main__":
    main(tyro.cli(Args))
