"""LIBERO wrapper for steering: one decision = embed -> (noise, k) -> act -> execute k actions."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import time
from typing import Any

import numpy as np

from . import common

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
MAX_RESET_ATTEMPTS = 20  # each attempt already retries the object placement 5000 times inside LIBERO


class ResetError(RuntimeError):
    """LIBERO could not place the objects; the caller should recreate the environment."""


@dataclasses.dataclass
class MacroResult:
    reward: float  # sum_i gamma^i r_i over the executed steps
    discount: float  # gamma^(steps executed)
    steps: int
    success: bool
    truncated: bool


class SteeringEnv:
    """Wraps one LIBERO task.

    Args:
        env: a LIBERO `OffScreenRenderEnv` (or a fake with reset/step/set_init_state and `.env` for predicates).
        prompt: task instruction sent to the VLA.
        max_steps: policy-step limit of the suite (`main.MAX_STEPS`).
        make_element: builds the server request from a raw observation and the prompt.
        robot_state: extracts the 8-dim robot state from a raw observation.
    """

    def __init__(
        self,
        env: Any,
        prompt: str,
        max_steps: int,
        client: common.SteeringClient,
        spec: common.SteeringSpec,
        *,
        make_element: Callable[[dict, str], dict],
        robot_state: Callable[[dict], np.ndarray],
        num_steps_wait: int = 10,
    ):
        self.env = env
        self.prompt = prompt
        self.max_steps = max_steps
        self.client = client
        self.spec = spec
        self._make_element = make_element
        self._robot_state = robot_state
        self.num_steps_wait = num_steps_wait
        self._goal = list(env.env.parsed_problem["goal_state"])
        if not self._goal:
            raise ValueError("Task has no goal predicates")
        self.obs: dict | None = None
        self.t = 0
        self._n_prev = 0
        self.vla_seconds = 0.0
        self.record = False  # when True, `frames` holds the agentview images of the current episode
        self.frames: list[np.ndarray] = []

    # ----------------------------------------------------------------------------------------------------------------
    def satisfied_predicates(self) -> int:
        """Number of goal predicates currently true (LIBERO's `_check_success` is their conjunction)."""
        inner = self.env.env
        return sum(bool(inner._eval_predicate(predicate)) for predicate in self._goal)  # noqa: SLF001

    def potential(self, n: int) -> float:
        return self.spec.shaping_scale * n / len(self._goal)

    def reset(self, init_state: np.ndarray | None = None) -> None:
        """Resets the task. Pass `init_state` only for evaluation; training uses LIBERO's randomized placement."""
        obs = self._reset_raw()
        if init_state is not None:
            obs = self.env.set_init_state(init_state)
        for _ in range(self.num_steps_wait):  # let objects settle, as in main.py
            obs, _, _, _ = self.env.step(LIBERO_DUMMY_ACTION)
        self.obs = obs
        self.t = 0
        self._n_prev = self.satisfied_predicates()
        self.vla_seconds = 0.0
        self.frames = [self._frame()] if self.record else []

    def _reset_raw(self) -> dict:
        # LIBERO's ControlEnv.reset retries a failed placement forever, which can hang a worker indefinitely.
        inner = self.env.env
        if not hasattr(inner, "reset"):  # test fakes
            return self.env.reset()
        from robosuite.utils.errors import RandomizationError

        for _ in range(MAX_RESET_ATTEMPTS):
            try:
                return inner.reset()
            except RandomizationError:
                continue
        raise ResetError(f"Could not place the objects in {MAX_RESET_ATTEMPTS} attempts")

    def _frame(self) -> np.ndarray:
        return np.ascontiguousarray(self.obs["agentview_image"][::-1, ::-1])  # upright, as in main.py

    def embed(self) -> np.ndarray:
        """Embeds the current observation; the server keeps its prefix for the next `step_macro`."""
        start = time.monotonic()
        embedding = self.client.embed(self._make_element(self.obs, self.prompt))
        self.vla_seconds += time.monotonic() - start
        return embedding

    def robot_state(self) -> np.ndarray:
        return self._robot_state(self.obs)

    def step_macro(self, noise: np.ndarray, k: int) -> MacroResult:
        """Samples a chunk from `noise` for the last embedded observation and executes its first k actions."""
        start = time.monotonic()
        actions = self.client.act(noise, self.spec.num_flow_steps)
        self.vla_seconds += time.monotonic() - start
        if len(actions) < k:
            raise ValueError(f"Policy returned {len(actions)} actions but k={k}")
        return self.execute(actions[:k])

    def sample_chunks(self, noise: np.ndarray) -> np.ndarray:
        """Candidate chunks [N, H, action_dim] from noise [N, H, D] for the last embedded observation."""
        start = time.monotonic()
        chunks = self.client.act_batch(noise, self.spec.num_flow_steps)
        self.vla_seconds += time.monotonic() - start
        return chunks

    def execute(self, actions: np.ndarray) -> MacroResult:
        """Executes actions until they run out, the task succeeds or the step limit is reached."""
        gamma = self.spec.gamma
        total, discount, success = 0.0, 1.0, False
        steps = 0
        for action in actions:
            self.obs, _, done, _ = self.env.step(np.asarray(action).tolist())
            self.t += 1
            steps += 1
            if self.record:
                self.frames.append(self._frame())
            success = bool(done)  # LIBERO returns done=True exactly when the goal holds.
            n = self.satisfied_predicates()
            reward = float(success) + gamma * self.potential(n) - self.potential(self._n_prev)
            total += discount * reward
            discount *= gamma
            self._n_prev = n
            if success or self.t >= self.max_steps:
                break
        return MacroResult(total, discount, steps, success, not success and self.t >= self.max_steps)


def run_episode(
    env: SteeringEnv,
    choose: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, int]],
    *,
    init_state: np.ndarray | None = None,
    state_fn: Callable[[np.ndarray, np.ndarray, int, int], np.ndarray],
    emit: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Runs one episode and emits one transition per decision.

    Args:
        choose: s -> (model noise [H, D], u [noise_dim], k_index).
        state_fn: (embedding, robot_state, t, max_steps) -> s.
        emit: receives each transition dict (s, u, k_index, reward, discount, done, s2).
    """
    start = time.monotonic()
    env.reset(init_state)
    s = state_fn(env.embed(), env.robot_state(), env.t, env.max_steps)
    horizons = env.spec.horizons
    episode_return, decisions, chosen_k = 0.0, 0, []
    while True:
        noise, u, k_index = choose(s)
        result = env.step_macro(noise, horizons[k_index])
        decisions += 1
        chosen_k.append(horizons[k_index])
        episode_return += result.reward
        # A successful step is terminal (no bootstrap); a time limit is a truncation and still bootstraps.
        s2 = s if result.success else state_fn(env.embed(), env.robot_state(), env.t, env.max_steps)
        if emit is not None:
            emit(
                {
                    "s": s,
                    "u": np.asarray(u, np.float32),
                    "k_index": int(k_index),
                    "reward": result.reward,
                    "discount": result.discount,
                    "done": float(result.success),
                    "s2": s2,
                }
            )
        if result.success or result.truncated:
            break
        s = s2
    return {
        "success": bool(result.success),
        "policy_steps": env.t,
        "decisions": decisions,
        "mean_k": float(np.mean(chosen_k)),
        "shaped_return": episode_return,
        "final_predicates": env.satisfied_predicates(),
        "num_predicates": len(env._goal),  # noqa: SLF001
        "vla_seconds": env.vla_seconds,
        "wall_seconds": time.monotonic() - start,
    }
