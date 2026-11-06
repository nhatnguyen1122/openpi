"""Shared pieces of noise-and-horizon steering: specification, noise mapping, features and the server client.

Nothing here imports torch or LIBERO, so both the learner and evaluation clients can use it.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
from typing import Any, Literal
import uuid

import numpy as np

STEERING_KEY = "_openpi_steering"


@dataclasses.dataclass(frozen=True)
class SteeringSpec:
    """Everything that must match between training, the replay buffer and evaluation."""

    action_horizon: int  # H: model action chunk length (10 for pi05_libero).
    model_action_dim: int  # D: padded model action dimension (32 for pi05_libero).
    control_dims: int = 7  # d: leading noise dimensions controlled by the actor (7 = real LIBERO actions).
    noise_bound: float = 2.5  # b: controlled noise is b * tanh(u).
    horizons: tuple[int, ...] = (5, 6, 7, 8, 9, 10)  # K: candidate numbers of actions executed per decision.
    embedding_dim: int = 2048  # Width of the pooled prefix embedding returned by the server.
    projection_dim: int = 256  # Rows of the fixed random projection applied to the embedding.
    projection_seed: int = 0
    robot_state_dim: int = 8
    gamma: float = 0.99  # Discount per environment step.
    shaping_scale: float = 1.0  # lambda in phi(n) = lambda * n / |G|; 0 disables shaping.
    num_flow_steps: int | None = None  # None keeps the server default.
    # Noise in dimensions the actor does not control. "gaussian" matches the distribution the VLA was trained with.
    pad_noise: Literal["gaussian", "zero"] = "gaussian"
    # False = ablation "learn k only": the noise stays eps ~ N(0, I) as in the unsteered VLA, the critic is Q(s, k).
    learn_noise: bool = True

    def __post_init__(self):
        object.__setattr__(self, "horizons", tuple(int(k) for k in self.horizons))
        if not 0 < self.control_dims <= self.model_action_dim:
            raise ValueError("control_dims must be in [1, model_action_dim]")
        if not self.horizons or len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must be nonempty and unique")
        if min(self.horizons) < 1 or max(self.horizons) > self.action_horizon:
            raise ValueError(f"horizons must be in [1, {self.action_horizon}]")
        if self.noise_bound <= 0 or not 0 < self.gamma <= 1:
            raise ValueError("noise_bound must be positive and gamma in (0, 1]")
        if self.num_flow_steps is not None and not 1 <= self.num_flow_steps <= 100:
            raise ValueError("num_flow_steps must be in [1, 100]")
        if self.pad_noise not in ("gaussian", "zero"):
            raise ValueError("pad_noise must be 'gaussian' or 'zero'")

    @property
    def noise_dim(self) -> int:
        return self.action_horizon * self.control_dims

    @property
    def num_horizons(self) -> int:
        return len(self.horizons)

    @property
    def state_dim(self) -> int:
        return self.projection_dim + self.robot_state_dim + 1

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SteeringSpec:
        return cls(**data)


def make_projection(spec: SteeringSpec) -> np.ndarray:
    """Fixed Gaussian projection [projection_dim, embedding_dim]; scaled so it roughly preserves feature variance."""
    rng = np.random.default_rng(spec.projection_seed)
    matrix = rng.standard_normal((spec.projection_dim, spec.embedding_dim)) / math.sqrt(spec.embedding_dim)
    return matrix.astype(np.float32)


def build_state(
    projection: np.ndarray, embedding: np.ndarray, robot_state: np.ndarray, t: int, max_steps: int
) -> np.ndarray:
    """s = [P e; q; t / T]."""
    embedding = np.asarray(embedding, np.float32)
    if embedding.shape != (projection.shape[1],):
        raise ValueError(f"Expected embedding of shape {(projection.shape[1],)}, got {embedding.shape}")
    return np.concatenate(
        [projection @ embedding, np.asarray(robot_state, np.float32), np.array([t / max_steps], np.float32)]
    ).astype(np.float32)


class NoiseMapper:
    """Maps actor outputs u (pre-tanh, [H * d]) to full model noise [H, D] and back."""

    def __init__(self, spec: SteeringSpec):
        self.spec = spec

    def controlled(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, np.float32).reshape(self.spec.action_horizon, self.spec.control_dims)
        return self.spec.noise_bound * np.tanh(u)

    def to_noise(self, u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        shape = (self.spec.action_horizon, self.spec.model_action_dim)
        if self.spec.pad_noise == "gaussian":
            noise = rng.standard_normal(shape).astype(np.float32)
        else:
            noise = np.zeros(shape, np.float32)
        noise[:, : self.spec.control_dims] = self.controlled(u)
        return noise

    def base_policy_noise(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Standard-normal noise (the unsteered VLA) and its u = atanh(eps / b) for the replay buffer.

        Controlled entries beyond 0.999 b (about 1% of them for b = 2.5) are clipped so that the noise sent to the
        VLA is exactly b * tanh(u), i.e. what the critic sees.
        """
        noise = rng.standard_normal((self.spec.action_horizon, self.spec.model_action_dim)).astype(np.float32)
        ratio = np.clip(noise[:, : self.spec.control_dims] / self.spec.noise_bound, -0.999, 0.999)
        u = np.arctanh(ratio).reshape(-1).astype(np.float32)
        noise[:, : self.spec.control_dims] = self.controlled(u)
        return noise, u


class SteeringClient:
    """Talks to a policy server started from this repository (see `Policy._infer_steering`).

    Each client has its own session, so several clients can share one server.
    """

    def __init__(self, host: str, port: int, *, policy: Any = None):
        if policy is None:
            from openpi_client import websocket_client_policy

            policy = websocket_client_policy.WebsocketClientPolicy(host, port)
        self._policy = policy
        self.server_metadata = policy.get_server_metadata()
        self.metadata = self.server_metadata.get("steering")
        if not self.metadata or self.metadata.get("key") != STEERING_KEY:
            raise ValueError("Policy server does not support steering; restart it with the updated openpi code")
        self.session = uuid.uuid4().hex

    @property
    def action_horizon(self) -> int:
        return int(self.metadata["action_horizon"])

    @property
    def model_action_dim(self) -> int:
        return int(self.metadata["action_dim"])

    def embed(self, element: dict) -> np.ndarray:
        """Computes the pooled prefix embedding and keeps the prefix for the next `act`."""
        response = self._policy.infer({**element, STEERING_KEY: {"op": "embed", "session": self.session}})
        self._check(response, "embed")
        return np.asarray(response["prefix_embedding"], np.float32)

    def act(self, noise: np.ndarray, num_steps: int | None = None) -> np.ndarray:
        """Integrates the flow from `noise` for the observation of the latest `embed`."""
        request = {"op": "act", "session": self.session, "noise": np.asarray(noise, np.float32)}
        if num_steps is not None:
            request["num_steps"] = int(num_steps)
        response = self._policy.infer({STEERING_KEY: request})
        self._check(response, "act")
        actions = np.asarray(response["actions"])
        if actions.ndim != 2 or len(actions) == 0 or not np.isfinite(actions).all():
            raise ValueError(f"Invalid action chunk from server: shape {actions.shape}")
        return actions

    def act_batch(self, noise: np.ndarray, num_steps: int | None = None) -> np.ndarray:
        """N candidate chunks [N, H, action_dim] for the latest `embed`, from noise [N, H, D] in one model call."""
        request = {"op": "act_batch", "session": self.session, "noise": np.asarray(noise, np.float32)}
        if num_steps is not None:
            request["num_steps"] = int(num_steps)
        response = self._policy.infer({STEERING_KEY: request})
        self._check(response, "act_batch")
        actions = np.asarray(response["actions"])
        if actions.ndim != 3 or len(actions) != len(noise) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid candidate chunks from server: shape {actions.shape}")
        return actions

    @staticmethod
    def _check(response: dict, op: str) -> None:
        if response.get("steering", {}).get("op") != op:
            raise ValueError(f"Server did not acknowledge steering op {op!r}")


def write_json(path: pathlib.Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def save_video(path: str | pathlib.Path, frames: list[np.ndarray], fps: int = 10) -> None:
    """Writes an mp4 (same codec settings as examples/libero/main.py)."""
    import imageio

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, [np.asarray(frame) for frame in frames], fps=fps)


def video_name(suite: str, task_description: str, index: int, *, success: bool) -> str:
    """main.py's naming, so rollouts line up with the baseline videos by initial-state index."""
    suffix = "success" if success else "failure"
    return f"rollout_{suite}_{task_description.replace(' ', '_')}_{index:03d}_{suffix}.mp4"
