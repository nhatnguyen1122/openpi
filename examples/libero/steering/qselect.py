"""Value-guided candidate selection for a frozen pi0.5 policy (no actor, no gradient through the noise).

At each decision the VLA samples N chunks from eps ~ N(0, I) for the current observation (one prefix, one batched
call). An ensemble critic Q(s, a) scores the *executed actions* a, and the agent executes the best candidate. Because
the noise always comes from the VLA's own prior, the policy never leaves the base policy's support; RL only re-ranks
what pi0.5 would do anyway (as in V-GPS / value-guided steering), and Q is learned on actions rather than on noise
(as in Q-chunking), so many noises that decode to similar actions share one value.

Two modes, set by `segment` and `allow_continue`:
    segment = H (10), no continue: choose one of N full chunks (fixed k = 10).
    segment = 5, allow_continue:   every 5 steps choose among N fresh half-chunks and the unexecuted second half of
        the previous chunk. "Continue" = k = 10 and "fresh" = replan after 5, so the replanning decision is taken
        with the newest observation (TempoRL-style "when to act") instead of being fixed when the chunk is drawn.
        All decisions last 5 steps, so k does not change how many transitions an episode produces.

Targets: y = r + gamma^steps * max_{a' in C(s')} [mean_e Q'_e(s', a') - beta * std_e Q'_e(s', a')], clipped to the
range of achievable returns. C(s') is the candidate set that was actually available at the next decision.
"""

from __future__ import annotations

import copy
import dataclasses
import pathlib
from typing import Any

import numpy as np
import torch
from torch import nn

from . import common
from .sac import StateEncoder
from .sac import _mlp
from .sac import save_atomic


@dataclasses.dataclass(frozen=True)
class QSelectConfig:
    num_candidates: int = 8  # N fresh chunks per decision.
    segment: int = 10  # Actions executed per decision.
    allow_continue: bool = False  # Also offer the unexecuted rest of the previous chunk (needs segment < H).
    action_dim: int = 7
    ensemble: int = 5
    hidden_dims: tuple[int, ...] = (256, 256)
    lr: float = 3e-4
    batch_size: int = 256
    tau: float = 0.005
    pessimism: float = 0.5  # beta in mean - beta * std, for targets and greedy selection.
    q_min: float = -1.5  # Returns are 1[success] + telescoped shaping in [-lambda, lambda].
    q_max: float = 2.5
    explore_eps: float = 0.2  # Probability of an exploratory (base-policy) choice during training.
    continue_prob: float = 0.5  # Exploratory choice: continue with this probability when continuing is possible.
    use_embedding: bool = True  # False (ablation): the critic sees only robot state and time; P e is zeroed.

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QSelectConfig:
        return cls(**{**data, "hidden_dims": tuple(data["hidden_dims"])})

    @property
    def num_options(self) -> int:
        return self.num_candidates + int(self.allow_continue)

    @property
    def segment_dim(self) -> int:
        return self.segment * self.action_dim


class EnsembleCritic(nn.Module):
    """E independent Q(s, a) heads; returns [E, ...]."""

    def __init__(self, spec: common.SteeringSpec, config: QSelectConfig):
        super().__init__()
        self.encoder = StateEncoder(spec)
        in_dim = spec.state_dim + config.segment_dim
        self.heads = nn.ModuleList(
            nn.Sequential(_mlp(in_dim, config.hidden_dims), nn.Linear(config.hidden_dims[-1], 1))
            for _ in range(config.ensemble)
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.encoder(s), a], -1)
        return torch.stack([head(x)[..., 0] for head in self.heads])

    def score(self, s: torch.Tensor, a: torch.Tensor, pessimism: float) -> torch.Tensor:
        q = self(s, a)
        return q.mean(0) - pessimism * q.std(0) if len(self.heads) > 1 else q[0]


class ReplayBuffer:
    def __init__(self, spec: common.SteeringSpec, config: QSelectConfig, capacity: int, seed: int = 0):
        self.capacity = capacity
        m, d = config.num_options, config.segment_dim
        self.s = np.zeros((capacity, spec.state_dim), np.float16)
        self.s2 = np.zeros((capacity, spec.state_dim), np.float16)
        self.a = np.zeros((capacity, d), np.float32)
        self.next_a = np.zeros((capacity, m, d), np.float32)
        self.next_mask = np.zeros((capacity, m), bool)
        self.reward = np.zeros(capacity, np.float32)
        self.discount = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.size = 0
        self.index = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, t: dict[str, Any]) -> None:
        i = self.index
        self.s[i], self.s2[i], self.a[i] = t["s"], t["s2"], t["a"]
        self.next_mask[i] = False
        n = len(t["next_a"])
        if n:
            self.next_a[i, :n] = t["next_a"]
            self.next_mask[i, :n] = True
        self.reward[i], self.discount[i], self.done[i] = t["reward"], t["discount"], t["done"]
        self.index = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = self.rng.integers(self.size, size=batch_size)
        as_tensor = lambda x: torch.as_tensor(x[idx])  # noqa: E731
        batch = {k: as_tensor(getattr(self, k)) for k in ("a", "next_a", "next_mask", "reward", "discount", "done")}
        batch["s"] = torch.as_tensor(self.s[idx].astype(np.float32))
        batch["s2"] = torch.as_tensor(self.s2[idx].astype(np.float32))
        return batch


class QSelectAgent:
    def __init__(self, spec: common.SteeringSpec, config: QSelectConfig, seed: int = 0):
        torch.manual_seed(seed)
        self.spec, self.config = spec, config
        self.critic = EnsembleCritic(spec, config)
        self.target = copy.deepcopy(self.critic).requires_grad_(requires_grad=False)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr)
        self.num_updates = 0

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        c = self.config
        with torch.no_grad():
            b, m, _ = batch["next_a"].shape
            s2 = batch["s2"][:, None].expand(-1, m, -1)
            score = self.target.score(s2, batch["next_a"], c.pessimism)  # [B, M]
            score = score.masked_fill(~batch["next_mask"], -torch.inf)
            best = torch.where(batch["next_mask"].any(-1), score.max(-1).values, torch.zeros(b))
            y = batch["reward"] + batch["discount"] * (1 - batch["done"]) * best
            y = y.clamp(c.q_min, c.q_max)
        q = self.critic(batch["s"], batch["a"])  # [E, B]
        loss = ((q - y) ** 2).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.optimizer.step()
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.target.parameters(), strict=True):
                tp.lerp_(p, c.tau)
        self.num_updates += 1
        return {
            "critic_loss": float(loss),
            "q_mean": float(q.mean()),
            "q_std_ensemble": float(q.std(0).mean()) if c.ensemble > 1 else 0.0,
            "target_mean": float(y.mean()),
        }

    def payload(self) -> dict[str, Any]:
        return {
            "critic": self.critic.state_dict(),
            "spec": self.spec.to_dict(),
            "config": self.config.to_dict(),
            "version": self.num_updates,
        }


class Selector:
    """CPU copy of the critic plus the episode logic shared by training workers and evaluation."""

    def __init__(self, spec: common.SteeringSpec, config: QSelectConfig, projection: np.ndarray, version: int = -1):
        self.spec, self.config, self.projection = spec, config, projection
        self.critic = EnsembleCritic(spec, config).eval()
        self.version = version  # -1 = untrained (only exploratory choices are meaningful).

    @classmethod
    def load(cls, path: str | pathlib.Path) -> Selector:
        path = pathlib.Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        spec = common.SteeringSpec.from_dict(payload["spec"])
        selector = cls(spec, QSelectConfig.from_dict(payload["config"]), np.load(path.parent / "projection.npy"))
        selector.load_weights(payload)
        return selector

    def load_weights(self, payload: dict[str, Any]) -> None:
        self.critic.load_state_dict(payload["critic"])
        self.version = int(payload["version"])

    def state(self, embedding: np.ndarray, robot_state: np.ndarray, t: int, max_steps: int) -> np.ndarray:
        s = common.build_state(self.projection, embedding, robot_state, t, max_steps)
        if not self.config.use_embedding:
            s[: self.projection.shape[0]] = 0.0
        return s

    @torch.no_grad()
    def scores(self, s: np.ndarray, options: np.ndarray) -> np.ndarray:
        a = torch.as_tensor(options.reshape(len(options), -1), dtype=torch.float32)
        s = torch.as_tensor(s, dtype=torch.float32)[None].expand(len(options), -1)
        return self.critic.score(s, a, self.config.pessimism).numpy()

    def explore(self, num_fresh: int, rng: np.random.Generator, *, has_continue: bool) -> int:
        """Base-policy choice: a random fresh chunk, or (when possible) continuing with probability continue_prob."""
        if has_continue and rng.random() < self.config.continue_prob:
            return num_fresh
        return int(rng.integers(num_fresh))

    def choose(
        self, s: np.ndarray, options: np.ndarray, rng: np.random.Generator, *, has_continue: bool, mode: str
    ) -> int:
        """mode: 'greedy' (evaluation), 'train' (epsilon-greedy), or 'random' (the unsteered base policy)."""
        num_fresh = len(options) - int(has_continue)
        if mode == "random" or (mode == "train" and rng.random() < self.config.explore_eps):
            return self.explore(num_fresh, rng, has_continue=has_continue)
        return int(np.argmax(self.scores(s, options)))


def run_episode(
    env,
    selector: Selector,
    rng: np.random.Generator,
    *,
    mode: str,
    init_state: np.ndarray | None = None,
    emit=None,
) -> dict[str, Any]:
    """One episode with candidate selection; emits one transition per decision when `emit` is given."""
    import time

    spec, config = selector.spec, selector.config
    seg, horizon = config.segment, spec.action_horizon
    if seg > horizon or (config.allow_continue and 2 * seg > horizon):
        raise ValueError("segment too long for the action horizon")
    start = time.monotonic()
    env.reset(init_state)
    leftover = None

    def observe(leftover):
        s = selector.state(env.embed(), env.robot_state(), env.t, env.max_steps)
        noise = rng.standard_normal((config.num_candidates, horizon, spec.model_action_dim)).astype(np.float32)
        chunks = env.sample_chunks(noise)
        options = chunks[:, :seg]
        has_continue = config.allow_continue and leftover is not None
        if has_continue:
            options = np.concatenate([options, leftover[None]])
        return s, chunks, options, has_continue

    s, chunks, options, has_continue = observe(leftover)
    decisions, continued, episode_return = 0, 0, 0.0
    while True:
        index = selector.choose(s, options, rng, has_continue=has_continue, mode=mode)
        chosen = options[index]
        is_continue = has_continue and index == len(options) - 1
        continued += int(is_continue)
        leftover = None if (is_continue or not config.allow_continue) else chunks[index, seg : 2 * seg]
        result = env.execute(chosen)
        decisions += 1
        episode_return += result.reward
        if result.success:
            s2, next_options = s, np.zeros((0, *chosen.shape), np.float32)
        else:
            s2, chunks, next_options, next_has_continue = observe(leftover)
        if emit is not None:
            emit(
                {
                    "s": s,
                    "a": chosen.reshape(-1).astype(np.float32),
                    "reward": result.reward,
                    "discount": result.discount,
                    "done": float(result.success),
                    "s2": s2,
                    "next_a": next_options.reshape(len(next_options), config.segment_dim).astype(np.float32),
                }
            )
        if result.success or result.truncated:
            break
        s, options, has_continue = s2, next_options, next_has_continue
    return {
        "success": bool(result.success),
        "policy_steps": env.t,
        "decisions": decisions,
        "continue_fraction": continued / decisions,
        "shaped_return": episode_return,
        "final_predicates": env.satisfied_predicates(),
        "vla_seconds": env.vla_seconds,
        "wall_seconds": time.monotonic() - start,
    }


__all__ = ["EnsembleCritic", "QSelectAgent", "QSelectConfig", "ReplayBuffer", "Selector", "run_episode", "save_atomic"]
