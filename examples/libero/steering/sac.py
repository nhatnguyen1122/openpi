"""Hybrid SAC for noise-and-horizon steering.

Actions are (u, k): a squashed-Gaussian latent u that becomes model noise eps = b * tanh(u), and a categorical number
of environment steps k executed before the next decision. Decisions last k steps, so targets use the discount
gamma^(steps actually executed) that the worker reports (a semi-MDP backup).

Reference implementations: DSRL (ref/dsrl_pi0/jaxrl2/agents/pixel_sac) and RLinf's embodied SAC.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import pathlib
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from . import common

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


@dataclasses.dataclass(frozen=True)
class SACConfig:
    hidden_dims: tuple[int, ...] = (256, 256)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    tau: float = 0.005
    init_alpha_continuous: float = 0.1
    init_alpha_discrete: float = 0.1
    # None -> -noise_dim / 2 (continuous, as in DSRL) and 0.5 * log(|K|) (discrete).
    target_entropy_continuous: float | None = None
    target_entropy_discrete: float | None = None
    max_grad_norm: float | None = 10.0
    # Whether the critic target includes the entropy bonus. Off, as in DSRL and RLinf: with a 70-dimensional u the
    # bonus (~10 nats per decision) dwarfs the task reward (<= ~2 per episode) and the bootstrapped Q diverged.
    backup_entropy: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SACConfig:
        return cls(**{**data, "hidden_dims": tuple(data["hidden_dims"])})


def _mlp(in_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for width in hidden_dims:
        layers += [nn.Linear(in_dim, width), nn.ReLU()]
        in_dim = width
    return nn.Sequential(*layers)


class StateEncoder(nn.Module):
    """Normalizes the projected VLA embedding; the robot state and t/T are already O(1) and pass through."""

    def __init__(self, spec: common.SteeringSpec):
        super().__init__()
        self.projection_dim = spec.projection_dim
        self.norm = nn.LayerNorm(spec.projection_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.norm(s[..., : self.projection_dim]), s[..., self.projection_dim :]], dim=-1)


def squashed_log_prob(u: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor, bound: float) -> torch.Tensor:
    """log density of eps = bound * tanh(u) where u ~ N(mean, exp(log_std)^2), summed over dimensions."""
    gaussian = -0.5 * (((u - mean) / log_std.exp()) ** 2 + 2 * log_std + math.log(2 * math.pi))
    # log |d eps / d u| = log(bound) + log(1 - tanh(u)^2), computed stably.
    log_det = math.log(bound) + 2 * (math.log(2) - u - F.softplus(-2 * u))
    return (gaussian - log_det).sum(-1)


def unit_variance_std(bound: float) -> float:
    """sigma such that Var[bound * tanh(sigma z)] = 1 for z ~ N(0, 1); 1.0 when bound <= 1 (unreachable)."""
    if bound <= 1:
        return 1.0
    z, w = np.polynomial.hermite_e.hermegauss(80)
    w = w / w.sum()

    def variance(sigma: float) -> float:
        return float(np.sum(w * (bound * np.tanh(sigma * z)) ** 2))

    low, high = 1e-4, 50.0
    for _ in range(100):
        mid = math.sqrt(low * high)
        low, high = (mid, high) if variance(mid) < 1 else (low, mid)
    return math.sqrt(low * high)


class Actor(nn.Module):
    def __init__(self, spec: common.SteeringSpec, hidden_dims: tuple[int, ...]):
        super().__init__()
        self.spec = spec
        self.encoder = StateEncoder(spec)
        self.trunk = _mlp(spec.state_dim, hidden_dims)
        self.mean = nn.Linear(hidden_dims[-1], spec.noise_dim)
        self.log_std = nn.Linear(hidden_dims[-1], spec.noise_dim)
        self.k_logits = nn.Linear(hidden_dims[-1], spec.num_horizons)
        # Start at mean 0, uniform k, and a std for which eps = b * tanh(u) has unit variance, i.e. the initial
        # steered policy samples (approximately) the same noise distribution as the unsteered VLA.
        for head in (self.mean, self.log_std, self.k_logits):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.constant_(self.log_std.bias, math.log(unit_variance_std(spec.noise_bound)))

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(self.encoder(s))
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return self.mean(h), log_std, self.k_logits(h)

    def sample_u(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reparameterized u, log pi(eps | s) and log pi(k | s) for all k."""
        mean, log_std, logits = self(s)
        u = mean + log_std.exp() * torch.randn_like(mean)
        return u, squashed_log_prob(u, mean, log_std, self.spec.noise_bound), F.log_softmax(logits, -1)

    @torch.no_grad()
    def act(
        self, s: np.ndarray, *, deterministic: bool, generator: torch.Generator | None = None
    ) -> tuple[np.ndarray, int]:
        """Returns u [noise_dim] and the index of k in spec.horizons."""
        mean, log_std, logits = self(torch.as_tensor(s, dtype=torch.float32)[None])
        if deterministic:
            return mean[0].numpy(), int(logits[0].argmax())
        u = mean + log_std.exp() * torch.randn(mean.shape, generator=generator)
        k_index = torch.multinomial(F.softmax(logits[0], -1), 1, generator=generator)
        return u[0].numpy(), int(k_index)


class Critic(nn.Module):
    """Q(s, eps_ctrl, k); `all_horizons` evaluates every k at once."""

    def __init__(self, spec: common.SteeringSpec, hidden_dims: tuple[int, ...]):
        super().__init__()
        self.spec = spec
        self.encoder = StateEncoder(spec)
        self.net = nn.Sequential(
            _mlp(spec.state_dim + spec.noise_dim + spec.num_horizons, hidden_dims), nn.Linear(hidden_dims[-1], 1)
        )

    def forward(self, s: torch.Tensor, eps: torch.Tensor, k_onehot: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([self.encoder(s), eps, k_onehot], -1))[..., 0]

    def all_horizons(self, s: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        n = self.spec.num_horizons
        onehot = torch.eye(n, device=s.device).expand(s.shape[0], n, n)
        return self(s[:, None].expand(-1, n, -1), eps[:, None].expand(-1, n, -1), onehot)


class ReplayBuffer:
    """Ring buffer; states are stored as float16."""

    def __init__(self, spec: common.SteeringSpec, capacity: int, seed: int = 0):
        self.capacity = capacity
        self.s = np.zeros((capacity, spec.state_dim), np.float16)
        self.s2 = np.zeros((capacity, spec.state_dim), np.float16)
        self.u = np.zeros((capacity, spec.noise_dim), np.float32)
        self.k_index = np.zeros(capacity, np.int64)
        self.reward = np.zeros(capacity, np.float32)
        self.discount = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, np.float32)
        self._next = 0
        self.size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, transition: dict[str, Any]) -> None:
        i = self._next
        self.s[i] = transition["s"]
        self.s2[i] = transition["s2"]
        self.u[i] = transition["u"]
        self.k_index[i] = transition["k_index"]
        self.reward[i] = transition["reward"]
        self.discount[i] = transition["discount"]
        self.done[i] = transition["done"]
        self._next = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        idx = self._rng.integers(0, self.size, batch_size)
        fields = {
            "s": self.s[idx].astype(np.float32),
            "s2": self.s2[idx].astype(np.float32),
            "u": self.u[idx],
            "k_index": self.k_index[idx],
            "reward": self.reward[idx],
            "discount": self.discount[idx],
            "done": self.done[idx],
        }
        return {k: torch.as_tensor(v, device=device) for k, v in fields.items()}


class SACAgent:
    def __init__(self, spec: common.SteeringSpec, config: SACConfig, device: str = "cpu"):
        self.spec = spec
        self.config = config
        self.device = torch.device(device)
        self.actor = Actor(spec, config.hidden_dims).to(self.device)
        self.critics = nn.ModuleList([Critic(spec, config.hidden_dims) for _ in range(2)]).to(self.device)
        self.target_critics = copy.deepcopy(self.critics).requires_grad_(requires_grad=False)
        self.log_alpha_c = torch.tensor(math.log(config.init_alpha_continuous), device=self.device, requires_grad=True)
        self.log_alpha_d = torch.tensor(math.log(config.init_alpha_discrete), device=self.device, requires_grad=True)
        self.target_entropy_c = (
            -spec.noise_dim / 2 if config.target_entropy_continuous is None else config.target_entropy_continuous
        )
        self.target_entropy_d = (
            0.5 * math.log(spec.num_horizons)
            if config.target_entropy_discrete is None
            else config.target_entropy_discrete
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=config.critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha_c, self.log_alpha_d], lr=config.alpha_lr)
        self.num_updates = 0

    def _eps(self, u: torch.Tensor) -> torch.Tensor:
        # With learn_noise=False the noise is not a decision, so the critic must not depend on it: Q(s, k).
        if not self.spec.learn_noise:
            return torch.zeros_like(u)
        return self.spec.noise_bound * torch.tanh(u)

    def _sample(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u, logp_u, logp_k = self.actor.sample_u(s)
        if not self.spec.learn_noise:
            logp_u = torch.zeros_like(logp_u)
        return u, logp_u, logp_k

    def _step(self, optimizer: torch.optim.Optimizer, loss: torch.Tensor, params) -> None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)
        optimizer.step()

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        alpha_c, alpha_d = self.log_alpha_c.exp().detach(), self.log_alpha_d.exp().detach()
        k_onehot = F.one_hot(batch["k_index"], self.spec.num_horizons).float()

        # Critic: y = R + gamma^steps * (1 - done) * E_k' [min Q'(s', eps', k') - a_c log pi(eps') - a_d log pi(k')].
        with torch.no_grad():
            u2, logp_u2, logp_k2 = self._sample(batch["s2"])
            eps2 = self._eps(u2)
            q2 = torch.minimum(*(critic.all_horizons(batch["s2"], eps2) for critic in self.target_critics))
            bonus = -(alpha_c * logp_u2[:, None] + alpha_d * logp_k2) if self.config.backup_entropy else 0.0
            v2 = (logp_k2.exp() * (q2 + bonus)).sum(-1)
            target = batch["reward"] + batch["discount"] * (1 - batch["done"]) * v2
        eps = self._eps(batch["u"])
        q1, q2_ = (critic(batch["s"], eps, k_onehot) for critic in self.critics)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2_, target)
        self._step(self.critic_optimizer, critic_loss, self.critics.parameters())

        # Actor: E_k~pi [a_d log pi(k) + a_c log pi(eps) - min Q(s, eps, k)].
        u, logp_u, logp_k = self._sample(batch["s"])
        q = torch.minimum(*(critic.all_horizons(batch["s"], self._eps(u)) for critic in self.critics))
        p_k = logp_k.exp()
        actor_loss = (p_k * (alpha_d * logp_k + alpha_c * logp_u[:, None] - q)).sum(-1).mean()
        self._step(self.actor_optimizer, actor_loss, self.actor.parameters())
        self.critics.zero_grad(set_to_none=True)

        # Temperatures.
        entropy_k = -(p_k * logp_k).sum(-1).detach()
        alpha_loss = (self.log_alpha_d * (entropy_k - self.target_entropy_d)).mean()
        if self.spec.learn_noise:
            alpha_loss = alpha_loss - (self.log_alpha_c * (logp_u.detach() + self.target_entropy_c)).mean()
        self._step(self.alpha_optimizer, alpha_loss, [self.log_alpha_c, self.log_alpha_d])

        with torch.no_grad():
            for target_param, param in zip(self.target_critics.parameters(), self.critics.parameters(), strict=True):
                target_param.lerp_(param, self.config.tau)
        self.num_updates += 1
        mean_k = (p_k.detach() * torch.tensor(self.spec.horizons, device=self.device, dtype=torch.float32)).sum(-1)
        return {
            "critic_loss": float(critic_loss),
            "actor_loss": float(actor_loss),
            "q_mean": float(q1.mean()),
            "target_mean": float(target.mean()),
            "alpha_continuous": float(alpha_c),
            "alpha_discrete": float(alpha_d),
            "entropy_continuous": float(-logp_u.mean()),
            "entropy_discrete": float(entropy_k.mean()),
            "policy_mean_k": float(mean_k.mean()),
        }

    def actor_payload(self, version: int) -> dict[str, Any]:
        state = {k: v.detach().cpu() for k, v in self.actor.state_dict().items()}
        return {"version": version, "spec": self.spec.to_dict(), "sac": self.config.to_dict(), "actor": state}

    def state_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "sac": self.config.to_dict(),
            "actor": self.actor.state_dict(),
            "critics": self.critics.state_dict(),
            "target_critics": self.target_critics.state_dict(),
            "log_alpha_c": self.log_alpha_c.detach().cpu(),
            "log_alpha_d": self.log_alpha_d.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critics.load_state_dict(state["critics"])
        self.target_critics.load_state_dict(state["target_critics"])
        with torch.no_grad():
            self.log_alpha_c.copy_(state["log_alpha_c"])
            self.log_alpha_d.copy_(state["log_alpha_d"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        self.num_updates = state["num_updates"]


def save_atomic(payload: Any, path: pathlib.Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


class SteeringPolicy:
    """CPU actor + projection: turns (embedding, robot state, t) into model noise and a horizon k."""

    def __init__(self, spec: common.SteeringSpec, projection: np.ndarray, actor: Actor, version: int = 0):
        if projection.shape != (spec.projection_dim, spec.embedding_dim):
            raise ValueError(f"Projection shape {projection.shape} does not match the spec")
        self.spec = spec
        self.projection = projection
        self.actor = actor.eval()
        self.mapper = common.NoiseMapper(spec)
        self.version = version

    @classmethod
    def load(cls, run_dir: str | pathlib.Path, checkpoint: str = "actor.pt") -> SteeringPolicy:
        run_dir = pathlib.Path(run_dir)
        path = pathlib.Path(checkpoint) if pathlib.Path(checkpoint).is_absolute() else run_dir / checkpoint
        payload = torch.load(path, map_location="cpu", weights_only=False)
        spec = common.SteeringSpec.from_dict(payload["spec"])
        actor = Actor(spec, tuple(payload["sac"]["hidden_dims"]))
        actor.load_state_dict(payload["actor"])
        return cls(spec, np.load(run_dir / "projection.npy"), actor, payload["version"])

    def load_weights(self, payload: dict[str, Any]) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.version = payload["version"]

    def state(self, embedding: np.ndarray, robot_state: np.ndarray, t: int, max_steps: int) -> np.ndarray:
        return common.build_state(self.projection, embedding, robot_state, t, max_steps)

    def decide(
        self,
        s: np.ndarray,
        rng: np.random.Generator,
        *,
        deterministic: bool,
        generator: torch.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Returns (model noise [H, D], u [noise_dim], k_index)."""
        u, k_index = self.actor.act(s, deterministic=deterministic, generator=generator)
        if not self.spec.learn_noise:
            noise, u = self.mapper.base_policy_noise(rng)
            return noise, u, k_index
        return self.mapper.to_noise(u, rng), u, k_index
