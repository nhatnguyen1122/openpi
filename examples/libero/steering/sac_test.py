import numpy as np
import pytest
import torch
from torch import distributions as D  # noqa: N812

from examples.libero.steering import common
from examples.libero.steering import fakes
from examples.libero.steering import sac


def _spec(**overrides):
    return fakes.make_spec(**overrides)


def test_squashed_log_prob_matches_torch_distributions():
    torch.manual_seed(0)
    mean, log_std = torch.randn(4, 70), torch.randn(4, 70).clamp(-1, 1)
    bound = 2.5
    dist = D.TransformedDistribution(D.Normal(mean, log_std.exp()), [D.TanhTransform(), D.AffineTransform(0.0, bound)])
    u = mean + log_std.exp() * torch.randn_like(mean)
    expected = dist.log_prob(bound * torch.tanh(u)).sum(-1)
    np.testing.assert_allclose(sac.squashed_log_prob(u, mean, log_std, bound), expected, rtol=1e-4, atol=1e-3)


def test_initial_actor_matches_unsteered_noise_with_uniform_k():
    spec = _spec()
    actor = sac.Actor(spec, (32, 32))
    mean, log_std, logits = actor(torch.randn(5, spec.state_dim))
    assert torch.all(mean == 0)
    assert torch.all(logits == 0)
    torch.testing.assert_close(log_std, torch.full_like(log_std, log_std[0, 0].item()))
    u, _, _ = actor.sample_u(torch.zeros(4000, spec.state_dim))
    eps = spec.noise_bound * torch.tanh(u)
    assert abs(eps.std().item() - 1) < 0.01


@pytest.mark.parametrize("bound", [1.5, 2.5, 5.0])
def test_unit_variance_std(bound):
    sigma = sac.unit_variance_std(bound)
    z = np.random.default_rng(0).standard_normal(1_000_000)
    assert np.std(bound * np.tanh(sigma * z)) == pytest.approx(1.0, abs=5e-3)
    assert sac.unit_variance_std(1.0) == 1.0


def test_critic_all_horizons_matches_individual_calls():
    spec = _spec()
    critic = sac.Critic(spec, (32, 32))
    s, eps = torch.randn(3, spec.state_dim), torch.randn(3, spec.noise_dim)
    all_q = critic.all_horizons(s, eps)
    assert all_q.shape == (3, spec.num_horizons)
    for k in range(spec.num_horizons):
        onehot = torch.nn.functional.one_hot(torch.full((3,), k), spec.num_horizons).float()
        torch.testing.assert_close(all_q[:, k], critic(s, eps, onehot))


def _batch(spec, n, *, reward, done, discount=0.9):
    return {
        "s": torch.randn(n, spec.state_dim),
        "s2": torch.randn(n, spec.state_dim),
        "u": torch.randn(n, spec.noise_dim),
        "k_index": torch.randint(spec.num_horizons, (n,)),
        "reward": torch.full((n,), reward),
        "discount": torch.full((n,), discount),
        "done": torch.full((n,), done),
    }


def test_terminal_targets_do_not_bootstrap():
    spec = _spec()
    agent = sac.SACAgent(spec, sac.SACConfig(hidden_dims=(32, 32)))
    info = agent.update(_batch(spec, 16, reward=1.0, done=1.0))
    assert info["target_mean"] == pytest.approx(1.0)
    info = agent.update(_batch(spec, 16, reward=1.0, done=0.0))
    assert info["target_mean"] != pytest.approx(1.0)
    assert agent.num_updates == 2


def test_learns_best_horizon_and_noise_direction_on_a_bandit():
    """One-step problem: reward 1 only for k = horizons[2] and positive controlled noise."""
    torch.manual_seed(0)
    spec = _spec()
    config = sac.SACConfig(hidden_dims=(64, 64), batch_size=128, actor_lr=1e-3, critic_lr=1e-3, alpha_lr=1e-3)
    agent = sac.SACAgent(spec, config)
    buffer = sac.ReplayBuffer(spec, 20_000, seed=0)
    rng = np.random.default_rng(0)
    s = np.zeros(spec.state_dim, np.float32)
    for _ in range(4000):
        u = rng.normal(0, 1.5, spec.noise_dim).astype(np.float32)
        k_index = int(rng.integers(spec.num_horizons))
        reward = float(k_index == 2 and np.tanh(u).mean() > 0)
        buffer.add({"s": s, "s2": s, "u": u, "k_index": k_index, "reward": reward, "discount": 0.9, "done": 1.0})
    for _ in range(1500):
        agent.update(buffer.sample(config.batch_size))
    u, k_index = agent.actor.act(s, deterministic=True)
    assert k_index == 2
    assert np.tanh(u).mean() > 0


def test_replay_buffer_wraps_and_stores_float16_states():
    spec = _spec()
    buffer = sac.ReplayBuffer(spec, 3)
    for i in range(5):
        s = np.full(spec.state_dim, i + 0.1, np.float32)
        buffer.add({"s": s, "s2": s, "u": np.zeros(70), "k_index": i % 6, "reward": i, "discount": 1, "done": 0})
    assert len(buffer) == 3
    assert sorted(buffer.reward.tolist()) == [2.0, 3.0, 4.0]
    assert buffer.s.dtype == np.float16
    batch = buffer.sample(8)
    assert batch["s"].dtype == torch.float32
    assert batch["k_index"].dtype == torch.int64


def test_steering_policy_roundtrip(tmp_path):
    spec = _spec()
    agent = sac.SACAgent(spec, sac.SACConfig(hidden_dims=(32, 32)))
    for _ in range(3):
        agent.update(_batch(spec, 8, reward=0.5, done=0.0))
    np.save(tmp_path / "projection.npy", common.make_projection(spec))
    sac.save_atomic(agent.actor_payload(version=3), tmp_path / "actor.pt")
    policy = sac.SteeringPolicy.load(tmp_path)
    assert policy.version == 3
    assert policy.spec == spec

    s = policy.state(np.ones(16), np.zeros(8), 0, 10)
    noise, u, k_index = policy.decide(s, np.random.default_rng(0), deterministic=True)
    expected_u, expected_k = agent.actor.act(s, deterministic=True)
    np.testing.assert_allclose(u, expected_u)
    assert k_index == expected_k
    assert noise.shape == (10, 32)

    restored = sac.SACAgent(spec, sac.SACConfig(hidden_dims=(32, 32)))
    restored.load_state_dict(agent.state_dict())
    assert restored.num_updates == 3
    np.testing.assert_allclose(restored.actor.act(s, deterministic=True)[0], expected_u)


def test_k_only_mode_ignores_noise_and_learns_horizon():
    torch.manual_seed(0)
    spec = _spec(learn_noise=False)
    config = sac.SACConfig(hidden_dims=(64, 64), batch_size=128, actor_lr=1e-3, critic_lr=1e-3, alpha_lr=1e-3)
    agent = sac.SACAgent(spec, config)
    buffer = sac.ReplayBuffer(spec, 5000, seed=0)
    rng = np.random.default_rng(0)
    s = np.zeros(spec.state_dim, np.float32)
    for _ in range(3000):
        k_index = int(rng.integers(spec.num_horizons))
        u = rng.normal(size=spec.noise_dim).astype(np.float32)
        buffer.add(
            {"s": s, "s2": s, "u": u, "k_index": k_index, "reward": float(k_index == 4), "discount": 0.9, "done": 1.0}
        )
    alpha_c = agent.log_alpha_c.item()
    for _ in range(800):
        info = agent.update(buffer.sample(config.batch_size))
    assert agent.log_alpha_c.item() == alpha_c  # continuous temperature is unused
    assert info["entropy_continuous"] == 0
    # Q does not depend on the noise.
    critic = agent.critics[0]
    st = torch.zeros(2, spec.state_dim)
    eps_a, eps_b = (agent._eps(torch.randn(2, 70)) for _ in range(2))  # noqa: SLF001
    torch.testing.assert_close(critic.all_horizons(st, eps_a), critic.all_horizons(st, eps_b))
    # The executed noise is the unsteered N(0, I); k comes from the actor.
    policy = sac.SteeringPolicy(spec, common.make_projection(spec), agent.actor)
    noise, u, k_index = policy.decide(s, np.random.default_rng(3), deterministic=True)
    reference, reference_u = common.NoiseMapper(spec).base_policy_noise(np.random.default_rng(3))
    np.testing.assert_array_equal(noise, reference)
    np.testing.assert_array_equal(u, reference_u)
    assert k_index == 4


def test_q_stays_bounded_with_high_entropy_actor_when_entropy_is_not_backed_up():
    """Regression: with a 70-dim u, an entropy bonus in the target (~10 per decision) swamped rewards and Q diverged."""
    spec = _spec()
    s = np.zeros(spec.state_dim, np.float32)

    def q_after_training(backup_entropy):
        torch.manual_seed(0)
        agent = sac.SACAgent(spec, sac.SACConfig(hidden_dims=(64, 64), backup_entropy=backup_entropy))
        buffer = sac.ReplayBuffer(spec, 2000, seed=0)
        rng = np.random.default_rng(0)
        for _ in range(2000):
            u = rng.normal(0, 0.5, spec.noise_dim).astype(np.float32)
            buffer.add(
                {"s": s, "s2": s, "u": u, "k_index": int(rng.integers(6)), "reward": 0.0, "discount": 0.95, "done": 0.0}
            )
        for _ in range(300):
            info = agent.update(buffer.sample(128))
        return info["q_mean"]

    assert abs(q_after_training(backup_entropy=False)) < 1.0
    assert q_after_training(backup_entropy=True) > 5.0  # the failure this guards against
