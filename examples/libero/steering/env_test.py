import numpy as np
import pytest

from examples.libero.steering import common
from examples.libero.steering import env as steering_env
from examples.libero.steering import fakes


def _env(spec=None, max_steps=20, **fake_kwargs):
    spec = spec or fakes.make_spec(gamma=0.9)
    server = fakes.FakeSteeringServer()
    client = common.SteeringClient("", 0, policy=server)
    raw = fakes.FakeLiberoEnv(**fake_kwargs)
    wrapped = steering_env.SteeringEnv(
        raw, "task", max_steps, client, spec, make_element=fakes.make_element, robot_state=fakes.robot_state
    )
    return wrapped, raw, server


def test_macro_step_shaping_and_discount():
    wrapped, raw, _ = _env()
    wrapped.reset()
    assert raw.steps == 10
    assert wrapped.t == 0
    assert wrapped.satisfied_predicates() == 0
    wrapped.embed()
    result = wrapped.step_macro(np.zeros((10, 32), np.float32), 5)
    # First predicate becomes true at policy step 3 (phi: 0 -> 0.5); shaping telescopes to gamma^5 phi_5 - phi_0.
    gamma = 0.9
    per_step = [0.0, 0.0, gamma * 0.5, gamma * 0.5 - 0.5, gamma * 0.5 - 0.5]
    assert result.reward == pytest.approx(sum(gamma**i * r for i, r in enumerate(per_step)))
    assert result.reward == pytest.approx(gamma**5 * 0.5)
    assert result.discount == pytest.approx(gamma**5)
    assert (result.steps, result.success, result.truncated) == (5, False, False)
    assert len(raw.actions) == 5


def test_success_stops_macro_early():
    wrapped, _, _ = _env()
    wrapped.reset()
    wrapped.embed()
    wrapped.step_macro(np.zeros((10, 32), np.float32), 5)
    wrapped.embed()
    result = wrapped.step_macro(np.zeros((10, 32), np.float32), 5)  # success at policy step 7
    gamma = 0.9
    assert result.steps == 2
    assert result.success
    assert not result.truncated
    assert result.discount == pytest.approx(gamma**2)
    # Step 6: nothing changes (phi 0.5 -> 0.5); step 7: success + gamma * 1.0 - 0.5.
    assert result.reward == pytest.approx(gamma * 0.5 - 0.5 + gamma * (1 + gamma * 1.0 - 0.5))


def test_shaped_returns_telescope_to_sparse_success():
    # With shaping, the discounted episode return equals the sparse return plus gamma^T phi(final) - phi(initial).
    wrapped, _, _ = _env()
    transitions = []
    policy = lambda s: (np.zeros((10, 32), np.float32), np.zeros(70, np.float32), 0)  # noqa: E731
    summary = steering_env.run_episode(
        wrapped, policy, state_fn=lambda e, q, t, max_t: np.zeros(17, np.float32), emit=transitions.append
    )
    assert summary["success"]
    assert summary["policy_steps"] == 7
    total, discount = 0.0, 1.0
    for transition in transitions:
        total += discount * transition["reward"]
        discount *= transition["discount"]
    gamma = 0.9
    assert total == pytest.approx(gamma**6 * 1 + gamma**7 * 1.0 - 0.0)


def test_run_episode_transitions_and_truncation():
    wrapped, _, server = _env(max_steps=12, first=100, second=100)
    transitions = []
    states = []

    def state_fn(embedding, robot_state, t, max_steps):
        states.append(t)
        return np.full(17, t, np.float32)

    choose = lambda s: (np.zeros((10, 32), np.float32), np.full(70, 0.1, np.float32), 1)  # noqa: E731  (k = 6)
    summary = steering_env.run_episode(wrapped, choose, state_fn=state_fn, emit=transitions.append)
    assert not summary["success"]
    assert summary["policy_steps"] == 12
    assert summary["decisions"] == 2
    assert states == [0, 6, 12]  # final embed for bootstrapping the truncated step
    assert [t["done"] for t in transitions] == [0.0, 0.0]
    assert transitions[0]["s2"][0] == 6
    assert transitions[1]["s"][0] == 6
    assert transitions[1]["s2"][0] == 12
    assert all(t["k_index"] == 1 for t in transitions)
    assert server.calls == ["embed", "act", "embed", "act", "embed"]


def test_terminal_transition_does_not_embed_again():
    wrapped, _, server = _env(first=1, second=2)
    transitions = []
    choose = lambda s: (np.zeros((10, 32), np.float32), np.zeros(70, np.float32), 0)  # noqa: E731
    summary = steering_env.run_episode(
        wrapped, choose, state_fn=lambda e, q, t, max_t: np.zeros(17, np.float32), emit=transitions.append
    )
    assert summary["success"]
    assert summary["decisions"] == 1
    assert transitions[0]["done"] == 1.0
    assert server.calls == ["embed", "act"]


def test_init_state_is_used_only_when_given():
    wrapped, raw, _ = _env()
    wrapped.reset()
    assert raw.init_states == []
    wrapped.reset(init_state="fixed")
    assert raw.init_states == ["fixed"]
