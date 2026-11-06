import types

import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model
from openpi.policies import policy
from openpi.shared import nnx_utils

HORIZON, DIM = 4, 6


class FakeModel:
    """Embedding = state; actions = noise + 10 * state[0] + num_steps (cache holds the state)."""

    action_horizon = HORIZON
    action_dim = DIM

    def __init__(self):
        self.prefills = 0

    def sample_actions(self, rng, observation, **kwargs):
        raise AssertionError("steering must not use sample_actions")

    def prefill_steering(self, observation):
        self.prefills += 1
        state = jnp.asarray(observation["state"], jnp.float32)
        return state, observation, jnp.ones((1, 3), bool), state

    def sample_actions_from_prefix(self, observation, prefix_mask, kv_cache, noise, *, num_steps=10):
        return noise + 10 * kv_cache[:, :1, None] + num_steps


@pytest.fixture
def steered(monkeypatch):
    monkeypatch.setattr(nnx_utils, "module_jit", lambda f: f)
    monkeypatch.setattr(model.Observation, "from_dict", lambda data: data)
    fake = FakeModel()
    instance = policy.Policy(
        fake,
        sample_kwargs={"num_steps": 5},
        output_transforms=[lambda data: {**data, "actions": data["actions"][..., :2]}],
        max_steering_sessions=2,
    )
    return instance, fake


def _request(**fields):
    return {policy.STEERING_KEY: fields}


def _noise(value=0.0):
    return np.full((HORIZON, DIM), value, np.float32)


def test_embed_then_act_reuses_prefix(steered):
    instance, fake = steered
    out = instance.infer({"state": np.array([1.0, 2.0]), **_request(op="embed", session="a")})
    np.testing.assert_array_equal(out["prefix_embedding"], [1.0, 2.0])
    assert "actions" not in out

    first = instance.infer(_request(op="act", session="a", noise=_noise(0.5)))
    second = instance.infer(_request(op="act", session="a", noise=_noise(0.5), num_steps=1))
    assert fake.prefills == 1
    np.testing.assert_array_equal(first["actions"], np.full((HORIZON, 2), 0.5 + 10 + 5))
    np.testing.assert_array_equal(second["actions"], np.full((HORIZON, 2), 0.5 + 10 + 1))
    assert first["steering"] == {"op": "act", "num_steps": 5}


def test_sessions_are_isolated_and_evicted(steered):
    instance, _ = steered
    for session, value in [("a", 1.0), ("b", 2.0)]:
        instance.infer({"state": np.array([value]), **_request(op="embed", session=session)})
    assert instance.infer(_request(op="act", session="a", noise=_noise()))["actions"][0, 0] == 15
    assert instance.infer(_request(op="act", session="b", noise=_noise()))["actions"][0, 0] == 25
    # "a" was used before "b", so a third session evicts "a".
    instance.infer({"state": np.array([3.0]), **_request(op="embed", session="c")})
    with pytest.raises(ValueError, match="Unknown steering session"):
        instance.infer(_request(op="act", session="a", noise=_noise()))
    assert instance.infer(_request(op="act", session="b", noise=_noise()))["actions"][0, 0] == 25


def test_stateless_infer_returns_actions_and_embedding(steered):
    instance, _ = steered
    out = instance.infer({"state": np.array([2.0]), **_request(op="infer", noise=_noise(1.0)[None])})
    np.testing.assert_array_equal(out["prefix_embedding"], [2.0])
    np.testing.assert_array_equal(out["actions"], np.full((HORIZON, 2), 1 + 20 + 5))


@pytest.mark.parametrize(
    ("obs", "match"),
    [
        (_request(op="nope"), "Steering op"),
        (_request(op="embed"), "takes keys"),
        (_request(op="act", session="a", noise=_noise(), extra=1), "takes keys"),
        (_request(op="act", session="", noise=_noise()), "nonempty string"),
        (_request(op="infer", noise=_noise(), num_steps=0), "num_steps"),
        (_request(op="infer", noise=np.zeros((HORIZON, DIM - 1))), "shape"),
        (_request(op="infer", noise=np.full((HORIZON, DIM), np.nan)), "finite"),
        ({"state": np.zeros(1), **_request(op="act", session="a", noise=_noise())}, "must not include"),
        ({**_request(op="infer", noise=_noise()), "_openpi_inference": {"num_steps": 1, "seed": 0}}, "combined"),
    ],
)
def test_invalid_requests_are_rejected(steered, obs, match):
    instance, _ = steered
    obs = {"state": np.zeros(1), **obs} if "act" not in str(obs.get(policy.STEERING_KEY, {}).get("op")) else obs
    with pytest.raises(ValueError, match=match):
        instance.infer(obs)


def test_metadata_describes_noise_shape(steered):
    instance, _ = steered
    meta = instance.steering_metadata
    assert meta["action_horizon"] == HORIZON
    assert meta["action_dim"] == DIM
    assert meta["default_num_steps"] == 5
    assert meta["key"] == policy.STEERING_KEY


def test_unsupported_models_report_no_steering(monkeypatch):
    monkeypatch.setattr(nnx_utils, "module_jit", lambda f: f)
    instance = policy.Policy(types.SimpleNamespace(sample_actions=lambda *a, **k: None))
    assert instance.steering_metadata is None
    with pytest.raises(ValueError, match="does not support"):
        instance.infer(_request(op="infer", noise=_noise()))


def test_act_batch_returns_one_chunk_per_noise(steered):
    instance, fake = steered
    instance.infer({"state": np.array([1.0, 2.0]), **_request(op="embed", session="a")})
    noise = np.stack([_noise(0.0), _noise(1.0), _noise(2.0)])
    out = instance.infer(_request(op="act_batch", session="a", noise=noise))
    assert out["actions"].shape == (3, HORIZON, 2)
    for i in range(3):
        single = instance.infer(_request(op="act", session="a", noise=noise[i]))["actions"]
        np.testing.assert_array_equal(out["actions"][i], single)
    assert out["steering"]["op"] == "act_batch"
    assert fake.prefills == 1
    with pytest.raises(ValueError, match="N, H, D"):
        instance.infer(_request(op="act_batch", session="a", noise=_noise()))
