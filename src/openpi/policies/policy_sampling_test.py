import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model
from openpi.policies import policy
from openpi.shared import nnx_utils


@pytest.mark.parametrize("steps", [10, 5, 2, 1])
def test_sampling_steps_reach_model_call(monkeypatch, steps):
    received = []

    def sample_actions(_rng, _observation, **kwargs):
        received.append(kwargs)
        return jnp.zeros((1, 10, 7))

    monkeypatch.setattr(nnx_utils, "module_jit", lambda function: function)
    monkeypatch.setattr(model.Observation, "from_dict", lambda data: data)
    instance = policy.Policy(
        types.SimpleNamespace(sample_actions=sample_actions), sample_kwargs={"num_steps": steps}, rng=jax.random.key(17)
    )
    result = instance.infer({"state": np.zeros(8)})
    assert received == [{"num_steps": steps}]
    assert result["actions"].shape == (10, 7)


def test_interleaved_requests_keep_sampler_and_noise_isolated(monkeypatch):
    seen = []

    def sample_actions(rng, observation, **kwargs):
        seen.append((np.asarray(jax.random.key_data(rng)).tolist(), kwargs))
        return jax.random.normal(rng, (1, 10, 7))

    monkeypatch.setattr(nnx_utils, "module_jit", lambda f: f)
    monkeypatch.setattr(model.Observation, "from_dict", lambda data: data)
    instance = policy.Policy(types.SimpleNamespace(sample_actions=sample_actions), sample_kwargs={"num_steps": 10})
    a = {"state": np.zeros(8), "_openpi_inference": {"num_steps": 2, "seed": 123}}
    b = {"state": np.zeros(8), "_openpi_inference": {"num_steps": 5, "seed": 456}}
    first = instance.infer(a)
    instance.infer(b)
    instance.infer({"state": np.zeros(8)})
    again = instance.infer(a)
    np.testing.assert_array_equal(first["actions"], again["actions"])
    assert [s[1]["num_steps"] for s in seen] == [2, 5, 10, 2]
    assert first["sampling_options"] == a["_openpi_inference"]
    assert "_openpi_inference" in a  # Caller data is not consumed.


@pytest.mark.parametrize(
    "options",
    [
        {"num_steps": 0, "seed": 0},
        {"num_steps": True, "seed": 0},
        {"num_steps": 1, "seed": -1},
        {"num_steps": 1, "seed": 2**32},
        {"num_steps": 1},
        {"num_steps": 1, "seed": 0, "unknown": 1},
    ],
)
def test_invalid_request_controls_rejected(monkeypatch, options):
    monkeypatch.setattr(nnx_utils, "module_jit", lambda f: f)
    instance = policy.Policy(types.SimpleNamespace(sample_actions=lambda *_: None))
    with pytest.raises(ValueError, match="Sampling options|Request"):
        instance.infer({"state": np.zeros(8), "_openpi_inference": options})
