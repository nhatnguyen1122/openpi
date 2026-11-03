import numpy as np
import pytest

from examples.libero.steering import common
from examples.libero.steering import fakes


def test_spec_dimensions_and_roundtrip():
    spec = fakes.make_spec()
    assert spec.noise_dim == 70
    assert spec.state_dim == 8 + 8 + 1
    assert spec.num_horizons == 6
    assert common.SteeringSpec.from_dict(spec.to_dict()) == spec


@pytest.mark.parametrize(
    "overrides",
    [
        {"control_dims": 0},
        {"control_dims": 33},
        {"horizons": (0, 5)},
        {"horizons": (5, 11)},
        {"horizons": (5, 5)},
        {"noise_bound": 0.0},
        {"gamma": 0.0},
        {"num_flow_steps": 0},
        {"pad_noise": "uniform"},
    ],
)
def test_spec_rejects_invalid_values(overrides):
    with pytest.raises(ValueError, match="must"):
        fakes.make_spec(**overrides)


def test_noise_mapping_bounds_and_padding():
    spec = fakes.make_spec()
    mapper = common.NoiseMapper(spec)
    u = np.linspace(-20, 20, spec.noise_dim)
    noise = mapper.to_noise(u, np.random.default_rng(0))
    assert noise.shape == (10, 32)
    assert noise.dtype == np.float32
    assert np.all(np.abs(noise[:, :7]) <= spec.noise_bound)
    np.testing.assert_allclose(noise[:, :7].reshape(-1), spec.noise_bound * np.tanh(u), rtol=1e-6)
    assert np.std(noise[:, 7:]) > 0.5  # Gaussian padding

    zero = common.NoiseMapper(fakes.make_spec(pad_noise="zero")).to_noise(u, np.random.default_rng(0))
    assert np.all(zero[:, 7:] == 0)


def test_base_policy_noise_is_standard_normal_and_invertible():
    spec = fakes.make_spec()
    mapper = common.NoiseMapper(spec)
    noise, u = mapper.base_policy_noise(np.random.default_rng(1))
    assert u.shape == (spec.noise_dim,)
    np.testing.assert_allclose(mapper.controlled(u), noise[:, :7], rtol=1e-5, atol=1e-6)
    samples = np.stack([mapper.base_policy_noise(np.random.default_rng(i))[0] for i in range(200)])
    assert abs(samples.mean()) < 0.05
    assert abs(samples.std() - 1) < 0.05


def test_projection_is_deterministic_and_state_layout():
    spec = fakes.make_spec()
    projection = common.make_projection(spec)
    np.testing.assert_array_equal(projection, common.make_projection(spec))
    assert projection.shape == (8, 16)
    s = common.build_state(projection, np.ones(16), np.arange(8), t=26, max_steps=52)
    assert s.shape == (spec.state_dim,)
    np.testing.assert_allclose(s[:8], projection @ np.ones(16), rtol=1e-6)
    np.testing.assert_array_equal(s[8:16], np.arange(8))
    assert s[-1] == 0.5
    with pytest.raises(ValueError, match="embedding"):
        common.build_state(projection, np.ones(15), np.arange(8), 0, 1)


def test_client_embed_then_act():
    server = fakes.FakeSteeringServer()
    client = common.SteeringClient("", 0, policy=server)
    assert (client.action_horizon, client.model_action_dim) == (10, 32)
    embedding = client.embed({"observation/state": np.full(8, 2.0, np.float32)})
    assert embedding.shape == (16,)
    actions = client.act(np.zeros((10, 32)), num_steps=3)
    np.testing.assert_array_equal(actions, np.full((10, 7), 2.0))
    assert server.calls == ["embed", "act"]


def test_client_requires_steering_server():
    server = fakes.FakeSteeringServer()
    server.get_server_metadata = lambda: {"num_inference_steps": None}
    with pytest.raises(ValueError, match="does not support steering"):
        common.SteeringClient("", 0, policy=server)
