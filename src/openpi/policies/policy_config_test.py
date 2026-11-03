import types

import jax
import numpy as np
import pytest

from openpi import transforms
from openpi.models import model
from openpi.policies import policy
from openpi.policies import policy_config


@pytest.mark.parametrize("seed", [0, 17])
def test_jax_factory_passes_explicit_rng_and_flow_steps(monkeypatch, tmp_path, seed):
    monkeypatch.setattr(model, "restore_params", lambda *_, **__: {})
    data = types.SimpleNamespace(
        data_transforms=transforms.Group(), model_transforms=transforms.Group(), use_quantile_norm=False
    )
    config = types.SimpleNamespace(
        model=types.SimpleNamespace(load=lambda _: object()),
        data=types.SimpleNamespace(create=lambda *_: data),
        assets_dirs=[],
        policy_metadata={},
    )
    received = {}
    monkeypatch.setattr(policy, "Policy", lambda *_, **kwargs: received.update(kwargs))
    policy_config.create_trained_policy(config, tmp_path, norm_stats={}, sample_kwargs={"num_steps": 2}, rng_seed=seed)
    np.testing.assert_array_equal(jax.random.key_data(received["rng"]), jax.random.key_data(jax.random.key(seed)))
    assert received["sample_kwargs"] == {"num_steps": 2}
