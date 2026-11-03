import dataclasses

import jax
import numpy as np
import pytest

from openpi.models import gemma as _gemma
from openpi.models import pi0_config
from openpi.shared import nnx_utils


@pytest.fixture(scope="module")
def tiny_pi05():
    config = pi0_config.Pi0Config(
        pi05=True, action_horizon=10, paligemma_variant="dummy", action_expert_variant="dummy"
    )
    model = config.create(jax.random.key(0))
    return config, model


def test_prefix_path_matches_sample_actions(tiny_pi05):
    config, model = tiny_pi05
    observation = config.fake_obs(1)
    noise = jax.random.normal(jax.random.key(3), (1, config.action_horizon, config.action_dim))

    reference = nnx_utils.module_jit(model.sample_actions)(jax.random.key(1), observation, noise=noise, num_steps=3)
    embedding, prepared, prefix_mask, kv_cache = nnx_utils.module_jit(model.prefill_steering)(observation)
    from_prefix = nnx_utils.module_jit(model.sample_actions_from_prefix)
    actions = from_prefix(prepared, prefix_mask, kv_cache, noise, num_steps=3)

    np.testing.assert_allclose(np.asarray(actions), np.asarray(reference), rtol=1e-5, atol=1e-5)
    assert embedding.shape == (1, _gemma.get_config(config.paligemma_variant).width)
    assert np.isfinite(np.asarray(embedding)).all()

    # The cached prefix can be reused: same noise -> same actions, other noise -> other actions.
    again = from_prefix(prepared, prefix_mask, kv_cache, noise, num_steps=3)
    np.testing.assert_array_equal(np.asarray(again), np.asarray(actions))
    other = from_prefix(prepared, prefix_mask, kv_cache, -noise, num_steps=3)
    assert not np.allclose(np.asarray(other), np.asarray(actions))


def test_embedding_ignores_masked_prompt_tokens(tiny_pi05):
    # Images are not used here: a randomly initialized SigLIP has a zero-initialized output head.
    config, model = tiny_pi05
    observation = config.fake_obs(1)
    mask = observation.tokenized_prompt_mask.at[:, -1].set(False)
    observation = dataclasses.replace(observation, tokenized_prompt_mask=mask)

    def with_token(index, token):
        prompt = observation.tokenized_prompt.at[:, index].set(token)
        return dataclasses.replace(observation, tokenized_prompt=prompt)

    prefill = nnx_utils.module_jit(model.prefill_steering)
    embedding = np.asarray(prefill(observation)[0])
    np.testing.assert_allclose(np.asarray(prefill(with_token(-1, 7))[0]), embedding, rtol=1e-5, atol=1e-5)
    assert not np.allclose(np.asarray(prefill(with_token(0, 7))[0]), embedding)
