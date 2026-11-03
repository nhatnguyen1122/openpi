import types

import pytest

from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as training_config
from scripts import serve_policy


@pytest.mark.parametrize("steps", [10, 5, 2, 1])
@pytest.mark.parametrize("default_policy", [True, False])
def test_flow_override_reaches_policy_factory(monkeypatch, steps, default_policy):
    calls = []
    monkeypatch.setattr(training_config, "get_config", lambda name: name)

    def create(config, directory, **kwargs):
        calls.append((config, directory, kwargs))
        return types.SimpleNamespace(metadata={})

    monkeypatch.setattr(policy_config, "create_trained_policy", create)
    policy = serve_policy.Default() if default_policy else serve_policy.Checkpoint("pi05_libero", "/checkpoint")
    serve_policy.create_policy(
        serve_policy.Args(env=serve_policy.EnvMode.LIBERO, policy=policy, num_inference_steps=steps, policy_seed=17)
    )
    assert len(calls) == 1
    config, _, kwargs = calls[0]
    assert config == "pi05_libero"
    assert kwargs["sample_kwargs"] == {"num_steps": steps}
    assert kwargs["rng_seed"] == 17


def test_default_preserves_model_sampling_options(monkeypatch):
    captured = {}
    monkeypatch.setattr(training_config, "get_config", lambda name: name)
    monkeypatch.setattr(policy_config, "create_trained_policy", lambda *_, **kwargs: captured.update(kwargs))
    serve_policy.create_policy(serve_policy.Args(env=serve_policy.EnvMode.LIBERO))
    assert captured["sample_kwargs"] is None
    assert captured["rng_seed"] == 0


def test_invalid_flow_steps_fail_before_loading():
    with pytest.raises(ValueError, match="must be positive"):
        serve_policy.create_policy(serve_policy.Args(num_inference_steps=0))


def test_metadata_reports_sampling_configuration(monkeypatch):
    policy = types.SimpleNamespace(metadata={"original": "metadata"})
    monkeypatch.setattr(serve_policy, "create_policy", lambda _: policy)
    received = {}

    def server(**kwargs):
        received.update(kwargs)
        return types.SimpleNamespace(serve_forever=lambda: None)

    monkeypatch.setattr(websocket_policy_server, "WebsocketPolicyServer", server)
    monkeypatch.setattr(serve_policy.socket, "gethostname", lambda: "localhost")
    monkeypatch.setattr(serve_policy.socket, "gethostbyname", lambda _: "127.0.0.1")
    args = serve_policy.Args(
        policy=serve_policy.Checkpoint("pi05_libero", "/checkpoint"), num_inference_steps=2, policy_seed=17
    )
    serve_policy.main(args)
    assert received["metadata"]["num_inference_steps"] == 2
    assert received["metadata"]["policy_seed"] == 17
    assert received["metadata"]["policy_config"] == "pi05_libero"
    assert policy.metadata == {"original": "metadata"}
