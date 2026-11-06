import collections
from collections.abc import Sequence
import dataclasses
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy

# Request key for noise-steering operations (see `Policy._infer_steering`).
STEERING_KEY = "_openpi_steering"
_STEERING_KEYS = {
    "embed": {"op", "session"},
    "act": {"op", "session", "noise", "num_steps"},
    "act_batch": {"op", "session", "noise", "num_steps"},
    "infer": {"op", "noise", "num_steps"},
}


_MAX_STEERING_BATCH = 64


@dataclasses.dataclass(frozen=True)
class _SteeringPrefix:
    state: np.ndarray  # Transformed (normalized) state, needed by the output transforms.
    observation: _model.Observation
    prefix_mask: Any
    kv_cache: Any


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        max_steering_sessions: int = 16,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
            max_steering_sessions: Number of cached steering prefixes (one per client session). Each holds a KV cache
                on the accelerator; the least recently used one is evicted first.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = jax.random.key(0) if rng is None else rng

        self._prefill_steering = None
        self._sample_from_prefix = None
        if not self._is_pytorch_model and all(
            hasattr(model, name) for name in ("prefill_steering", "sample_actions_from_prefix")
        ):
            self._prefill_steering = nnx_utils.module_jit(model.prefill_steering)
            self._sample_from_prefix = nnx_utils.module_jit(model.sample_actions_from_prefix)
        if max_steering_sessions < 1:
            raise ValueError("max_steering_sessions must be positive")
        self._max_steering_sessions = max_steering_sessions
        self._steering_sessions: collections.OrderedDict[str, _SteeringPrefix] = collections.OrderedDict()

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Request-local controls let concurrent evaluation clients share one model
        # without changing each other's sampler settings or random-number stream.
        obs = dict(obs)
        options = obs.pop("_openpi_inference", None)
        steering = obs.pop(STEERING_KEY, None)
        if steering is not None:
            if options is not None or noise is not None:
                raise ValueError("Steering requests carry their own noise and cannot be combined with other options")
            return self._infer_steering(obs, steering)
        if options is not None:
            if self._is_pytorch_model:
                raise ValueError("Request sampling options currently require a JAX policy")
            if not isinstance(options, dict) or set(options) != {"num_steps", "seed"}:
                raise ValueError("Sampling options must contain exactly num_steps and seed")
            if type(options["num_steps"]) is not int or not 1 <= options["num_steps"] <= 100:
                raise ValueError("Request num_steps must be an integer in [1, 100]")
            if type(options["seed"]) is not int or not 0 <= options["seed"] < 2**32:
                raise ValueError("Request seed must be a uint32 integer")
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            if options is None:
                self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
            else:
                sample_rng_or_pytorch_device = jax.random.key(options["seed"])
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if options is not None:
            sample_kwargs["num_steps"] = options["num_steps"]
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if options is not None:
            outputs["sampling_options"] = dict(options)
        return outputs

    def _infer_steering(self, obs: dict, request: dict) -> dict:
        """Serves noise-steering requests for an external RL actor.

        Ops:
            embed: observation -> pooled prefix embedding. The prefix (KV cache) is kept under `session`.
            act: noise [+ num_steps] -> actions, reusing the prefix of the last `embed` in `session`. No observation.
            act_batch: like act for noise [N, H, D] -> actions [N, ...]: N candidate chunks from one prefix.
            infer: observation + noise -> actions and embedding in one call (stateless).

        Noise lives in the model's normalized action space with shape [action_horizon, action_dim], including padded
        dimensions. Given the same observation and noise, the returned actions are deterministic.
        """
        if self._prefill_steering is None:
            raise ValueError("This policy does not support steering requests (JAX pi0/pi05 models only)")
        if not isinstance(request, dict) or request.get("op") not in _STEERING_KEYS:
            raise ValueError(f"Steering op must be one of {sorted(_STEERING_KEYS)}")
        op = request["op"]
        required = _STEERING_KEYS[op] - {"num_steps"}
        if not required <= set(request) <= _STEERING_KEYS[op]:
            raise ValueError(f"Steering op {op!r} takes keys {sorted(_STEERING_KEYS[op])}, got {sorted(request)}")
        if "session" in request and (not isinstance(request["session"], str) or not 0 < len(request["session"]) <= 128):
            raise ValueError("Steering session must be a nonempty string of at most 128 characters")
        num_steps = request.get("num_steps", self._sample_kwargs.get("num_steps", 10))
        if type(num_steps) is not int or not 1 <= num_steps <= 100:
            raise ValueError("Steering num_steps must be an integer in [1, 100]")

        start_time = time.monotonic()
        embedding = None
        if op in ("act", "act_batch"):
            if obs:
                raise ValueError("Steering act requests reuse the session prefix and must not include observations")
            prefix = self._steering_sessions.get(request["session"])
            if prefix is None:
                raise ValueError(f"Unknown steering session {request['session']!r}; send an embed request first")
            self._steering_sessions.move_to_end(request["session"])
        else:
            inputs = self._input_transform(jax.tree.map(lambda x: x, obs))
            state = np.asarray(inputs["state"])
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            embedding, observation, prefix_mask, kv_cache = self._prefill_steering(_model.Observation.from_dict(inputs))
            embedding = np.asarray(embedding[0], dtype=np.float32)
            prefix = _SteeringPrefix(state, observation, prefix_mask, kv_cache)

        if op == "embed":
            self._steering_sessions[request["session"]] = prefix
            self._steering_sessions.move_to_end(request["session"])
            while len(self._steering_sessions) > self._max_steering_sessions:
                self._steering_sessions.popitem(last=False)
            outputs = {"prefix_embedding": embedding}
        elif op == "act_batch":
            noise = np.asarray(request["noise"], dtype=np.float32)
            if noise.ndim != 3 or not 1 <= len(noise) <= _MAX_STEERING_BATCH:
                raise ValueError(f"act_batch noise must have shape [N, H, D] with 1 <= N <= {_MAX_STEERING_BATCH}")
            noise = np.stack([self._validate_steering_noise(n) for n in noise])
            actions = np.asarray(
                self._sample_from_prefix(
                    prefix.observation, prefix.prefix_mask, prefix.kv_cache, jnp.asarray(noise), num_steps=num_steps
                )
            )
            outputs = {
                "actions": np.stack(
                    [self._output_transform({"state": prefix.state, "actions": chunk})["actions"] for chunk in actions]
                )
            }
        else:
            actions = self._sample_from_prefix(
                prefix.observation,
                prefix.prefix_mask,
                prefix.kv_cache,
                jnp.asarray(self._validate_steering_noise(request["noise"]))[np.newaxis, ...],
                num_steps=num_steps,
            )
            outputs = self._output_transform({"state": prefix.state, "actions": np.asarray(actions[0])})
            if embedding is not None:
                outputs["prefix_embedding"] = embedding
        outputs["policy_timing"] = {"infer_ms": (time.monotonic() - start_time) * 1000}
        outputs["steering"] = {"op": op, "num_steps": num_steps}
        return outputs

    def _validate_steering_noise(self, noise: Any) -> np.ndarray:
        noise = np.asarray(noise, dtype=np.float32)
        expected = (self._model.action_horizon, self._model.action_dim)
        if noise.shape == (1, *expected):
            noise = noise[0]
        if noise.shape != expected:
            raise ValueError(f"Steering noise must have shape {expected}, got {noise.shape}")
        if not np.isfinite(noise).all():
            raise ValueError("Steering noise must be finite")
        return noise

    @property
    def supports_request_sampling(self) -> bool:
        return not self._is_pytorch_model

    @property
    def steering_metadata(self) -> dict[str, Any] | None:
        """Describes the steering protocol to clients, or None if unsupported."""
        if self._prefill_steering is None:
            return None
        return {
            "key": STEERING_KEY,
            "ops": sorted(_STEERING_KEYS),
            "action_horizon": int(self._model.action_horizon),
            "action_dim": int(self._model.action_dim),
            "max_sessions": self._max_steering_sessions,
            "default_num_steps": int(self._sample_kwargs.get("num_steps", 10)),
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
