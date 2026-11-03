"""Test doubles for the policy server and LIBERO (used by the steering tests)."""

from __future__ import annotations

import types

import numpy as np

from . import common

HORIZON, MODEL_DIM, EMBED_DIM = 10, 32, 16


class FakeSteeringServer:
    """Implements the steering protocol of `Policy._infer_steering` in numpy.

    embedding = first EMBED_DIM entries of tile(state); actions[i] = noise[i, :7] + state[0].
    """

    def __init__(self):
        self.prefixes: dict[str, np.ndarray] = {}
        self.calls: list[str] = []
        self.noises: list[np.ndarray] = []

    def get_server_metadata(self) -> dict:
        return {
            "num_inference_steps": None,
            "policy_seed": 0,
            "steering": {
                "key": common.STEERING_KEY,
                "ops": ["act", "embed", "infer"],
                "action_horizon": HORIZON,
                "action_dim": MODEL_DIM,
                "max_sessions": 16,
                "default_num_steps": 10,
            },
        }

    def infer(self, obs: dict) -> dict:
        request = obs[common.STEERING_KEY]
        self.calls.append(request["op"])
        if request["op"] == "embed":
            state = np.asarray(obs["observation/state"], np.float32)
            self.prefixes[request["session"]] = state
            embedding = np.resize(state, EMBED_DIM).astype(np.float32)
            return {"prefix_embedding": embedding, "steering": {"op": "embed", "num_steps": 10}}
        noise = np.asarray(request["noise"], np.float32)
        assert noise.shape == (HORIZON, MODEL_DIM)
        if request["op"] == "infer":
            state = np.asarray(obs["observation/state"], np.float32)
            return {
                "actions": noise[:, :7] + state[0],
                "prefix_embedding": np.resize(state, EMBED_DIM).astype(np.float32),
                "steering": {"op": "infer", "num_steps": request.get("num_steps", 10)},
            }
        assert request["op"] == "act"
        assert set(obs) == {common.STEERING_KEY}
        self.noises.append(noise)
        state = self.prefixes[request["session"]]
        actions = noise[:, :7] + state[0]
        return {"actions": actions, "steering": {"op": "act", "num_steps": request.get("num_steps", 10)}}


class FakeLiberoEnv:
    """Goal = two predicates that become true after `first` and `second` policy steps; done when both hold."""

    def __init__(self, first: int = 3, second: int = 7, wait: int = 10):
        self.first, self.second, self.wait = first, second, wait
        self.steps = 0
        self.actions: list[list[float]] = []
        self.resets = 0
        self.init_states: list[object] = []
        self.env = types.SimpleNamespace(
            parsed_problem={"goal_state": [("on", "a", "b"), ("open", "c")]},
            _eval_predicate=self._eval,
        )

    def _policy_steps(self) -> int:
        return max(0, self.steps - self.wait)

    def _eval(self, predicate) -> bool:
        threshold = self.first if predicate[0] == "on" else self.second
        return self._policy_steps() >= threshold

    def _obs(self) -> dict:
        return {
            "agentview_image": np.zeros((8, 8, 3), np.uint8),
            "robot0_eye_in_hand_image": np.zeros((8, 8, 3), np.uint8),
            "robot0_eef_pos": np.full(3, float(self._policy_steps())),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.zeros(2),
        }

    def reset(self) -> dict:
        self.steps = 0
        self.resets += 1
        return self._obs()

    def set_init_state(self, state) -> dict:
        self.init_states.append(state)
        return self._obs()

    def step(self, action):
        self.steps += 1
        if self.steps > self.wait:
            self.actions.append(list(action))
        done = self._eval(("on",)) and self._eval(("open",))
        return self._obs(), float(done), done, {}

    def close(self) -> None:
        pass


def make_element(obs: dict, prompt: str) -> dict:
    return {"observation/state": robot_state(obs), "prompt": prompt}


def robot_state(obs: dict) -> np.ndarray:
    return np.concatenate([obs["robot0_eef_pos"], np.zeros(3), obs["robot0_gripper_qpos"]]).astype(np.float32)


def make_spec(**overrides) -> common.SteeringSpec:
    fields = {"action_horizon": HORIZON, "model_action_dim": MODEL_DIM, "embedding_dim": EMBED_DIM, "projection_dim": 8}
    return common.SteeringSpec(**{**fields, **overrides})
