import json

import numpy as np
import pytest
import torch

from examples.libero.steering import common
from examples.libero.steering import env as steering_env
from examples.libero.steering import fakes
from examples.libero.steering import sac
from examples.libero.steering import train


def _args(tmp_path, **overrides):
    fields = {
        "output_dir": str(tmp_path),
        "total_episodes": 6,
        "warmup_episodes_per_task": 1,
        "batch_size": 8,
        "hidden_dims": [32, 32],
        "projection_dim": 8,
        "save_every_updates": 5,
        "log_every_updates": 5,
    }
    return train.Args(**{**fields, **overrides})


def _setup(tmp_path, args, tasks):
    spec = fakes.make_spec()
    np.save(tmp_path / "projection.npy", common.make_projection(spec))
    learner = train.Learner(args, spec, tmp_path, warmup_total=args.warmup_episodes_per_task * len(tasks))
    return spec, learner


def _run(tmp_path, args, tasks, learner, spec):
    server = fakes.FakeSteeringServer()
    client = common.SteeringClient("", 0, policy=server)
    envs = {}

    def get_env(suite, task_id):
        if (suite, task_id) not in envs:
            envs[(suite, task_id)] = steering_env.SteeringEnv(
                fakes.FakeLiberoEnv(first=3, second=12),
                f"{suite}:{task_id}",
                20,
                client,
                spec,
                make_element=fakes.make_element,
                robot_state=fakes.robot_state,
            )
        return envs[(suite, task_id)]

    counter = iter(range(args.total_episodes))
    episodes = []

    def send(kind, payload):
        if kind == "transition":
            learner.on_transition(payload)
        else:
            episodes.append(payload)
            learner.on_episode(payload)

    policy = sac.SteeringPolicy.load(tmp_path)
    train.run_worker_episodes(
        0,
        policy,
        get_env,
        tasks,
        learner.warmup_total,
        lambda: next(counter, None),
        send,
        tmp_path / "actor.pt",
        np.random.default_rng(0),
        torch.Generator().manual_seed(0),
    )
    return episodes, server


def test_worker_and_learner_loop(tmp_path):
    tasks = [("libero_10", 0), ("libero_10", 3)]
    args = _args(tmp_path)
    spec, learner = _setup(tmp_path, args, tasks)
    episodes, server = _run(tmp_path, args, tasks, learner, spec)
    learner.close()

    assert [e["episode"] for e in episodes] == list(range(6))
    assert [e["warmup"] for e in episodes] == [True, True, False, False, False, False]
    assert [(e["suite"], e["task_id"]) for e in episodes] == tasks * 3
    assert all(e["success"] for e in episodes)  # the fake goal completes at policy step 12 < 20
    # Updates start once both warmup episodes have finished and the buffer holds a batch; then one per transition.
    transitions = sum(e["decisions"] for e in episodes)
    warmup_transitions = sum(e["decisions"] for e in episodes[:2])
    assert learner.agent.num_updates == transitions - max(warmup_transitions, args.batch_size - 1)
    # Steered episodes reload the actor saved by the learner (versions are update counts).
    assert episodes[2]["actor_version"] == 0
    assert all(e["actor_version"] % args.save_every_updates == 0 for e in episodes[2:])
    # Warmup noise is standard normal in the padded dimensions as well.
    assert all(n.shape == (10, 32) for n in server.noises)

    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert sum(r["kind"] == "episode" for r in records) == 6
    assert any(r["kind"] == "update" for r in records)
    full = torch.load(tmp_path / "agent.pt", weights_only=False)
    assert full["num_updates"] == learner.agent.num_updates
    assert sac.SteeringPolicy.load(tmp_path).version == learner.agent.num_updates


def test_learner_waits_for_warmup(tmp_path):
    args = _args(tmp_path, warmup_episodes_per_task=2, batch_size=2)
    spec, learner = _setup(tmp_path, args, [("libero_10", 0)])
    s = np.zeros(spec.state_dim, np.float32)
    transition = {"s": s, "s2": s, "u": np.zeros(70), "k_index": 0, "reward": 0.0, "discount": 0.9, "done": 0.0}
    for _ in range(5):
        learner.on_transition(transition)
    assert learner.agent.num_updates == 0
    summary = {
        "episode": 0, "suite": "libero_10", "task_id": 0, "warmup": True, "actor_version": None,
        "success": False, "policy_steps": 1, "decisions": 1, "mean_k": 5.0,
    }  # fmt: skip
    learner.on_episode(summary)
    learner.on_transition(transition)
    assert learner.agent.num_updates == 0
    learner.on_episode({**summary, "episode": 1})
    learner.on_transition(transition)
    assert learner.agent.num_updates == 1
    learner.close()


def test_make_spec_uses_server_shapes(tmp_path):
    metadata = fakes.FakeSteeringServer().get_server_metadata()
    spec = train.make_spec(_args(tmp_path, horizons=[2, 4], control_dims=32), metadata, embedding_dim=16)
    assert (spec.action_horizon, spec.model_action_dim, spec.embedding_dim) == (10, 32, 16)
    assert spec.horizons == (2, 4)
    assert spec.noise_dim == 320
    with pytest.raises(ValueError, match="horizons"):
        train.make_spec(_args(tmp_path, horizons=[11]), metadata, embedding_dim=16)


def test_learner_aborts_when_q_diverges(tmp_path):
    args = _args(tmp_path, warmup_episodes_per_task=0, batch_size=2, max_abs_q=5.0)
    spec, learner = _setup(tmp_path, args, [("libero_10", 0)])
    s = np.zeros(spec.state_dim, np.float32)
    transition = {"s": s, "s2": s, "u": np.zeros(70), "k_index": 0, "reward": 1e4, "discount": 0.9, "done": 1.0}

    def feed():
        for _ in range(500):
            learner.on_transition(transition)

    with pytest.raises(RuntimeError, match="Q diverged"):
        feed()
    assert any(json.loads(line)["kind"] == "abort" for line in (tmp_path / "metrics.jsonl").read_text().splitlines())
