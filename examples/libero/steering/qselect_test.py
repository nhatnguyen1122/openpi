import numpy as np
import pytest
import torch

from examples.libero.steering import common
from examples.libero.steering import env as steering_env
from examples.libero.steering import fakes
from examples.libero.steering import qselect


def _setup(**config):
    spec = fakes.make_spec(horizons=(config.get("segment", 10),))
    cfg = qselect.QSelectConfig(**{"num_candidates": 4, "ensemble": 3, "hidden_dims": (32,), **config})
    client = common.SteeringClient("", 0, policy=fakes.FakeSteeringServer())
    raw = fakes.FakeLiberoEnv(first=8, second=30)
    env = steering_env.SteeringEnv(
        raw, "task", 60, client, spec, make_element=fakes.make_element, robot_state=fakes.robot_state
    )
    selector = qselect.Selector(spec, cfg, common.make_projection(spec))
    return spec, cfg, env, raw, selector


def test_full_chunk_mode_executes_chosen_candidates_and_links_next_options():
    spec, cfg, env, raw, selector = _setup()
    transitions = []
    summary = qselect.run_episode(env, selector, np.random.default_rng(0), mode="random", emit=transitions.append)
    assert summary["success"]
    assert summary["policy_steps"] == 30
    assert summary["decisions"] == 3
    assert summary["continue_fraction"] == 0
    executed = np.asarray(raw.actions)
    for i, t in enumerate(transitions):
        np.testing.assert_allclose(t["a"].reshape(10, 7), executed[10 * i : 10 * (i + 1)])
        assert len(t["next_a"]) == (0 if t["done"] else cfg.num_candidates)
    assert transitions[-1]["done"] == 1.0
    # The next options of one decision contain the action executed at the following decision.
    assert any(np.allclose(transitions[1]["a"], option) for option in transitions[0]["next_a"])


def test_continue_mode_reuses_second_half_of_previous_chunk():
    spec, cfg, env, raw, selector = _setup(segment=5, allow_continue=True, continue_prob=1.0)
    transitions = []
    summary = qselect.run_episode(env, selector, np.random.default_rng(1), mode="random", emit=transitions.append)
    # Always continuing when possible = fresh, continue, fresh, ... i.e. the k = 10 base policy.
    assert summary["decisions"] == 6
    assert summary["continue_fraction"] == pytest.approx(0.5)
    assert [len(t["next_a"]) for t in transitions[:-1]] == [5, 4, 5, 4, 5]
    server_noise = np.asarray(env.client._policy.noises)  # noqa: SLF001
    executed = np.asarray(raw.actions)
    chunk = executed[:10] - executed[0] + server_noise[0, 0, :7]  # actions = noise + constant offset per prefix
    assert np.allclose(np.diff(executed[:10], axis=0), np.diff(chunk, axis=0))


def test_greedy_choice_follows_critic_scores(monkeypatch):
    _, _, _, _, selector = _setup()
    options = np.zeros((4, 10, 7), np.float32)
    monkeypatch.setattr(selector, "scores", lambda s, o: np.array([0.1, 0.9, 0.3, 0.2]))
    assert selector.choose(np.zeros(1), options, np.random.default_rng(0), has_continue=False, mode="greedy") == 1


def test_update_regresses_to_clipped_targets_and_ignores_empty_next_sets():
    spec, cfg, *_ = _setup()
    agent = qselect.QSelectAgent(spec, cfg)
    buffer = qselect.ReplayBuffer(spec, cfg, 64)
    rng = np.random.default_rng(0)
    for i in range(64):
        done = i % 2 == 0
        buffer.add(
            {
                "s": rng.standard_normal(spec.state_dim),
                "s2": rng.standard_normal(spec.state_dim),
                "a": rng.standard_normal(cfg.segment_dim),
                "next_a": np.zeros((0 if done else cfg.num_options, cfg.segment_dim)),
                "reward": 5.0 if done else 0.0,  # above q_max: targets must be clipped
                "discount": 0.9,
                "done": float(done),
            }
        )
    for _ in range(300):
        info = agent.update(buffer.sample(64))
    assert np.isfinite(info["q_mean"])
    batch = buffer.sample(64)
    with torch.no_grad():
        q = agent.critic(batch["s"], batch["a"]).mean(0)
    assert float(q[batch["done"] == 1].mean()) == pytest.approx(cfg.q_max, abs=0.3)


def test_payload_round_trip(tmp_path):
    spec, cfg, *_ = _setup()
    agent = qselect.QSelectAgent(spec, cfg)
    np.save(tmp_path / "projection.npy", common.make_projection(spec))
    qselect.save_atomic(agent.payload(), tmp_path / "critic.pt")
    loaded = qselect.Selector.load(tmp_path / "critic.pt")
    s, options = np.zeros(spec.state_dim, np.float32), np.ones((3, 10, 7), np.float32)
    selector = qselect.Selector(spec, cfg, common.make_projection(spec))
    selector.critic.load_state_dict(agent.critic.state_dict())
    np.testing.assert_allclose(loaded.scores(s, options), selector.scores(s, options))


def test_recording_keeps_one_frame_per_step_and_writes_a_video(tmp_path):
    _, _, env, _, selector = _setup()
    env.record = True
    summary = qselect.run_episode(env, selector, np.random.default_rng(0), mode="random")
    assert len(env.frames) == summary["policy_steps"] + 1  # initial frame plus one per executed action
    name = common.video_name("libero_10", "put it there", 3, success=summary["success"])
    assert name == "rollout_libero_10_put_it_there_003_success.mp4"
    common.save_video(tmp_path / name, env.frames)
    assert (tmp_path / name).stat().st_size > 0


def test_state_without_embedding_keeps_only_robot_state_and_time():
    spec, cfg, _, _, full = _setup()
    _, _, _, _, ablated = _setup(use_embedding=False)
    e, q = np.ones(full.projection.shape[1], np.float32), np.arange(8, dtype=np.float32)
    s_full, s_abl = full.state(e, q, 3, 10), ablated.state(e, q, 3, 10)
    k = full.projection.shape[0]
    assert s_abl.shape == s_full.shape
    assert not s_abl[:k].any()
    assert s_full[:k].any()
    np.testing.assert_array_equal(s_abl[k:], s_full[k:])
    assert qselect.QSelectConfig.from_dict({**cfg.to_dict(), "use_embedding": False}).use_embedding is False
    legacy = {k: v for k, v in cfg.to_dict().items() if k != "use_embedding"}  # checkpoints from before the flag
    assert qselect.QSelectConfig.from_dict(legacy).use_embedding is True
