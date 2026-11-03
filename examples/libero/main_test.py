import dataclasses
import json
import types

import numpy as np
from openpi_client import websocket_client_policy
import pytest
import tyro

from examples.libero import main


@pytest.fixture
def simulated_evaluation(monkeypatch):
    visited = []
    requests = []
    actions = []
    simulation = types.SimpleNamespace(visited=visited, requests=requests, actions=actions, done_after=1)
    observation = {
        "agentview_image": np.full((224, 224, 3), 127, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((224, 224, 3), 63, dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
    }
    suite = types.SimpleNamespace(
        n_tasks=2,
        get_task=lambda task_id: types.SimpleNamespace(task_id=task_id, language=f"test task {task_id}"),
        get_task_init_states=lambda _: np.arange(50),
    )
    monkeypatch.setattr(main.benchmark, "get_benchmark_dict", lambda: {"libero_spatial": lambda: suite})

    def make_env(task, *_):
        episode_steps = 0

        def set_init_state(state):
            nonlocal episode_steps
            episode_steps = 0
            visited.append((task.task_id, int(state)))
            return observation

        def step(action):
            nonlocal episode_steps
            episode_steps += 1
            actions.append(action)
            return observation, 1, episode_steps >= simulation.done_after, {}

        env = types.SimpleNamespace(
            reset=lambda: None,
            set_init_state=set_init_state,
            step=step,
            close=lambda: None,
        )
        return env, f"test task {task.task_id}"

    monkeypatch.setattr(main, "_get_libero_env", make_env)

    def infer(element):
        requests.append(element)
        return {"actions": np.arange(70).reshape(10, 7), "server_timing": {"infer_ms": 125}}

    policy = types.SimpleNamespace(
        infer=infer, get_server_metadata=lambda: {"num_inference_steps": 10, "policy_seed": 0}
    )
    monkeypatch.setattr(websocket_client_policy, "WebsocketClientPolicy", lambda *_: policy)
    return simulation


def test_extend_ten_trials_to_fifty(tmp_path, simulated_evaluation):
    results = tmp_path / "episodes.jsonl"
    args = main.Args(
        num_trials_per_task=10,
        num_steps_wait=0,
        save_video=False,
        video_out_path=str(tmp_path / "videos"),
        results_path=str(results),
    )
    main.eval_libero(args)
    main.eval_libero(dataclasses.replace(args, start_episode_idx=10, num_trials_per_task=40))

    records = [json.loads(line) for line in results.read_text().splitlines()]
    expected = {(task_id, episode_idx) for task_id in range(2) for episode_idx in range(50)}
    assert len(records) == len(expected)
    assert {(r["task_id"], r["episode_idx"]) for r in records} == expected
    assert len(simulated_evaluation.visited) == len(expected)
    assert set(simulated_evaluation.visited) == expected
    assert all(r["success"] for r in records)


def test_extension_cannot_exceed_available_initial_states(tmp_path, simulated_evaluation):
    args = main.Args(start_episode_idx=10, num_trials_per_task=41, video_out_path=str(tmp_path))
    with pytest.raises(ValueError, match="only has 50 initial states"):
        main.eval_libero(args)
    assert not simulated_evaluation.visited


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("original", "test task 0"), ("empty", ""), ("paraphrase", "rewritten task 0"), ("wrong_task", "test task 1")],
)
def test_instruction_changes_only_policy_prompt(tmp_path, simulated_evaluation, mode, expected):
    mapping = tmp_path / "paraphrases.json"
    mapping.write_text(
        json.dumps(
            {
                "suite": "libero_spatial",
                "instructions": {"test task 0": "rewritten task 0", "test task 1": "rewritten task 1"},
            }
        )
    )
    results = tmp_path / "episodes.jsonl"
    main.eval_libero(
        main.Args(
            task_ids=[0],
            num_trials_per_task=1,
            num_steps_wait=0,
            save_video=False,
            video_out_path=str(tmp_path),
            results_path=str(results),
            instruction_mode=mode,
            instruction_map_path=str(mapping) if mode == "paraphrase" else None,
        )
    )
    assert simulated_evaluation.requests[0]["prompt"] == expected
    record = json.loads(results.read_text())
    assert record["task"] == "test task 0"
    assert record["policy_prompt"] == expected
    assert record["instruction_source_task_id"] == (1 if mode == "wrong_task" else None)


@pytest.mark.parametrize(
    ("mode", "base", "wrist"), [("both", 127, 63), ("no_wrist", 127, 0), ("no_third_person", 0, 63)]
)
def test_camera_blackout_preserves_other_view(tmp_path, simulated_evaluation, mode, base, wrist):
    main.eval_libero(
        main.Args(
            task_ids=[0],
            num_trials_per_task=1,
            num_steps_wait=0,
            save_video=False,
            video_out_path=str(tmp_path),
            camera_mode=mode,
        )
    )
    request = simulated_evaluation.requests[0]
    assert np.all(request["observation/image"] == base)
    assert np.all(request["observation/wrist_image"] == wrist)
    assert request["observation/image"].shape == (224, 224, 3)
    assert request["observation/wrist_image"].dtype == np.uint8


@pytest.mark.parametrize(("replan", "calls", "prefix"), [(1, 12, 1), (5, 3, 5), ("full", 2, 10)])
def test_replanning_executes_requested_chunk_prefix(tmp_path, simulated_evaluation, replan, calls, prefix):
    simulated_evaluation.done_after = 12
    results = tmp_path / "episodes.jsonl"
    main.eval_libero(
        main.Args(
            task_ids=[0],
            num_trials_per_task=1,
            num_steps_wait=0,
            save_video=False,
            video_out_path=str(tmp_path),
            results_path=str(results),
            replan_steps=replan,
        )
    )
    assert len(simulated_evaluation.requests) == calls
    assert len(simulated_evaluation.actions) == 12
    chunk = np.arange(70).reshape(10, 7)
    expected = np.concatenate([chunk[:prefix]] * calls)[:12]
    np.testing.assert_array_equal(simulated_evaluation.actions, expected)
    row = json.loads(results.read_text())
    assert row["predicted_chunk_lengths"] == [10] * calls
    assert row["queued_chunk_lengths"] == [prefix] * calls


def test_warmups_are_unscored_and_server_timing_is_recorded(tmp_path, simulated_evaluation):
    results = tmp_path / "episodes.jsonl"
    main.eval_libero(
        main.Args(
            task_ids=[0],
            num_trials_per_task=1,
            num_steps_wait=0,
            save_video=False,
            video_out_path=str(tmp_path),
            results_path=str(results),
            warmup_requests=2,
            expected_num_inference_steps=10,
        )
    )
    assert len(simulated_evaluation.requests) == 3
    assert len(simulated_evaluation.actions) == 1
    row = json.loads(results.read_text())
    assert len(row["warmup_seconds"]) == 2
    assert len(row["inference_seconds"]) == 1
    assert row["server_inference_seconds"] == [0.125]
    assert row["policy_seed"] == 0


def test_wrong_flow_server_is_rejected_before_rollout(tmp_path, simulated_evaluation):
    with pytest.raises(ValueError, match="Expected 5 flow steps"):
        main.eval_libero(main.Args(video_out_path=str(tmp_path), expected_num_inference_steps=5))
    assert not simulated_evaluation.visited


def test_wrong_policy_seed_is_rejected_before_rollout(tmp_path, simulated_evaluation):
    with pytest.raises(ValueError, match="Expected policy seed 17"):
        main.eval_libero(main.Args(video_out_path=str(tmp_path), expected_policy_seed=17))
    assert not simulated_evaluation.visited


@pytest.mark.parametrize("value", ["1", "5", "full"])
def test_replan_cli_accepts_integer_or_full(value):
    args = tyro.cli(main.Args, args=["--replan-steps", value])
    assert args.replan_steps == ("full" if value == "full" else int(value))


def test_checked_in_goal_paraphrases_cover_all_tasks():
    from pathlib import Path

    suite = main.benchmark.get_benchmark_dict()["libero_goal"]()
    args = main.Args(
        task_suite_name="libero_goal",
        instruction_mode="paraphrase",
        instruction_map_path=str(Path(__file__).parent / "instructions/libero_goal_paraphrases.json"),
    )
    prompts = main._get_policy_prompts(suite, args)  # noqa: SLF001
    assert len(prompts) == 10
    assert all(prompts[i] != suite.get_task(i).language and prompts[i] for i in range(10))


def test_request_sampling_reproducible_across_warmup_and_ports(tmp_path, simulated_evaluation, monkeypatch):
    def infer(element):
        simulated_evaluation.requests.append(element)
        return {
            "actions": np.zeros((10, 7)),
            "server_timing": {"infer_ms": 1},
            "sampling_options": dict(element["_openpi_inference"]),
        }

    fake = types.SimpleNamespace(
        infer=infer,
        get_server_metadata=lambda: {
            "num_inference_steps": 10,
            "policy_seed": 0,
            "request_sampling_options": ["num_steps", "seed"],
        },
    )
    monkeypatch.setattr(websocket_client_policy, "WebsocketClientPolicy", lambda *_: fake)
    args = main.Args(
        task_ids=[0],
        num_trials_per_task=1,
        num_steps_wait=0,
        save_video=False,
        video_out_path=str(tmp_path),
        results_path=str(tmp_path / "results.jsonl"),
        request_num_inference_steps=2,
        request_seed=0,
        expected_num_inference_steps=2,
        warmup_requests=2,
    )
    main.eval_libero(args)
    first = simulated_evaluation.requests[-1]["_openpi_inference"]
    assert all(r["_openpi_inference"] == first for r in simulated_evaluation.requests)
    main.eval_libero(dataclasses.replace(args, port=8002, warmup_requests=0))
    assert simulated_evaluation.requests[-1]["_openpi_inference"] == first
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert all(r["num_inference_steps"] == 2 for r in rows)
    assert all(r["rng_protocol"].startswith("sha256-v1:") for r in rows)


def test_request_sampling_rejects_old_server(tmp_path, simulated_evaluation):
    args = main.Args(
        task_ids=[0], num_trials_per_task=1, video_out_path=str(tmp_path), request_num_inference_steps=2, request_seed=0
    )
    with pytest.raises(ValueError, match="does not support"):
        main.eval_libero(args)
    assert not simulated_evaluation.visited


def _steering_checkpoint(tmp_path, horizons=(3, 4)):
    from examples.libero.steering import common
    from examples.libero.steering import fakes
    from examples.libero.steering import sac

    spec = fakes.make_spec(horizons=horizons)
    agent = sac.SACAgent(spec, sac.SACConfig(hidden_dims=(16, 16)))
    np.save(tmp_path / "projection.npy", common.make_projection(spec))
    sac.save_atomic(agent.actor_payload(version=7), tmp_path / "actor.pt")
    return tmp_path / "actor.pt"


def test_steering_chooses_noise_and_replan_steps(tmp_path, simulated_evaluation, monkeypatch):
    from examples.libero.steering import fakes

    server = fakes.FakeSteeringServer()
    monkeypatch.setattr(websocket_client_policy, "WebsocketClientPolicy", lambda *_: server)
    simulated_evaluation.done_after = 9
    results = tmp_path / "episodes.jsonl"
    args = main.Args(
        task_ids=[0],
        num_trials_per_task=2,
        num_steps_wait=0,
        save_video=False,
        video_out_path=str(tmp_path),
        results_path=str(results),
        steering_checkpoint=str(_steering_checkpoint(tmp_path)),
    )
    main.eval_libero(args)

    records = [json.loads(line) for line in results.read_text().splitlines()]
    assert len(records) == 2
    for record in records:
        assert record["replan_steps"] == "steering"
        assert record["steering_actor_version"] == 7
        assert set(record["steering_chosen_k"]) <= {3, 4}
        assert record["queued_chunk_lengths"] == record["steering_chosen_k"]
        assert sum(record["steering_chosen_k"][:-1]) < 9 <= sum(record["steering_chosen_k"])
    assert server.calls == ["embed", "act"] * len(server.noises)
    # Deterministic actor + episode-keyed padding: both episodes differ only through their initial-state index.
    first_episode = len(records[0]["steering_chosen_k"])
    assert not np.array_equal(server.noises[0][:, 7:], server.noises[first_episode][:, 7:])
    np.testing.assert_array_equal(server.noises[0][:, :7], server.noises[first_episode][:, :7])


def test_steering_rejects_request_sampling(tmp_path, simulated_evaluation, monkeypatch):
    from examples.libero.steering import fakes

    server = fakes.FakeSteeringServer()
    metadata = {**server.get_server_metadata(), "request_sampling_options": ["num_steps", "seed"]}
    server.get_server_metadata = lambda: metadata
    monkeypatch.setattr(websocket_client_policy, "WebsocketClientPolicy", lambda *_: server)
    args = main.Args(
        video_out_path=str(tmp_path),
        steering_checkpoint=str(_steering_checkpoint(tmp_path)),
        request_num_inference_steps=5,
        request_seed=0,
    )
    with pytest.raises(ValueError, match="Steering chooses its own noise"):
        main.eval_libero(args)
