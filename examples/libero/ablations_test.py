import dataclasses
import json
import subprocess

import pytest

from examples.libero import ablations


def test_default_matrix_has_thirteen_controlled_runs():
    runs = ablations.build_runs()
    assert len({r.name for r in runs}) == len(runs) == 13
    baseline = next(r for r in runs if r.name == "goal_baseline")
    for r in runs:
        if r.name.startswith("goal_flow_"):
            assert dataclasses.replace(r, name=baseline.name, flow_steps=10) == baseline
    assert {r.flow_steps for r in runs if r.suite == "libero_goal"} == {10, 5, 2, 1}
    assert {r.replan for r in runs if r.suite == "libero_10"} == {1, 5, "full"}


def test_spatial_flow_matrix_keeps_goal_instruction_controls():
    runs = ablations.build_runs("libero_spatial")
    assert len(runs) == 14
    assert {r.flow_steps for r in runs if r.suite == "libero_spatial"} == {10, 5, 2, 1}
    assert {r.instruction for r in runs if r.suite == "libero_goal"} == {
        "original",
        "empty",
        "paraphrase",
        "wrong_task",
    }


def test_plan_assigns_matching_gpu_and_port_and_unique_outputs(tmp_path):
    args = ablations.parse_args(["--output-dir", str(tmp_path / "runs"), "--gpus", "1", "2", "--ports", "8001", "8002"])
    plan = ablations.make_plan(args)
    assert plan["protocol"]["total_episodes"] == 1300
    assert plan["runs"][0]["name"] == "libero10_replan_1"
    destinations = set()
    for index, server in enumerate(plan["servers"]):
        assert server["environment"]["CUDA_VISIBLE_DEVICES"] == str(index + 1)
        assert server["environment"]["JAX_PLATFORMS"] == "cuda"
        assert server["command"][server["command"].index("--port") + 1] == str(8001 + index)
        for run_spec in plan["runs"]:
            run = ablations.Run(**run_spec)
            launch = ablations.client_launch(run, args, server["gpu"], server["port"])
            command = launch["client_command"]
            assert command[command.index("--args.port") + 1] == str(server["port"])
            assert command[command.index("--args.num-trials-per-task") + 1] == "10"
            assert command[command.index("--args.request-num-inference-steps") + 1] == str(run.flow_steps)
            assert "--args.task-ids" not in command
            assert launch["client_environment"]["JAX_PLATFORMS"] == "cpu"
            assert launch["client_environment"]["MUJOCO_EGL_DEVICE_ID"] == str(server["gpu"])
            destinations.add(launch["directory"])
    assert len(destinations) == 13


@pytest.mark.parametrize(
    ("gpus", "ports", "clients"),
    [
        (["1", "1"], ["8001", "8002"], "2"),
        (["1", "2"], ["8001"], "2"),
        (["1", "2"], ["8001", "8001"], "2"),
        (["1"], ["8001"], "0"),
    ],
)
def test_invalid_resources_rejected(tmp_path, gpus, ports, clients):
    args = ablations.parse_args(
        ["--output-dir", str(tmp_path), "--gpus", *gpus, "--ports", *ports, "--clients-per-server", clients]
    )
    with pytest.raises(ValueError, match="unique|Invalid"):
        ablations.make_plan(args)


def test_dry_run_never_launches_processes(monkeypatch, tmp_path, capsys):
    def forbidden(*_, **__):
        pytest.fail("Dry run launched a process")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    output, plan_path = tmp_path / "outputs", tmp_path / "plan.json"
    ablations.main(["--output-dir", str(output), "--plan-path", str(plan_path)])
    assert len(json.loads(plan_path.read_text())["runs"]) == 13
    assert not output.exists()
    capsys.readouterr()


@pytest.fixture
def completed_run(tmp_path):
    run = ablations.Run("goal_baseline", "libero_goal")
    args = ablations.parse_args(["--output-dir", str(tmp_path)])
    rows = [
        {
            "suite": "libero_goal",
            "task_id": t,
            "episode_idx": e,
            "seed": 7,
            "policy_seed": 0,
            "request_seed": 0,
            "rng_protocol": ablations.RNG_PROTOCOL,
            "num_inference_steps": 10,
            "instruction_mode": "original",
            "camera_mode": "both",
            "replan_steps": 5,
            "success": True,
            "warmup_seconds": [0.5, 0.3] if t == 0 and e == 0 else [],
            "inference_seconds": [0.12],
            "server_inference_seconds": [0.1],
            "evaluation_seconds": 1.0,
            "policy_steps": 5,
            "predicted_chunk_lengths": [10],
        }
        for t in range(10)
        for e in range(10)
    ]
    return run, args, rows, tmp_path / "episodes.jsonl"


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_summary_counts_unique_episodes_and_excludes_warmup(completed_run):
    run, args, rows, path = completed_run
    _write(path, rows)
    result = ablations.summarize_run(path, run, args)
    assert result["successes"] == 100
    assert result["client_latency"]["calls"] == 100
    assert result["client_latency"]["mean_ms"] == pytest.approx(120)
    assert result["server_latency"]["mean_ms"] == pytest.approx(100)
    assert result["warmup_seconds"] == [0.5, 0.3]
    ablations.write_summaries([result], path.parent)
    assert (path.parent / "summary.csv").read_text().splitlines()[1].startswith("goal_baseline,")
    assert "100.00 / 100.00" in (path.parent / "SUMMARY.md").read_text()


@pytest.mark.parametrize("failure", ["duplicate", "wrong_flow", "missing_server_timing", "wrong_rng"])
def test_summary_rejects_incomplete_or_misconfigured_results(completed_run, failure):
    run, args, rows, path = completed_run
    if failure == "duplicate":
        rows[-1] = rows[0]
    elif failure == "wrong_flow":
        rows[-1]["num_inference_steps"] = 5
    elif failure == "missing_server_timing":
        rows[-1]["server_inference_seconds"] = [None]
    else:
        rows[-1]["request_seed"] = 1
    _write(path, rows)
    message = {
        "duplicate": "100 unique episodes",
        "wrong_flow": "configuration differs",
        "missing_server_timing": "Server timing must be present",
        "wrong_rng": "RNG protocol",
    }[failure]
    with pytest.raises(ValueError, match=message):
        ablations.summarize_run(path, run, args)
