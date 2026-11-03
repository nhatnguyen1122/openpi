"""Run the LIBERO ablation matrix: one policy server per GPU, several concurrent evaluation clients.

Each condition evaluates one suite, all 10 tasks and initial states 0-9 (100 episodes). Flow steps and noise
seeds are sent with every request, so clients sharing a server cannot affect one another. Without --execute
the script only prints the plan.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
import pathlib
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
PARAPHRASES = ROOT / "examples/libero/instructions/libero_goal_paraphrases.json"
SOURCES = [
    "examples/libero/ablations.py",
    "examples/libero/main.py",
    "scripts/serve_policy.py",
    "src/openpi/policies/policy_config.py",
    "src/openpi/policies/policy.py",
]
RNG_PROTOCOL = "sha256-v1:policy_seed:env_seed:suite:task:episode:call"
EPISODES_PER_RUN = 100  # 10 tasks x initial states 0-9
WARMUP_CALLS = 2


@dataclasses.dataclass(frozen=True)
class Run:
    name: str
    suite: str
    flow_steps: int = 10
    instruction: str = "original"
    camera: str = "both"
    replan: int | str = 5


def build_runs(flow_suite: str = "libero_goal") -> list[Run]:
    runs = [
        Run("goal_baseline", "libero_goal"),
        Run("goal_instruction_empty", "libero_goal", instruction="empty"),
        Run("goal_instruction_paraphrase", "libero_goal", instruction="paraphrase"),
        Run("goal_instruction_wrong_task", "libero_goal", instruction="wrong_task"),
    ]
    prefix = "goal" if flow_suite == "libero_goal" else "spatial"
    if flow_suite == "libero_spatial":
        runs.append(Run("spatial_baseline", flow_suite))
    runs.extend(Run(f"{prefix}_flow_{steps}", flow_suite, flow_steps=steps) for steps in [5, 2, 1])
    runs.extend(
        [
            Run("object_baseline", "libero_object"),
            Run("object_no_wrist", "libero_object", camera="no_wrist"),
            Run("object_no_third_person", "libero_object", camera="no_third_person"),
            Run("libero10_replan_1", "libero_10", replan=1),
            Run("libero10_baseline_replan_5", "libero_10"),
            Run("libero10_replan_full", "libero_10", replan="full"),
        ]
    )
    return runs


def write_json(path: pathlib.Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _base_environment(gpu: int, args: argparse.Namespace) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "OPENPI_DATA_HOME": str(ROOT / "data/openpi-cache"),
        "JAX_COMPILATION_CACHE_DIR": str(ROOT / "data/jax-cache"),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": str(args.memory_fraction),
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "1",
    }


def server_launch(gpu: int, port: int, args: argparse.Namespace) -> dict:
    command = [
        sys.executable,
        "scripts/serve_policy.py",
        "--port",
        str(port),
        "--policy-seed",
        str(args.policy_seed),
        "policy:checkpoint",
        "--policy.config",
        "pi05_libero",
        "--policy.dir",
        str(args.checkpoint),
    ]
    return {
        "gpu": gpu,
        "port": port,
        "command": command,
        "environment": {**_base_environment(gpu, args), "JAX_PLATFORMS": "cuda"},
    }


def client_launch(run: Run, args: argparse.Namespace, gpu: int, port: int) -> dict:
    directory = args.output_dir / run.name
    environment = {
        **_base_environment(gpu, args),
        "JAX_PLATFORMS": "cpu",
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": str(gpu),
        "PYOPENGL_PLATFORM": "egl",
        "LIBERO_CONFIG_PATH": str(ROOT / "data/libero/config"),
        "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "third_party/libero")]),
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
    }
    # fmt: off
    command = [
        sys.executable, "-u", "examples/libero/main.py",
        "--args.host", "127.0.0.1",
        "--args.port", str(port),
        "--args.task-suite-name", run.suite,
        "--args.num-trials-per-task", "10",
        "--args.start-episode-idx", "0",
        "--args.seed", str(args.seed),
        "--args.replan-steps", str(run.replan),
        "--args.instruction-mode", run.instruction,
        "--args.camera-mode", run.camera,
        "--args.warmup-requests", str(WARMUP_CALLS),
        "--args.expected-num-inference-steps", str(run.flow_steps),
        "--args.expected-policy-seed", str(args.policy_seed),
        "--args.request-num-inference-steps", str(run.flow_steps),
        "--args.request-seed", str(args.policy_seed),
        "--args.render-gpu-device-id", str(gpu),
        "--args.results-path", str(directory / "episodes.jsonl"),
        "--args.video-out-path", str(directory / "videos"),
    ]
    # fmt: on
    if run.instruction == "paraphrase":
        command.extend(["--args.instruction-map-path", str(PARAPHRASES)])
    if args.no_videos:
        command.append("--args.no-save-video")
    return {
        "run": dataclasses.asdict(run),
        "directory": str(directory),
        "client_environment": environment,
        "client_command": command,
    }


def _latency(values: list[float]) -> dict:
    values = sorted(values)
    if not values:
        raise ValueError("Missing latency measurements")
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Latency measurements must be finite and nonnegative")
    return {
        "calls": len(values),
        "mean_ms": statistics.mean(values) * 1000,
        "median_ms": statistics.median(values) * 1000,
        "p95_ms": values[math.ceil(0.95 * len(values)) - 1] * 1000,
    }


def summarize_run(path: pathlib.Path, run: Run, args: argparse.Namespace) -> dict:
    """Validates a finished condition (identities, controls, warmups, timings) and summarizes it."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    expected = {(task, episode) for task in range(10) for episode in range(10)}
    if len(rows) != EPISODES_PER_RUN or {(r["task_id"], r["episode_idx"]) for r in rows} != expected:
        raise ValueError("Run must contain exactly 100 unique episodes: 10 tasks x states 0-9")
    wanted = (run.suite, args.seed, run.flow_steps, run.instruction, run.camera, run.replan, args.policy_seed)
    for row in rows:
        actual = (
            row["suite"],
            row["seed"],
            row["num_inference_steps"],
            row["instruction_mode"],
            row["camera_mode"],
            row["replan_steps"],
            row["policy_seed"],
        )
        if actual != wanted:
            raise ValueError(f"Episode configuration differs from the plan: {actual} != {wanted}")
        if row.get("rng_protocol") != RNG_PROTOCOL or row.get("request_seed") != args.policy_seed:
            raise ValueError("Incorrect request-scoped RNG protocol")
    if sum(len(r["warmup_seconds"]) for r in rows) != WARMUP_CALLS:
        raise ValueError(f"Expected exactly {WARMUP_CALLS} unscored warmup calls per run")
    client_times = [t for row in rows for t in row["inference_seconds"]]
    server_times = [t for row in rows for t in row["server_inference_seconds"]]
    if any(t is None for t in server_times) or len(client_times) != len(server_times):
        raise ValueError("Server timing must be present for every scored policy call")
    successes = sum(bool(r["success"]) for r in rows)
    return {
        "run": dataclasses.asdict(run),
        "environment_seed": args.seed,
        "policy_seed": args.policy_seed,
        "checkpoint": str(args.checkpoint),
        "episodes": len(rows),
        "successes": successes,
        "success_rate_percent": 100 * successes / len(rows),
        "client_latency": _latency(client_times),
        "server_latency": _latency(server_times),
        "warmup_seconds": [t for row in rows for t in row["warmup_seconds"]],
        "mean_episode_evaluation_seconds": statistics.mean(r["evaluation_seconds"] for r in rows),
        "mean_policy_steps": statistics.mean(r["policy_steps"] for r in rows),
        "mean_policy_calls_per_episode": len(client_times) / len(rows),
        "predicted_chunk_lengths": sorted({n for r in rows for n in r["predicted_chunk_lengths"]}),
        "per_task": [
            {
                "task_id": task,
                "episodes": 10,
                "successes": sum(bool(r["success"]) for r in rows if r["task_id"] == task),
            }
            for task in range(10)
        ],
    }


def write_summaries(summaries: list[dict], output_dir: pathlib.Path) -> None:
    write_json(output_dir / "summaries.json", summaries)
    table = [
        {
            **summary["run"],
            "episodes": summary["episodes"],
            "success_rate_percent": summary["success_rate_percent"],
            "client_mean_ms": summary["client_latency"]["mean_ms"],
            "client_p95_ms": summary["client_latency"]["p95_ms"],
            "server_mean_ms": summary["server_latency"]["mean_ms"],
            "server_p95_ms": summary["server_latency"]["p95_ms"],
            "mean_policy_calls_per_episode": summary["mean_policy_calls_per_episode"],
        }
        for summary in summaries
    ]
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    lines = [
        "# LIBERO ablation results",
        "",
        "Each run: 10 tasks x 10 episodes; warmup calls excluded from latency. One seed pair, not a multi-seed estimate.",
        "",
        "| Run | Success | Client mean / p95 (ms) | Server mean / p95 (ms) |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['name']} | {row['success_rate_percent']:.1f}% | "
        f"{row['client_mean_ms']:.2f} / {row['client_p95_ms']:.2f} | "
        f"{row['server_mean_ms']:.2f} / {row['server_p95_ms']:.2f} |"
        for row in table
    )
    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def stop_process(process: subprocess.Popen | None) -> None:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def make_plan(args: argparse.Namespace) -> dict:
    gpus, ports = args.gpus, args.ports
    if not gpus or len(gpus) != len(ports) or len(set(gpus)) != len(gpus) or len(set(ports)) != len(ports):
        raise ValueError("Provide one unique port per unique GPU")
    if any(g < 0 for g in gpus) or any(not 0 < p < 65536 for p in ports) or args.clients_per_server < 1:
        raise ValueError("Invalid GPU, port, or client limit")
    runs = build_runs(args.flow_suite)
    if args.runs:
        unknown = set(args.runs) - {r.name for r in runs}
        if unknown:
            raise ValueError(f"Unknown runs: {sorted(unknown)}")
        runs = [r for r in runs if r.name in args.runs]
    # Start the expensive replan=1 condition first to avoid a long final tail.
    runs.sort(key=lambda r: r.name != "libero10_replan_1")
    return {
        "servers": [server_launch(gpu, port, args) for gpu, port in zip(gpus, ports, strict=True)],
        "clients_per_server": args.clients_per_server,
        "runs": [dataclasses.asdict(r) for r in runs],
        "checkpoint": str(args.checkpoint),
        "protocol": {
            "episodes_per_run": EPISODES_PER_RUN,
            "tasks": list(range(10)),
            "initial_states": list(range(10)),
            "total_episodes": EPISODES_PER_RUN * len(runs),
            "environment_seed": args.seed,
            "policy_seed": args.policy_seed,
            "rng_protocol": RNG_PROTOCOL,
            "warmup_calls_per_run": WARMUP_CALLS,
            "camera_intervention": "blackout_pixels_keep_image_mask",
            "wrong_instruction_mapping": "task i -> task (i+1) modulo 10",
            "paraphrase_sha256": hashlib.sha256(PARAPHRASES.read_bytes()).hexdigest(),
            "videos": not args.no_videos,
        },
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCES},
        "package_versions": {
            name: importlib.metadata.version(name) for name in ["jax", "torch", "numpy", "mujoco", "robosuite"]
        },
    }


def _popen(command: list[str], environment: dict[str, str], log_path: pathlib.Path) -> subprocess.Popen:
    with log_path.open("w") as log:
        return subprocess.Popen(
            command, cwd=ROOT, env={**os.environ, **environment}, stdout=log, stderr=subprocess.STDOUT
        )


def _wait_until_ready(servers: list[dict], timeout: float = 900) -> None:
    deadline = time.monotonic() + timeout
    ready: set[int] = set()
    while len(ready) < len(servers):
        for server in servers:
            if server["process"].poll() is not None:
                raise RuntimeError(f"GPU {server['gpu']} server exited; inspect its log")
            if server["port"] in ready:
                continue
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{server['port']}/healthz", timeout=2) as response:
                    if response.status == 200:
                        ready.add(server["port"])
            except (urllib.error.URLError, TimeoutError):
                pass
        if time.monotonic() > deadline:
            raise TimeoutError("Servers did not become ready within 15 minutes")
        if len(ready) < len(servers):
            time.sleep(5)


def execute(plan: dict, args: argparse.Namespace) -> None:
    root = args.output_dir
    if root.exists():
        raise FileExistsError(f"Choose a fresh output directory: {root}")
    if not (args.checkpoint / "params").is_dir():
        raise FileNotFoundError(args.checkpoint / "params")
    for server in plan["servers"]:
        with socket.socket() as probe:
            probe.bind(("0.0.0.0", server["port"]))
    root.mkdir(parents=True)
    write_json(root / "plan.json", plan)
    # Preserve the exact source used by these long-lived processes.
    for relative in plan["source_sha256"]:
        target = root / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    (root / "source_changes.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT))

    pending = [Run(**r) for r in plan["runs"]]
    servers, active, summaries, failures = [], [], [], []
    status = {"complete": False, "started_utc": datetime.datetime.now(datetime.UTC).isoformat()}

    def write_status():
        jobs = []
        for job in active:
            path = job["directory"] / "episodes.jsonl"
            count = len(path.read_text().splitlines()) if path.exists() else 0
            jobs.append(
                {"run": job["run"].name, "pid": job["process"].pid, "port": job["server"]["port"], "episodes": count}
            )
        status.update(
            updated_utc=datetime.datetime.now(datetime.UTC).isoformat(),
            servers=[{k: s[k] for k in ["gpu", "port", "pid"]} for s in servers],
            running=jobs,
            pending=[r.name for r in pending],
            finished=[s["run"]["name"] for s in summaries],
            failures=failures,
            completed_episodes=EPISODES_PER_RUN * len(summaries) + sum(j["episodes"] for j in jobs),
        )
        write_json(root / "status.json", status)

    def finish(job: dict, code: int) -> None:
        directory, run = job["directory"], job["run"]
        job_status = {"exit_code": code, "complete": False}
        try:
            if code != 0:
                raise RuntimeError(f"Client exited with {code}; see {directory / 'client.log'}")
            summary = summarize_run(directory / "episodes.jsonl", run, args)
            summary.update(gpu=job["server"]["gpu"], port=job["server"]["port"], protocol=plan["protocol"])
            write_json(directory / "summary.json", summary)
            summaries.append(summary)
            write_summaries(summaries, root)
            job_status["complete"] = True
            print(f"DONE {run.name}: {summary['successes']}/{EPISODES_PER_RUN}", flush=True)
        except Exception as error:
            job_status["error"] = str(error)
            failures.append({"run": run.name, "error": str(error)})
            print(f"FAILED {run.name}: {error}", flush=True)
        write_json(directory / "status.json", job_status)

    def start(run: Run, server: dict) -> None:
        launch = client_launch(run, args, server["gpu"], server["port"])
        directory = pathlib.Path(launch["directory"])
        directory.mkdir()
        write_json(
            directory / "manifest.json",
            {
                **launch,
                "protocol": plan["protocol"],
                "source_sha256": plan["source_sha256"],
                "server_pid": server["pid"],
                "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            },
        )
        process = _popen(launch["client_command"], launch["client_environment"], directory / "client.log")
        active.append({"run": run, "directory": directory, "server": server, "process": process})
        write_json(directory / "status.json", {"complete": False, "client_pid": process.pid})
        print(f"START {run.name} gpu={server['gpu']} port={server['port']} pid={process.pid}", flush=True)

    try:
        for spec in plan["servers"]:
            process = _popen(
                spec["command"], spec["environment"], root / f"server_gpu{spec['gpu']}_port{spec['port']}.log"
            )
            servers.append({**spec, "process": process, "pid": process.pid})
            print(f"SERVER gpu={spec['gpu']} port={spec['port']} pid={process.pid}", flush=True)
        _wait_until_ready(servers)
        print("Servers ready; launching evaluation clients", flush=True)
        last_report = 0.0
        while pending or active:
            for server in servers:
                if server["process"].poll() is not None:
                    raise RuntimeError(f"GPU {server['gpu']} server exited during evaluation")
            for job in list(active):
                code = job["process"].poll()
                if code is not None:
                    active.remove(job)
                    finish(job, code)
            # Fill one slot on each server before assigning additional slots.
            for _ in range(plan["clients_per_server"]):
                for server in servers:
                    if pending and sum(j["server"] is server for j in active) < plan["clients_per_server"]:
                        start(pending.pop(0), server)
            write_status()
            if time.monotonic() - last_report >= 60:
                print(
                    f"PROGRESS {status['completed_episodes']}/{plan['protocol']['total_episodes']} episodes; "
                    f"{len(summaries)} runs complete, {len(active)} active, {len(pending)} queued",
                    flush=True,
                )
                last_report = time.monotonic()
            if pending or active:
                time.sleep(5)
        if failures:
            raise RuntimeError(f"{len(failures)} run(s) failed; inspect status.json")
        status["complete"] = True
    except BaseException as error:
        status["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        for job in active:
            stop_process(job["process"])
            write_json(
                job["directory"] / "status.json",
                {
                    "complete": False,
                    "exit_code": job["process"].poll(),
                    "error": "Runner interrupted; partial results preserved",
                },
            )
        for server in servers:
            stop_process(server["process"])
        write_status()
        status["finished_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
        status["server_exit_codes"] = {str(s["gpu"]): s["process"].poll() for s in servers}
        write_json(root / "status.json", status)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True, help="Fresh directory for all results")
    parser.add_argument("--checkpoint", type=pathlib.Path, default=ROOT / "checkpoints/pi05_libero")
    parser.add_argument("--gpus", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--ports", nargs="+", type=int, default=[8001, 8002])
    parser.add_argument("--clients-per-server", type=int, default=2)
    parser.add_argument("--memory-fraction", type=float, default=0.42)
    parser.add_argument("--seed", type=int, default=7, help="LIBERO environment seed")
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--flow-suite", choices=["libero_goal", "libero_spatial"], default="libero_goal")
    parser.add_argument("--runs", nargs="+", help="Run only these condition names")
    parser.add_argument("--plan-path", type=pathlib.Path, help="Also save the plan as JSON")
    parser.add_argument("--no-videos", action="store_true", help="Disable videos for all selected runs")
    parser.add_argument("--execute", action="store_true", help="Launch jobs; otherwise only print the plan")
    args = parser.parse_args(argv)
    if not 0 < args.memory_fraction <= 1:
        parser.error("Invalid GPU memory fraction")
    args.output_dir = args.output_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plan = make_plan(args)
    if args.plan_path:
        args.plan_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.plan_path, plan)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    print(
        f"EXECUTE {len(plan['runs'])} runs / {plan['protocol']['total_episodes']} episodes on GPUs {args.gpus}",
        flush=True,
    )

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt("Runner received termination signal")

    signal.signal(signal.SIGTERM, interrupt)
    execute(plan, args)


if __name__ == "__main__":
    main()
