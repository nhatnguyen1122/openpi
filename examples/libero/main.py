from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import logging
import math
import pathlib
import sys
import time
from typing import Literal

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
# Policy-step limits per suite (longest training demos: 193, 254, 270, 505, 373 steps).
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int | Literal["full"] = 5
    # Warmup calls use the first real observation but their actions are discarded.
    warmup_requests: int = 0
    expected_num_inference_steps: int | None = None
    expected_policy_seed: int | None = None
    # Both options are required together, and require an updated server.
    request_num_inference_steps: int | None = None
    request_seed: int | None = None
    instruction_mode: Literal["original", "empty", "paraphrase", "wrong_task"] = "original"
    instruction_map_path: str | None = None
    # Black out one view while preserving its shape and the model's image mask.
    camera_mode: Literal["both", "no_wrist", "no_third_person"] = "both"
    # Noise-and-horizon steering actor (examples/libero/steering): path to an actor.pt next to projection.npy.
    # The actor chooses the flow noise and the number of executed actions, so replan_steps is ignored.
    steering_checkpoint: str | None = None
    # Use the actor mean and the most likely k; otherwise sample both.
    steering_deterministic: bool = True

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    start_episode_idx: int = 0  # First saved initial-state index; use 10 with 40 trials to extend 10 to 50
    task_ids: list[int] | None = None  # Evaluate selected tasks, or all tasks when omitted
    render_gpu_device_id: int = -1

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    save_video: bool = True
    results_path: str | None = None  # Append one JSON record per completed episode

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    task_ids = list(range(num_tasks_in_suite)) if args.task_ids is None else args.task_ids
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids must be nonempty and contain no duplicates")
    if any(task_id < 0 or task_id >= num_tasks_in_suite for task_id in task_ids):
        raise ValueError(f"task_ids must be in [0, {num_tasks_in_suite})")
    if args.num_trials_per_task < 1 or (args.replan_steps != "full" and args.replan_steps < 1):
        raise ValueError("num_trials_per_task and replan_steps must be positive")
    if args.start_episode_idx < 0:
        raise ValueError("start_episode_idx must be nonnegative")
    if args.warmup_requests < 0:
        raise ValueError("warmup_requests must be nonnegative")
    policy_prompts = _get_policy_prompts(task_suite, args)
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    if args.results_path is not None:
        pathlib.Path(args.results_path).parent.mkdir(parents=True, exist_ok=True)

    if args.task_suite_name not in MAX_STEPS:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    max_steps = MAX_STEPS[args.task_suite_name]

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    server_metadata = client.get_server_metadata()
    request_sampling = args.request_num_inference_steps is not None or args.request_seed is not None
    if request_sampling:
        if args.request_num_inference_steps is None or args.request_seed is None:
            raise ValueError("request_num_inference_steps and request_seed must be set together")
        if not 1 <= args.request_num_inference_steps <= 100 or args.request_seed < 0:
            raise ValueError("Invalid request sampling steps or seed")
        if set(server_metadata.get("request_sampling_options", [])) != {"num_steps", "seed"}:
            raise ValueError("Server does not support isolated request sampling; restart with updated code")
    effective_steps = (
        args.request_num_inference_steps if request_sampling else server_metadata.get("num_inference_steps")
    )
    steering = None
    if args.steering_checkpoint is not None:
        if request_sampling:
            raise ValueError("Steering chooses its own noise; do not combine it with request sampling")
        steering = _load_steering(args.steering_checkpoint, client, deterministic=args.steering_deterministic)
        effective_steps = steering.policy.spec.num_flow_steps or steering.client.metadata["default_num_steps"]
    if args.expected_num_inference_steps is not None and effective_steps != args.expected_num_inference_steps:
        raise ValueError(
            f"Expected {args.expected_num_inference_steps} flow steps, but server metadata is {server_metadata}. "
            "Start a new server with --num-inference-steps."
        )
    if args.expected_policy_seed is not None and server_metadata.get("policy_seed") != args.expected_policy_seed:
        raise ValueError(f"Expected policy seed {args.expected_policy_seed}, but server metadata is {server_metadata}")
    needs_warmup = args.warmup_requests > 0

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)
        episode_end = args.start_episode_idx + args.num_trials_per_task
        if episode_end > len(initial_states):
            raise ValueError(f"Task {task_id} only has {len(initial_states)} initial states")

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed, args.render_gpu_device_id)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.start_episode_idx, episode_end)):
            episode_start = time.monotonic()
            inference_seconds = []
            server_inference_seconds = []
            warmup_seconds = []
            predicted_chunk_lengths = []
            queued_chunk_lengths = []
            done = False
            if steering is not None:
                steering.start_episode(f"{args.seed}:{args.task_suite_name}:{task_id}:{episode_idx}")
            logging.info(f"\nTask: {task_description}")
            logging.info("Policy instruction (%s): %r", args.instruction_mode, policy_prompts[task_id])

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {episode_idx + 1} (initial-state index {episode_idx})...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed images (rotated 180 degrees to match train preprocessing)
                    img, wrist_img = preprocess_images(obs, args.resize_size)

                    # Save preprocessed image for replay video
                    if args.save_video:
                        replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = make_policy_element(
                            obs, policy_prompts[task_id], args.resize_size, args.camera_mode, images=(img, wrist_img)
                        )
                        if request_sampling:
                            # Pair action noise across conditions, independently of
                            # server assignment, client scheduling, and warmup calls.
                            identity = f"{args.request_seed}:{args.seed}:{args.task_suite_name}:{task_id}:{episode_idx}:{len(inference_seconds)}"
                            sample_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big")
                            element["_openpi_inference"] = {
                                "num_steps": args.request_num_inference_steps,
                                "seed": sample_seed,
                            }

                        if needs_warmup:
                            for _ in range(args.warmup_requests):
                                warmup_start = time.monotonic()
                                if steering is None:
                                    client.infer(element)
                                else:
                                    steering.warmup(element)
                                warmup_seconds.append(time.monotonic() - warmup_start)
                            needs_warmup = False

                        # Query model to get action
                        inference_start = time.monotonic()
                        if steering is None:
                            response = client.infer(element)
                            if request_sampling and response.get("sampling_options") != element["_openpi_inference"]:
                                raise ValueError("Server did not acknowledge the requested sampler settings")
                            action_chunk = np.asarray(response["actions"])
                            server_ms = response.get("server_timing", {}).get("infer_ms")
                        else:
                            action_chunk, steered_k = steering.query(
                                element, robot_state(obs), t - args.num_steps_wait, max_steps
                            )
                            server_ms = None
                        inference_seconds.append(time.monotonic() - inference_start)
                        server_inference_seconds.append(None if server_ms is None else float(server_ms) / 1000)
                        if action_chunk.ndim != 2 or len(action_chunk) == 0:
                            raise ValueError(f"Invalid action chunk shape: {action_chunk.shape}")
                        if not np.isfinite(action_chunk).all():
                            raise ValueError("Policy returned nonfinite actions")
                        if steering is not None:
                            replan_steps = steered_k
                        else:
                            replan_steps = len(action_chunk) if args.replan_steps == "full" else args.replan_steps
                        assert len(action_chunk) >= replan_steps, (
                            f"We want to replan every {replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        )
                        predicted_chunk_lengths.append(len(action_chunk))
                        queued_chunk_lengths.append(replan_steps)
                        action_plan.extend(action_chunk[:replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    t += 1
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                except Exception:
                    logging.exception("Evaluation failed; aborting instead of counting a benchmark failure")
                    env.close()
                    raise

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            if args.save_video:
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path)
                    / f"rollout_{args.task_suite_name}_{task_segment}_{episode_idx:03d}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            episode_wall_seconds = time.monotonic() - episode_start
            episode_result = {
                "suite": args.task_suite_name,
                "task_id": task_id,
                "task": task_description,
                "episode_idx": episode_idx,
                "seed": args.seed,
                "success": bool(done),
                "policy_steps": t - args.num_steps_wait,
                "max_steps": max_steps,
                "replan_steps": "steering" if steering is not None else args.replan_steps,
                "instruction_mode": args.instruction_mode,
                "policy_prompt": policy_prompts[task_id],
                "instruction_source_task_id": (task_id + 1) % num_tasks_in_suite
                if args.instruction_mode == "wrong_task"
                else None,
                "camera_mode": args.camera_mode,
                "camera_intervention": "none" if args.camera_mode == "both" else "blackout_pixels_keep_image_mask",
                "num_inference_steps": effective_steps,
                "policy_seed": server_metadata.get("policy_seed"),
                "request_seed": args.request_seed,
                "rng_protocol": "sha256-v1:policy_seed:env_seed:suite:task:episode:call"
                if request_sampling
                else "server_stream",
                "save_video": args.save_video,
                "wall_seconds": episode_wall_seconds,
                "evaluation_seconds": episode_wall_seconds - sum(warmup_seconds),
                "inference_seconds": inference_seconds,
                "server_inference_seconds": server_inference_seconds,
                "warmup_seconds": warmup_seconds,
                "predicted_chunk_lengths": predicted_chunk_lengths,
                "queued_chunk_lengths": queued_chunk_lengths,
            }
            if steering is not None:
                episode_result.update(steering.episode_record())
            if args.results_path is not None:
                with pathlib.Path(args.results_path).open("a") as results_file:
                    results_file.write(json.dumps(episode_result) + "\n")
            logging.info(
                "Episode took %.2fs (%d policy calls)",
                episode_result["wall_seconds"],
                len(inference_seconds),
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        env.close()

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


class _Steering:
    """Evaluation-time noise-and-horizon steering on top of the policy-server connection."""

    def __init__(self, checkpoint: str, client):
        import torch

        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
        from examples.libero.steering import common
        from examples.libero.steering import sac

        self._torch = torch
        path = pathlib.Path(checkpoint)
        self.checkpoint = str(path)
        self.policy = sac.SteeringPolicy.load(path.parent, path.name)
        self.client = common.SteeringClient("", 0, policy=client)
        spec = self.policy.spec
        if (spec.action_horizon, spec.model_action_dim) != (self.client.action_horizon, self.client.model_action_dim):
            raise ValueError("Steering checkpoint was trained for a different action shape than the server's")
        self._rng = np.random.default_rng(0)
        self._generator = torch.Generator()
        self.chosen_k: list[int] = []
        self.deterministic = True

    def start_episode(self, identity: str) -> None:
        # Pad noise (and sampled actions when not deterministic) depend only on the episode identity.
        seed = int.from_bytes(hashlib.sha256(f"steering:{identity}".encode()).digest()[:8], "big")
        self._rng = np.random.default_rng(seed)
        self._generator.manual_seed(seed % 2**63)
        self.chosen_k = []

    def query(self, element: dict, state: np.ndarray, t: int, max_steps: int) -> tuple[np.ndarray, int]:
        embedding = self.client.embed(element)
        s = self.policy.state(embedding, state, t, max_steps)
        noise, _, k_index = self.policy.decide(
            s, self._rng, deterministic=self.deterministic, generator=self._generator
        )
        actions = self.client.act(noise, self.policy.spec.num_flow_steps)
        k = self.policy.spec.horizons[k_index]
        self.chosen_k.append(k)
        return actions, k

    def warmup(self, element: dict) -> None:
        self.client.embed(element)
        shape = (self.policy.spec.action_horizon, self.policy.spec.model_action_dim)
        self.client.act(np.zeros(shape, np.float32), self.policy.spec.num_flow_steps)

    def episode_record(self) -> dict:
        return {
            "steering_checkpoint": self.checkpoint,
            "steering_actor_version": self.policy.version,
            "steering_deterministic": self.deterministic,
            "steering_pad_noise": self.policy.spec.pad_noise,
            "steering_chosen_k": list(self.chosen_k),
        }


def _load_steering(checkpoint: str, client, *, deterministic: bool = True) -> _Steering:
    steering = _Steering(checkpoint, client)
    steering.deterministic = deterministic
    return steering


def _get_policy_prompts(task_suite, args: Args) -> dict[int, str]:
    originals = {i: str(task_suite.get_task(i).language) for i in range(task_suite.n_tasks)}
    if args.instruction_map_path is not None and args.instruction_mode != "paraphrase":
        raise ValueError("instruction_map_path is only used with instruction_mode=paraphrase")
    if args.instruction_mode == "original":
        return originals
    if args.instruction_mode == "empty":
        return dict.fromkeys(originals, "")
    if args.instruction_mode == "wrong_task":
        prompts = {i: originals[(i + 1) % task_suite.n_tasks] for i in originals}
        if any(prompts[i] == originals[i] for i in originals):
            raise ValueError("wrong_task requires different instructions for consecutive tasks")
        return prompts
    if args.instruction_map_path is None:
        raise ValueError("paraphrase requires instruction_map_path")
    mapping = json.loads(pathlib.Path(args.instruction_map_path).read_text())
    if mapping.get("suite") != args.task_suite_name:
        raise ValueError("Paraphrase mapping suite does not match task_suite_name")
    instructions = mapping["instructions"]
    prompts = {}
    for task_id, original in originals.items():
        paraphrase = instructions.get(original)
        if not isinstance(paraphrase, str) or not paraphrase.strip() or paraphrase == original:
            raise ValueError(f"Missing, empty, or unchanged paraphrase for task {task_id}: {original}")
        prompts[task_id] = paraphrase
    return prompts


def preprocess_images(obs: dict, resize_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns the (third-person, wrist) images exactly as the policy saw them during training."""
    # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))
    return img, wrist_img


def robot_state(obs: dict) -> np.ndarray:
    """8-dim state: end-effector position (3), axis-angle (3), gripper joint positions (2)."""
    return np.concatenate((obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))


def make_policy_element(
    obs: dict,
    prompt: str,
    resize_size: int = 224,
    camera_mode: str = "both",
    *,
    images: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Builds the policy-server request from a raw LIBERO observation."""
    img, wrist_img = preprocess_images(obs, resize_size) if images is None else images
    return {
        "observation/image": np.zeros_like(img) if camera_mode == "no_third_person" else img,
        "observation/wrist_image": np.zeros_like(wrist_img) if camera_mode == "no_wrist" else wrist_img,
        "observation/state": robot_state(obs),
        "prompt": prompt,
    }


def _get_libero_env(task, resolution, seed, render_gpu_device_id=-1):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "render_gpu_device_id": render_gpu_device_id,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
