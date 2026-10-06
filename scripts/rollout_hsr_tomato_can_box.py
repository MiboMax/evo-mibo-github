#!/usr/bin/env python3

import argparse
import asyncio
import csv
import json
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
import websockets

sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.initial_scene_slots import InitialSceneSlotConfig, InitialSceneSlotPerceptor, slots_from_prediction

import robo_manip_baselines.envs  # noqa: F401  # Register HSR envs


DEFAULT_ENV_ID = "robo_manip_baselines/MujocoHsrTomatoCanBoxEnv-v0"
DEFAULT_PROMPT = "Pick up the can and move it to the box."
DEFAULT_CUP_PROMPT = "Pick up the cup and pour its contents into the box."
EXPECTED_CAN_X = 0.84
EXPECTED_BOX_X = 1.04
EXPECTED_GOAL_X = 1.04
RANDOM_CAN_X_MIN = 0.795000
RANDOM_CAN_X_MAX = 0.899639
RANDOM_CAN_Y_MIN = -0.048856
RANDOM_CAN_Y_MAX = 0.048856
EXPECTED_CUP_X = 0.95
EXPECTED_CUP_Y = 0.16
EXPECTED_CUP_BOX_X = 0.95
EXPECTED_CUP_BOX_Y = 0.0
CUP_XY_JITTER = 0.02
VIS_PANEL_SIZE = (480, 360)
VIS_PANEL_COUNT = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evo-1 flash rollout evaluation for HSR tomato-can box and cup-pouring scenes."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="tomato_can_box",
        choices=["tomato_can_box", "tomato_can_box_random", "cup_box"],
        help="Task-specific layout validation and trajectory logging.",
    )
    parser.add_argument("--uri", type=str, default="ws://127.0.0.1:9017")
    parser.add_argument("--env_id", type=str, default=DEFAULT_ENV_ID)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument(
        "--exec_horizon",
        type=int,
        default=50,
        help="How many actions to execute from each predicted chunk. Matches the HSR horizon used for this run.",
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--inference_seed",
        type=int,
        default=None,
        help=(
            "Optional base seed for deterministic flow-matching inference. Each episode/request "
            "receives a stable derived seed; omit to preserve stochastic server behavior."
        ),
    )
    parser.add_argument(
        "--seed_list",
        type=str,
        default="11,22,33,44,55",
        help="Comma-separated fixed seeds. Overrides --episodes/--seed schedule.",
    )
    parser.add_argument(
        "--world_idx_list",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated world indices for env.modify_world.",
    )
    parser.add_argument("--success_reward", type=float, default=1.0)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument(
        "--history_len",
        type=int,
        default=1,
        help="Number of recent frames to aggregate before each action prediction.",
    )
    parser.add_argument(
        "--history_weight_mode",
        type=str,
        default="uniform",
        choices=["uniform", "linear", "exp"],
        help="Temporal weighting for history aggregation.",
    )
    parser.add_argument(
        "--third_view",
        type=str,
        default="zeros",
        choices=["hand_depth", "zeros"],
        help="Third image sent to the server. Stage1 can still send hand_depth with image_mask disabled.",
    )
    parser.add_argument(
        "--image_mask_mode",
        type=str,
        default="stage1",
        choices=["stage1", "none", "rgb", "rgbd"],
        help=(
            "Image mask sent to Evo-1. Use stage1/none for this stage1 checkpoint because it was trained "
            "with vision_masked=1; use rgb for stage2 tomato rollouts because training uses head+hand RGB only."
        ),
    )
    parser.add_argument(
        "--state_layout",
        type=str,
        default="no_wrench15",
        choices=["no_wrench15", "full21", "no_wrench15_slots4x7", "no_wrench15_slots4x3"],
        help=(
            "State layout sent to Evo-1. no_wrench15_slots4x7 appends flattened "
            "env scene slots (4*7); no_wrench15_slots4x3 appends only [dx,dy,valid] (4*3)."
        ),
    )
    parser.add_argument(
        "--slot_source",
        type=str,
        default="ground_truth",
        choices=["ground_truth", "predicted", "masked"],
        help=(
            "Source for compact 2D slots. predicted runs a reset-time RGBD perceptor once, then "
            "uses base odometry to update dx/dy without accessing environment object positions."
        ),
    )
    parser.add_argument(
        "--initial_scene_slot_ckpt",
        type=Path,
        default=None,
        help="Checkpoint from train_initial_scene_slot_perceptor.py; required for --slot_source=predicted.",
    )
    parser.add_argument(
        "--slot_perceptor_device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used by the small reset-time slot perceptor.",
    )
    parser.add_argument(
        "--slot_valid_threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to turn perceptor valid logits into binary attention masks.",
    )
    parser.add_argument(
        "--mask_scene_slots",
        action="store_true",
        help=(
            "For state_layout=no_wrench15_slots4x7, replace all 28 scene-slot values "
            "with zeros before inference while preserving the 43D state shape."
        ),
    )
    parser.add_argument(
        "--hand_depth_min",
        type=float,
        default=0.10,
        help="Optional fixed minimum depth in meters for converting hand depth to an RGB-like third view.",
    )
    parser.add_argument(
        "--hand_depth_max",
        type=float,
        default=2.00,
        help="Optional fixed maximum depth in meters for converting hand depth to an RGB-like third view.",
    )
    parser.add_argument(
        "--hand_depth_encoding",
        type=str,
        default="inverse",
        choices=["inverse", "linear"],
        help="Depth-to-RGB encoding for the third view. Use inverse with 0.10..2.00m for tomato invdepth checkpoints.",
    )
    parser.add_argument(
        "--hand_depth_colormap",
        action="store_true",
        help="Use a JET colormap for the depth view instead of grayscale. Grayscale matches the dataset conversion default.",
    )
    parser.add_argument(
        "--pre_inference_forward_vel",
        type=float,
        default=0.0,
        help=(
            "Optional local +x mobile-base initial velocity in m/s applied after env reset "
            "and before the first model inference. 0 disables this."
        ),
    )
    parser.add_argument(
        "--initial_settle_steps",
        type=int,
        default=0,
        help=(
            "Step the env with the reset joint command before first inference. "
            "For current tomato slots data, 30 steps aligns rollout reset state with recorded first measured state."
        ),
    )
    parser.add_argument(
        "--close_guard",
        action="store_true",
        help="Delay the first detected gripper-close command and nudge the mobile base forward first.",
    )
    parser.add_argument(
        "--close_guard_forward_vel",
        type=float,
        default=0.10,
        help="Local +x base velocity [m/s] used during close-guard forward nudge.",
    )
    parser.add_argument(
        "--close_guard_steps",
        type=int,
        default=16,
        help="Number of env steps for the close-guard forward nudge.",
    )
    parser.add_argument(
        "--close_guard_min_step",
        type=int,
        default=30,
        help="Do not allow close-guard trigger before this rollout step.",
    )
    parser.add_argument(
        "--close_guard_trigger_direction",
        type=str,
        default="decrease",
        choices=["decrease", "increase", "base_stop"],
        help=(
            "Gripper-command direction that indicates closing. For the HSR tomato model, "
            "decrease catches the pre-grasp close; base_stop triggers when the model stops "
            "forward base motion before closing; increase is kept for backward-compatible experiments."
        ),
    )
    parser.add_argument(
        "--close_guard_base_arm_vx",
        type=float,
        default=0.03,
        help="Base-stop mode: arm after model local +x base velocity has exceeded this value.",
    )
    parser.add_argument(
        "--close_guard_base_stop_vx",
        type=float,
        default=0.01,
        help="Base-stop mode: trigger after armed when model local +x base velocity drops below this value.",
    )
    parser.add_argument(
        "--close_guard_base_stop_consecutive",
        type=int,
        default=3,
        help="Base-stop mode: number of consecutive low-vx model actions required before triggering.",
    )
    parser.add_argument(
        "--close_guard_arm_below",
        type=float,
        default=0.15,
        help="Increase-mode only: arm the close guard after gripper command has dropped below this value.",
    )
    parser.add_argument(
        "--close_guard_arm_above",
        type=float,
        default=0.35,
        help="Decrease-mode only: arm the close guard after gripper command has risen above this open-gripper value.",
    )
    parser.add_argument(
        "--close_guard_trigger_abs",
        type=float,
        default=0.35,
        help="Increase-mode only: trigger when armed and gripper command rises above this absolute value.",
    )
    parser.add_argument(
        "--close_guard_trigger_below",
        type=float,
        default=0.15,
        help="Decrease-mode only: trigger when armed and gripper command falls below this absolute value.",
    )
    parser.add_argument(
        "--close_guard_trigger_delta",
        type=float,
        default=0.25,
        help="Trigger when armed and gripper command changes this much in the configured closing direction.",
    )
    parser.add_argument(
        "--gripper_filter",
        type=str,
        default="none",
        choices=["none", "hold_deadband", "ema", "binary_hysteresis", "binary_task_latch"],
        help=(
            "Execution-side filter for action[8]. hold_deadband keeps the previous gripper command "
            "until the model command differs enough and the minimum hold time has elapsed. "
            "binary_hysteresis maps the gripper to fixed open/close commands with a hysteresis band. "
            "binary_task_latch permits only open->closed->released transitions for one grasp-and-place task."
        ),
    )
    parser.add_argument(
        "--gripper_deadband",
        type=float,
        default=0.08,
        help="hold_deadband mode: minimum absolute command change required before updating action[8].",
    )
    parser.add_argument(
        "--gripper_min_hold_steps",
        type=int,
        default=12,
        help="hold_deadband mode: minimum rollout steps to hold a gripper command after each update.",
    )
    parser.add_argument(
        "--gripper_close_min_hold_steps",
        type=int,
        default=120,
        help=(
            "binary_task_latch mode: minimum closed duration before one final release may be accepted. "
            "After release, later close requests are ignored."
        ),
    )
    parser.add_argument(
        "--gripper_filter_warmup_steps",
        type=int,
        default=0,
        help="Hold the initial observed gripper command for this many rollout steps before filtering model commands.",
    )
    parser.add_argument(
        "--gripper_ema_alpha",
        type=float,
        default=0.20,
        help="ema mode: smoothing factor for action[8].",
    )
    parser.add_argument(
        "--gripper_binary_threshold",
        type=float,
        default=0.25,
        help="binary_hysteresis mode: midpoint threshold between close and open commands.",
    )
    parser.add_argument(
        "--gripper_hysteresis_margin",
        type=float,
        default=0.05,
        help="binary_hysteresis mode: margin around threshold before switching state.",
    )
    parser.add_argument(
        "--gripper_open_cmd",
        type=float,
        default=0.5,
        help="binary_hysteresis mode: executed command for open gripper.",
    )
    parser.add_argument(
        "--gripper_close_cmd",
        type=float,
        default=-0.1,
        help="binary_hysteresis mode: executed command for closed gripper.",
    )
    return parser.parse_args()


def parse_int_csv(text: str) -> list[int]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    if not values:
        raise ValueError("Empty integer list.")
    return values


def resolve_eval_schedule(args):
    if args.seed_list is not None:
        eval_seeds = parse_int_csv(args.seed_list)
    else:
        eval_seeds = [int(args.seed + i) for i in range(args.episodes)]

    if args.world_idx_list is not None:
        eval_world_indices = parse_int_csv(args.world_idx_list)
        if len(eval_world_indices) != len(eval_seeds):
            raise ValueError(
                "Length mismatch: world_idx_list has "
                f"{len(eval_world_indices)} values but seed_list has {len(eval_seeds)} values."
            )
    else:
        eval_world_indices = list(range(len(eval_seeds)))

    args.eval_seeds = eval_seeds
    args.eval_world_indices = eval_world_indices
    args.episodes = len(eval_seeds)


def image_mask_from_mode(mode: str) -> list[int]:
    if mode in {"stage1", "none"}:
        return [0, 0, 0]
    if mode == "rgb":
        return [1, 1, 0]
    if mode == "rgbd":
        return [1, 1, 1]
    raise ValueError(f"Unknown image_mask_mode: {mode}")


def count_gripper_spikes(actions: np.ndarray, threshold: float = 0.05) -> int:
    if actions.size == 0 or actions.shape[0] < 2:
        return 0
    return int(np.count_nonzero(np.abs(np.diff(actions[:, 8])) > threshold))


def count_gripper_threshold_crossings(actions: np.ndarray, threshold: float) -> int:
    if actions.size == 0 or actions.shape[0] < 2:
        return 0
    above = actions[:, 8] > threshold
    return int(np.count_nonzero(above[1:] != above[:-1]))


def preprocess_rgb_for_server(img_rgb: np.ndarray) -> np.ndarray:
    # Evo1_server.py decodes each array with cv2.COLOR_BGR2RGB, so send BGR here.
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (448, 448), interpolation=cv2.INTER_AREA)
    return img_bgr.astype(np.uint8)


def preprocess_depth_for_server(
    depth: np.ndarray,
    min_depth: float | None = None,
    max_depth: float | None = None,
    encoding: str = "inverse",
    colormap: bool = False,
) -> np.ndarray:
    depth_arr = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth_arr) & (depth_arr > 0)
    if not np.any(finite):
        depth_u8 = np.zeros(depth_arr.shape, dtype=np.uint8)
    else:
        valid = depth_arr[finite]
        lo = float(min_depth) if min_depth is not None else float(np.percentile(valid, 1.0))
        hi = float(max_depth) if max_depth is not None else float(np.percentile(valid, 99.0))
        if hi <= lo + 1e-6:
            hi = lo + 1.0
        clipped = np.clip(depth_arr, lo, hi)
        if encoding == "inverse":
            inv = np.zeros_like(clipped, dtype=np.float32)
            inv[finite] = 1.0 / clipped[finite]
            inv_near = 1.0 / lo
            inv_far = 1.0 / hi
            depth_norm = (inv - inv_far) / (inv_near - inv_far + 1e-8)
        elif encoding == "linear":
            depth_norm = (clipped - lo) / (hi - lo)
        else:
            raise ValueError(f"Unsupported hand depth encoding: {encoding}")
        depth_norm = np.where(finite, depth_norm, 0.0)
        depth_u8 = np.clip(depth_norm * 255.0, 0.0, 255.0).astype(np.uint8)

    depth_u8 = cv2.resize(depth_u8, (448, 448), interpolation=cv2.INTER_NEAREST)
    if colormap:
        # cv2.applyColorMap returns BGR, which is what the server-side decoder expects.
        return cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET).astype(np.uint8)
    return np.repeat(depth_u8[:, :, None], 3, axis=2).astype(np.uint8)


def depth_to_bgr_vis(depth: np.ndarray, size=VIS_PANEL_SIZE) -> np.ndarray:
    depth_bgr = preprocess_depth_for_server(depth, encoding="inverse", colormap=True)
    return cv2.resize(depth_bgr, size, interpolation=cv2.INTER_NEAREST)


def _history_weights(length: int, mode: str) -> np.ndarray:
    if length <= 0:
        raise ValueError(f"Invalid history length: {length}")
    if mode == "uniform":
        w = np.ones(length, dtype=np.float32)
    elif mode == "linear":
        w = np.linspace(1.0, float(length), num=length, dtype=np.float32)
    elif mode == "exp":
        w = np.exp(np.linspace(-1.0, 0.0, num=length, dtype=np.float32))
    else:
        raise ValueError(f"Unknown history weight mode: {mode}")
    w /= np.sum(w)
    return w


def aggregate_history(history_items: list[dict], weight_mode: str):
    if len(history_items) == 0:
        raise ValueError("aggregate_history() got empty history.")

    weights = _history_weights(len(history_items), weight_mode)
    state_stack = np.stack([item["state"] for item in history_items], axis=0)
    head_stack = np.stack([item["head_rgb"] for item in history_items], axis=0)
    hand_stack = np.stack([item["hand_rgb"] for item in history_items], axis=0)
    depth_stack = np.stack([item["hand_depth"] for item in history_items], axis=0)

    state_agg = np.sum(state_stack * weights[:, None], axis=0).astype(np.float32)
    head_agg = np.sum(head_stack.astype(np.float32) * weights[:, None, None, None], axis=0)
    hand_agg = np.sum(hand_stack.astype(np.float32) * weights[:, None, None, None], axis=0)
    depth_agg = np.sum(depth_stack.astype(np.float32) * weights[:, None, None], axis=0)

    head_agg = np.clip(head_agg, 0.0, 255.0).astype(np.uint8)
    hand_agg = np.clip(hand_agg, 0.0, 255.0).astype(np.uint8)
    depth_agg = depth_agg.astype(np.float32)
    return state_agg, head_agg, hand_agg, depth_agg


def to_bgr_vis(img_rgb: np.ndarray, size=VIS_PANEL_SIZE) -> np.ndarray:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)


def blank_vis_panel(label: str, size=VIS_PANEL_SIZE) -> np.ndarray:
    width, height = size
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(panel, label, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2, cv2.LINE_AA)
    return panel


def put_panel_label(canvas: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (35, 35, 35), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)


def build_state(obs: dict, state_layout: str) -> np.ndarray:
    parts = [
        np.asarray(obs["joint_pos"], dtype=np.float32),
        np.asarray(obs["joint_vel"], dtype=np.float32),
    ]
    if state_layout == "full21":
        parts.append(np.asarray(obs["wrench"], dtype=np.float32))
    elif state_layout not in {"no_wrench15", "no_wrench15_slots4x7", "no_wrench15_slots4x3"}:
        raise ValueError(f"Unsupported state_layout: {state_layout}")
    parts.append(np.asarray(obs["mobile_vel"], dtype=np.float32))
    return np.concatenate(parts, axis=0)


def build_scene_slots(env, slot_dim: int) -> np.ndarray:
    if not hasattr(env.unwrapped, "get_scene_slots"):
        raise ValueError(
            "state_layout=no_wrench15_slots4x7 requires env.unwrapped.get_scene_slots()."
        )
    slots = np.asarray(env.unwrapped.get_scene_slots(), dtype=np.float32)
    if slots.shape != (4, 7):
        raise ValueError(f"Expected scene slots shape (4,7), got {slots.shape}")
    if slot_dim == 7:
        return slots.reshape(-1)
    if slot_dim == 3:
        return slots[:, [0, 1, 6]].reshape(-1)
    raise ValueError(f"Unsupported scene slot dim: {slot_dim}")


def update_predicted_scene_slots(
    env,
    initial_slots: np.ndarray,
    initial_base_xy: np.ndarray,
) -> np.ndarray:
    if initial_slots.shape != (4, 3):
        raise ValueError(f"Expected predicted initial slots shape (4,3), got {initial_slots.shape}")
    current_base_xy = get_base_pose(env)[:2]
    base_delta_xy = current_base_xy - initial_base_xy
    slots = initial_slots.copy()
    valid = slots[:, 2] > 0.5
    slots[valid, :2] -= base_delta_xy[None, :]
    slots[~valid, :2] = 0.0
    return slots.reshape(-1)


def build_rollout_state(
    env,
    obs: dict,
    state_layout: str,
    mask_scene_slots: bool = False,
    slot_source: str = "ground_truth",
    predicted_initial_slots: np.ndarray | None = None,
    initial_base_xy: np.ndarray | None = None,
) -> np.ndarray:
    state = build_state(obs, state_layout)
    if state_layout == "no_wrench15_slots4x7":
        slots = build_scene_slots(env, slot_dim=7)
        if mask_scene_slots or slot_source == "masked":
            slots = np.zeros_like(slots)
        state = np.concatenate([state, slots], axis=0)
    elif state_layout == "no_wrench15_slots4x3":
        if mask_scene_slots or slot_source == "masked":
            slots = np.zeros(12, dtype=np.float32)
        elif slot_source == "ground_truth":
            slots = build_scene_slots(env, slot_dim=3)
        elif slot_source == "predicted":
            if predicted_initial_slots is None or initial_base_xy is None:
                raise ValueError("Predicted slot source requires initial slots and initial base xy")
            slots = update_predicted_scene_slots(env, predicted_initial_slots, initial_base_xy)
        else:
            raise ValueError(f"Unsupported slot source: {slot_source}")
        state = np.concatenate([state, slots], axis=0)
    return state


def expected_state_dim_for_layout(state_layout: str) -> int:
    if state_layout == "no_wrench15":
        return 15
    if state_layout == "full21":
        return 21
    if state_layout == "no_wrench15_slots4x7":
        return 43
    if state_layout == "no_wrench15_slots4x3":
        return 27
    raise ValueError(f"Unsupported state_layout: {state_layout}")


def load_initial_scene_slot_perceptor(args):
    if args.initial_scene_slot_ckpt is None:
        raise ValueError("--initial_scene_slot_ckpt is required for --slot_source=predicted")
    if not args.initial_scene_slot_ckpt.is_file():
        raise FileNotFoundError(f"Initial-scene slot checkpoint not found: {args.initial_scene_slot_ckpt}")
    device = torch.device(args.slot_perceptor_device)
    payload = torch.load(args.initial_scene_slot_ckpt, map_location=device)
    model_config = InitialSceneSlotConfig(**payload["model_config"])
    model = InitialSceneSlotPerceptor(model_config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, device


def predict_initial_scene_slots(
    model,
    device: torch.device,
    hand_rgb: np.ndarray,
    hand_depth_m: np.ndarray,
    valid_threshold: float,
    depth_min: float,
    depth_max: float,
    head_rgb: np.ndarray | None = None,
) -> np.ndarray:
    rgb = np.asarray(hand_rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected hand RGB HxWx3, got {rgb.shape}")
    head_tensor = None
    if head_rgb is not None:
        head = np.asarray(head_rgb, dtype=np.float32)
        if head.ndim != 3 or head.shape[-1] != 3:
            raise ValueError(f"Expected head RGB HxWx3, got {head.shape}")
        head_tensor = torch.from_numpy(head).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    depth = np.asarray(hand_depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    clipped = np.clip(depth, depth_min, depth_max)
    inv = np.zeros_like(clipped, dtype=np.float32)
    inv[valid] = 1.0 / clipped[valid]
    inv_near = 1.0 / float(depth_min)
    inv_far = 1.0 / float(depth_max)
    inverse_depth = np.clip((inv - inv_far) / (inv_near - inv_far), 0.0, 1.0)
    inverse_depth[~valid] = 0.0
    rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    depth_tensor = torch.from_numpy(inverse_depth).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        prediction = model(rgb_tensor, depth_tensor, head_tensor)
        slots = slots_from_prediction(prediction, valid_threshold=valid_threshold)[0]
        invalid = slots[:, 2] <= 0.5
        slots[invalid, :2] = 0.0
    return slots.float().cpu().numpy()


def get_base_pose(env) -> np.ndarray:
    data = env.unwrapped.data
    return np.array(
        [
            float(data.joint("mobile_x_joint").qpos[0]),
            float(data.joint("mobile_y_joint").qpos[0]),
            float(data.joint("mobile_theta_joint").qpos[0]),
        ],
        dtype=np.float32,
    )


def get_eef_xyz(env) -> np.ndarray:
    return env.unwrapped.data.body("hand_palm_link").xpos.copy().astype(np.float32)


def get_body_xyz(env, body_name: str) -> np.ndarray:
    return env.unwrapped.data.body(body_name).xpos.copy().astype(np.float32)


def validate_tomato_layout(env) -> dict:
    can_xyz = get_body_xyz(env, "obj1")
    box_xyz = get_body_xyz(env, "target_box")
    goal_xyz = get_body_xyz(env, "goal_region")
    checks = [
        ("can_x", can_xyz[0], EXPECTED_CAN_X),
        ("box_x", box_xyz[0], EXPECTED_BOX_X),
        ("goal_x", goal_xyz[0], EXPECTED_GOAL_X),
    ]
    for name, actual, expected in checks:
        if abs(float(actual) - float(expected)) > 1e-3:
            raise RuntimeError(
                f"Unexpected tomato-can-box layout: {name}={actual:.6f}, expected {expected:.6f}"
            )
    return {
        "can_xyz": can_xyz.tolist(),
        "box_xyz": box_xyz.tolist(),
        "goal_xyz": goal_xyz.tolist(),
    }


def validate_random_tomato_layout(env) -> dict:
    can_xyz = get_body_xyz(env, "obj1")
    box_xyz = get_body_xyz(env, "target_box")
    goal_xyz = get_body_xyz(env, "goal_region")
    for name, actual, expected in [
        ("box_x", box_xyz[0], EXPECTED_BOX_X),
        ("goal_x", goal_xyz[0], EXPECTED_GOAL_X),
    ]:
        if abs(float(actual) - float(expected)) > 1e-3:
            raise RuntimeError(
                f"Unexpected random tomato layout: {name}={actual:.6f}, expected {expected:.6f}"
            )
    if not (RANDOM_CAN_X_MIN - 1e-3 <= float(can_xyz[0]) <= RANDOM_CAN_X_MAX + 1e-3):
        raise RuntimeError(
            f"Random tomato can x={can_xyz[0]:.6f} is outside recorded range "
            f"[{RANDOM_CAN_X_MIN:.6f}, {RANDOM_CAN_X_MAX:.6f}]"
        )
    if not (RANDOM_CAN_Y_MIN - 1e-3 <= float(can_xyz[1]) <= RANDOM_CAN_Y_MAX + 1e-3):
        raise RuntimeError(
            f"Random tomato can y={can_xyz[1]:.6f} is outside recorded range "
            f"[{RANDOM_CAN_Y_MIN:.6f}, {RANDOM_CAN_Y_MAX:.6f}]"
        )
    return {
        "can_xyz": can_xyz.tolist(),
        "box_xyz": box_xyz.tolist(),
        "goal_xyz": goal_xyz.tolist(),
        "can_sampling_bounds": {
            "x": [RANDOM_CAN_X_MIN, RANDOM_CAN_X_MAX],
            "y": [RANDOM_CAN_Y_MIN, RANDOM_CAN_Y_MAX],
        },
    }


def validate_cup_layout(env) -> dict:
    cup_xyz = get_body_xyz(env, "obj1")
    box_xyz = get_body_xyz(env, "target_box")
    goal_xyz = get_body_xyz(env, "goal_region")
    cube_xyz = {name: get_body_xyz(env, name).tolist() for name in ["cube1", "cube2", "cube3"]}

    box_checks = [
        ("box_x", box_xyz[0], EXPECTED_CUP_BOX_X),
        ("box_y", box_xyz[1], EXPECTED_CUP_BOX_Y),
        ("goal_x", goal_xyz[0], EXPECTED_CUP_BOX_X),
        ("goal_y", goal_xyz[1], EXPECTED_CUP_BOX_Y),
    ]
    for name, actual, expected in box_checks:
        if abs(float(actual) - float(expected)) > 1e-3:
            raise RuntimeError(
                f"Unexpected cup-box layout: {name}={actual:.6f}, expected {expected:.6f}"
            )

    if abs(float(cup_xyz[0]) - EXPECTED_CUP_X) > CUP_XY_JITTER + 2e-3:
        raise RuntimeError(
            f"Unexpected cup x={cup_xyz[0]:.6f}; expected around {EXPECTED_CUP_X:.6f} +/- {CUP_XY_JITTER:.3f}"
        )
    if abs(float(cup_xyz[1]) - EXPECTED_CUP_Y) > CUP_XY_JITTER + 2e-3:
        raise RuntimeError(
            f"Unexpected cup y={cup_xyz[1]:.6f}; expected around {EXPECTED_CUP_Y:.6f} +/- {CUP_XY_JITTER:.3f}"
        )

    return {
        "cup_xyz": cup_xyz.tolist(),
        "box_xyz": box_xyz.tolist(),
        "goal_xyz": goal_xyz.tolist(),
        "cube_xyz": cube_xyz,
    }


def validate_task_layout(env, task: str) -> dict:
    if task == "tomato_can_box":
        return validate_tomato_layout(env)
    if task == "tomato_can_box_random":
        return validate_random_tomato_layout(env)
    if task == "cup_box":
        return validate_cup_layout(env)
    raise ValueError(f"Unsupported task: {task}")


def task_object_body(task: str) -> str:
    if task in {"tomato_can_box", "tomato_can_box_random", "cup_box"}:
        return "obj1"
    raise ValueError(f"Unsupported task: {task}")


def task_extra_body_names(task: str) -> list[str]:
    if task == "cup_box":
        return ["cube1", "cube2", "cube3"]
    return []


def apply_pre_inference_forward_velocity(env, obs: dict, forward_vel: float) -> dict:
    if abs(float(forward_vel)) <= 1e-12:
        return obs

    unwrapped = env.unwrapped
    local_vel = np.array([float(forward_vel), 0.0, 0.0], dtype=np.float64)
    world_vel = unwrapped.convert_mobile_vel_frame(local_vel, world_to_local=False)
    for joint_name, vel in zip(unwrapped.mobile_joint_name_list, world_vel):
        unwrapped.data.joint(joint_name).qvel[0] = vel

    mujoco.mj_forward(unwrapped.model, unwrapped.data)
    obs = unwrapped._get_obs()
    obs["mobile_vel"] = local_vel.astype(np.float64)
    return obs


def apply_initial_settle(env, obs: dict, steps: int):
    if int(steps) <= 0:
        return obs, None

    settle_action = np.zeros(9, dtype=np.float64)
    settle_action[3:9] = np.asarray(obs["joint_pos"], dtype=np.float64)
    start_joint_pos = np.asarray(obs["joint_pos"], dtype=np.float64).copy()
    start_mobile_pose = np.asarray(obs.get("mobile_pose", []), dtype=np.float64).copy()

    info = None
    terminated = False
    truncated = False
    for _ in range(int(steps)):
        obs, _, terminated, truncated, info = env.step(settle_action.copy())
        if terminated or truncated:
            break

    settle_info = {
        "steps_requested": int(steps),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "start_joint_pos": start_joint_pos.tolist(),
        "final_joint_pos": np.asarray(obs["joint_pos"], dtype=np.float64).tolist(),
        "start_mobile_pose": start_mobile_pose.tolist(),
        "final_mobile_pose": np.asarray(obs.get("mobile_pose", []), dtype=np.float64).tolist(),
    }
    return obs, settle_info


def project_xy(points_xy: np.ndarray, width: int, height: int):
    if len(points_xy) == 0:
        return np.empty((0, 2), dtype=np.int32)
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    dx = max(0.2, x_max - x_min)
    dy = max(0.2, y_max - y_min)
    margin = 0.1
    x_min -= dx * margin
    x_max += dx * margin
    y_min -= dy * margin
    y_max += dy * margin

    u = (x - x_min) / (x_max - x_min + 1e-8) * (width - 1)
    v = (1.0 - (y - y_min) / (y_max - y_min + 1e-8)) * (height - 1)
    return np.stack([u, v], axis=1).astype(np.int32)


def draw_path_panel(base_xy: np.ndarray, width=480, height=360) -> np.ndarray:
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(panel, "Base XY trajectory", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)

    if len(base_xy) < 2:
        return panel

    px = project_xy(base_xy, width, height)
    cv2.polylines(panel, [px.reshape(-1, 1, 2)], isClosed=False, color=(20, 80, 220), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(panel, tuple(px[0]), 5, (0, 180, 0), -1, cv2.LINE_AA)
    cv2.circle(panel, tuple(px[-1]), 5, (0, 0, 255), -1, cv2.LINE_AA)
    return panel


def make_vis_frame(info: dict, base_xy: np.ndarray, ep_idx: int, step_idx: int, reward: float) -> np.ndarray:
    rgb_images = info.get("rgb_images", {})
    depth_images = info.get("depth_images", {})
    overview_rgb = rgb_images.get("task_overview")
    overview = (
        to_bgr_vis(overview_rgb)
        if overview_rgb is not None
        else blank_vis_panel("task overview unavailable")
    )
    head = to_bgr_vis(rgb_images["head"])
    hand = to_bgr_vis(rgb_images["hand"])
    hand_depth = depth_to_bgr_vis(depth_images["hand"])
    path_panel = draw_path_panel(base_xy)
    canvas = np.hstack([overview, head, hand, hand_depth, path_panel])
    label = f"ep={ep_idx:03d} step={step_idx:04d} reward={reward:.3f}"
    cv2.putText(canvas, label, (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, label, (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 40, 40), 1, cv2.LINE_AA)
    put_panel_label(canvas, "task overview", (20, 30))
    put_panel_label(canvas, "head rgb", (500, 30))
    put_panel_label(canvas, "hand rgb", (980, 30))
    put_panel_label(canvas, "hand depth", (1460, 30))
    return canvas


def save_episode_plots(ep_dir: Path, base_xy: np.ndarray, eef_xyz: np.ndarray, rewards: np.ndarray):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    if len(base_xy) > 0:
        axes[0].plot(base_xy[:, 0], base_xy[:, 1], color="#1f77b4")
        axes[0].scatter(base_xy[0, 0], base_xy[0, 1], c="green", s=30, label="start")
        axes[0].scatter(base_xy[-1, 0], base_xy[-1, 1], c="red", s=30, label="end")
        axes[0].legend(loc="best")
    axes[0].set_title("Base XY Path")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)

    if len(eef_xyz) > 0:
        axes[1].plot(eef_xyz[:, 0], eef_xyz[:, 2], color="#ff7f0e")
        axes[1].scatter(eef_xyz[0, 0], eef_xyz[0, 2], c="green", s=30)
        axes[1].scatter(eef_xyz[-1, 0], eef_xyz[-1, 2], c="red", s=30)
    axes[1].set_title("EEF XZ Path")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("z [m]")
    axes[1].grid(True, alpha=0.3)

    if len(rewards) > 0:
        axes[2].plot(rewards, color="#2ca02c")
    axes[2].set_title("Reward Over Time")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("reward")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(ep_dir / "trajectory_plot.png", dpi=160)
    plt.close(fig)


def save_summary_plots(out_dir: Path, episode_metrics: list[dict], all_base_paths: list[np.ndarray]):
    eps = [m["episode"] for m in episode_metrics]
    max_rewards = [m["max_reward"] for m in episode_metrics]
    total_rewards = [m["total_reward"] for m in episode_metrics]
    success = [m["success"] for m in episode_metrics]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(eps, max_rewards, color="#1f77b4", label="max_reward")
    axes[0].plot(eps, total_rewards, color="#ff7f0e", marker="o", label="total_reward")
    axes[0].set_title("Episode Rewards")
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].bar(eps, [int(s) for s in success], color="#2ca02c")
    axes[1].set_title("Episode Success (1/0)")
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("success")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "summary_metrics.png", dpi=160)
    plt.close(fig)

    plt.figure(figsize=(6, 6))
    for i, path_xy in enumerate(all_base_paths):
        if len(path_xy) == 0:
            continue
        plt.plot(path_xy[:, 0], path_xy[:, 1], alpha=0.8, label=f"ep{i:03d}")
    plt.title("All Base XY Trajectories")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    if len(all_base_paths) <= 10:
        plt.legend(loc="best", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "summary_base_paths.png", dpi=160)
    plt.close()


async def run_one_episode(
    ws,
    env,
    args,
    ep_idx: int,
    ep_dir: Path,
    episode_seed: int,
    world_idx: int,
    slot_perceptor=None,
    slot_perceptor_device: torch.device | None = None,
):
    if hasattr(env.unwrapped, "modify_world"):
        env.unwrapped.modify_world(cumulative_idx=world_idx)

    obs, info = env.reset(seed=episode_seed)
    task_layout = validate_task_layout(env, args.task)
    obs, settle_info = apply_initial_settle(env, obs, args.initial_settle_steps)
    if args.initial_settle_steps > 0:
        info = env.unwrapped._get_info()
    obs = apply_pre_inference_forward_velocity(
        env, obs, args.pre_inference_forward_vel
    )
    initial_base_xy = get_base_pose(env)[:2]
    predicted_initial_slots = None
    if args.state_layout == "no_wrench15_slots4x3" and args.slot_source == "predicted":
        if slot_perceptor is None or slot_perceptor_device is None:
            raise ValueError("Predicted 2D slots requested but no initial-scene slot perceptor was loaded")
        rgb_images = info.get("rgb_images", {})
        depth_images = info.get("depth_images", {})
        if "hand" not in rgb_images or "hand" not in depth_images:
            raise KeyError("Initial-scene slot prediction requires hand RGB and hand depth")
        predicted_initial_slots = predict_initial_scene_slots(
            slot_perceptor,
            slot_perceptor_device,
            rgb_images["hand"],
            depth_images["hand"],
            valid_threshold=args.slot_valid_threshold,
            depth_min=args.hand_depth_min,
            depth_max=args.hand_depth_max,
            head_rgb=rgb_images.get("head"),
        )

    obj_body = task_object_body(args.task)
    extra_body_names = task_extra_body_names(args.task)
    base_list = [get_base_pose(env)]
    eef_list = [get_eef_xyz(env)]
    obj_list = [get_body_xyz(env, obj_body)]
    extra_body_lists = {
        name: [get_body_xyz(env, name)]
        for name in extra_body_names
    }
    goal_list = [get_body_xyz(env, "goal_region")]
    reward_list = []
    action_list = []
    model_action_list = []
    image_mask = image_mask_from_mode(args.image_mask_mode)

    writer = None
    if args.save_video:
        video_path = ep_dir / "rollout.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.video_fps,
            (VIS_PANEL_SIZE[0] * VIS_PANEL_COUNT, VIS_PANEL_SIZE[1]),
        )

    terminated = False
    truncated = False
    step = 0
    total_reward = 0.0
    max_reward = -1e9
    first_success_step = None
    history_buffer = deque(maxlen=max(1, int(args.history_len)))
    close_guard_armed = False
    close_guard_used = False
    close_guard_remaining = 0
    close_guard_hold_gripper = None
    min_gripper_cmd_seen = np.inf
    max_gripper_cmd_seen = -np.inf
    max_base_vx_cmd_seen = -np.inf
    base_stop_low_vx_count = 0
    close_guard_events = []
    gripper_filter_value = float(obs["joint_pos"][-1])
    gripper_binary_is_open = (
        gripper_filter_value >= float(args.gripper_binary_threshold)
    )
    if args.gripper_filter in {"binary_hysteresis", "binary_task_latch"}:
        gripper_filter_value = (
            float(args.gripper_open_cmd)
            if gripper_binary_is_open
            else float(args.gripper_close_cmd)
        )
    gripper_filter_hold_count = int(args.gripper_min_hold_steps)
    gripper_filter_update_count = 0
    gripper_filter_events = []
    gripper_task_phase = "open" if gripper_binary_is_open else "closed"
    gripper_task_close_step = 0 if gripper_task_phase == "closed" else None
    inference_request_idx = 0

    while step < args.max_steps and not (terminated or truncated):
        state_now = build_rollout_state(
            env,
            obs,
            args.state_layout,
            mask_scene_slots=args.mask_scene_slots,
            slot_source=args.slot_source,
            predicted_initial_slots=predicted_initial_slots,
            initial_base_xy=initial_base_xy,
        )
        expected_state_dim = expected_state_dim_for_layout(args.state_layout)
        if state_now.shape[0] != expected_state_dim:
            raise ValueError(f"Expected state dim={expected_state_dim}, got {state_now.shape[0]}")

        rgb_images = info.get("rgb_images", {})
        depth_images = info.get("depth_images", {})
        if "head" not in rgb_images or "hand" not in rgb_images:
            raise KeyError("Missing required cameras in info['rgb_images']: need 'head' and 'hand'")
        if "hand" not in depth_images:
            raise KeyError("Missing required hand depth in info['depth_images']['hand']")

        history_buffer.append(
            {
                "state": state_now.copy(),
                "head_rgb": np.asarray(rgb_images["head"], dtype=np.uint8),
                "hand_rgb": np.asarray(rgb_images["hand"], dtype=np.uint8),
                "hand_depth": np.asarray(depth_images["hand"], dtype=np.float32),
            }
        )
        hist_items = list(history_buffer)
        state, head_rgb, hand_rgb, hand_depth = aggregate_history(hist_items, args.history_weight_mode)

        head_img = preprocess_rgb_for_server(head_rgb)
        hand_img = preprocess_rgb_for_server(hand_rgb)
        if args.third_view == "hand_depth":
            third_img = preprocess_depth_for_server(
                hand_depth,
                min_depth=args.hand_depth_min,
                max_depth=args.hand_depth_max,
                encoding=args.hand_depth_encoding,
                colormap=args.hand_depth_colormap,
            )
        else:
            third_img = np.zeros((448, 448, 3), dtype=np.uint8)

        req = {
            "image": [head_img.tolist(), hand_img.tolist(), third_img.tolist()],
            "image_mask": image_mask,
            "state": state.tolist(),
            "action_mask": [[1] * 9],
            "prompt": args.prompt,
        }
        if args.inference_seed is not None:
            req["inference_seed"] = (
                int(args.inference_seed) + int(ep_idx) * 10_000 + inference_request_idx
            )

        await ws.send(json.dumps(req))
        action_chunk = np.asarray(json.loads(await ws.recv()), dtype=np.float32)
        inference_request_idx += 1

        if action_chunk.ndim != 2 or action_chunk.shape[1] != 9:
            raise ValueError(f"Expected action chunk shape [T,9], got {action_chunk.shape}")

        take_n = min(args.exec_horizon, action_chunk.shape[0], args.max_steps - step)
        for i in range(take_n):
            model_action = action_chunk[i].astype(np.float64)
            base_vx_cmd = float(model_action[0])
            gripper_cmd = float(model_action[8])
            min_gripper_cmd_seen = min(min_gripper_cmd_seen, gripper_cmd)
            max_gripper_cmd_seen = max(max_gripper_cmd_seen, gripper_cmd)
            max_base_vx_cmd_seen = max(max_base_vx_cmd_seen, base_vx_cmd)

            if (
                args.close_guard
                and not close_guard_used
                and step >= args.close_guard_min_step
            ):
                if args.close_guard_trigger_direction == "increase":
                    if gripper_cmd <= args.close_guard_arm_below:
                        close_guard_armed = True
                elif args.close_guard_trigger_direction == "decrease":
                    if gripper_cmd >= args.close_guard_arm_above:
                        close_guard_armed = True
                elif args.close_guard_trigger_direction == "base_stop":
                    if base_vx_cmd >= args.close_guard_base_arm_vx:
                        close_guard_armed = True
                else:
                    raise ValueError(
                        f"Unsupported close_guard_trigger_direction: {args.close_guard_trigger_direction}"
                    )

            if args.close_guard and not close_guard_used and close_guard_armed:
                if args.close_guard_trigger_direction == "increase":
                    closing_by_abs = gripper_cmd >= args.close_guard_trigger_abs
                    closing_by_delta = (
                        gripper_cmd - min_gripper_cmd_seen
                        >= args.close_guard_trigger_delta
                    )
                    hold_gripper_cmd = float(obs["joint_pos"][-1])
                elif args.close_guard_trigger_direction == "decrease":
                    closing_by_abs = gripper_cmd <= args.close_guard_trigger_below
                    closing_by_delta = (
                        max_gripper_cmd_seen - gripper_cmd
                        >= args.close_guard_trigger_delta
                    )
                    # In decrease mode, high command corresponds to the open pre-grasp state.
                    hold_gripper_cmd = float(max_gripper_cmd_seen)
                elif args.close_guard_trigger_direction == "base_stop":
                    if base_vx_cmd <= args.close_guard_base_stop_vx:
                        base_stop_low_vx_count += 1
                    else:
                        base_stop_low_vx_count = 0
                    closing_by_abs = (
                        base_stop_low_vx_count
                        >= int(args.close_guard_base_stop_consecutive)
                    )
                    closing_by_delta = (
                        max_base_vx_cmd_seen - base_vx_cmd
                        >= args.close_guard_base_arm_vx
                    )
                    hold_gripper_cmd = float(obs["joint_pos"][-1])
                else:
                    raise ValueError(
                        f"Unsupported close_guard_trigger_direction: {args.close_guard_trigger_direction}"
                    )

                if closing_by_abs and closing_by_delta:
                    close_guard_used = True
                    close_guard_remaining = max(0, int(args.close_guard_steps))
                    close_guard_hold_gripper = hold_gripper_cmd
                    close_guard_events.append(
                        {
                            "trigger_step": int(step),
                            "trigger_direction": args.close_guard_trigger_direction,
                            "model_gripper_cmd": gripper_cmd,
                            "model_base_vx_cmd": base_vx_cmd,
                            "hold_gripper_cmd": close_guard_hold_gripper,
                            "min_gripper_cmd_seen": float(min_gripper_cmd_seen),
                            "max_gripper_cmd_seen": float(max_gripper_cmd_seen),
                            "max_base_vx_cmd_seen": float(max_base_vx_cmd_seen),
                            "base_stop_low_vx_count": int(base_stop_low_vx_count),
                            "forward_vel": float(args.close_guard_forward_vel),
                            "nudge_steps": int(close_guard_remaining),
                        }
                    )

            while (
                args.close_guard
                and close_guard_remaining > 0
                and step < args.max_steps
                and not (terminated or truncated)
            ):
                hold_joint_pos = np.asarray(obs["joint_pos"], dtype=np.float64)
                action = np.zeros(9, dtype=np.float64)
                action[0] = float(args.close_guard_forward_vel)
                action[3:9] = hold_joint_pos
                if close_guard_hold_gripper is not None:
                    action[8] = float(close_guard_hold_gripper)
                close_guard_remaining -= 1

                obs, reward, terminated, truncated, info = env.step(action)
                reward_f = float(reward)

                step += 1
                total_reward += reward_f
                max_reward = max(max_reward, reward_f)
                if first_success_step is None and reward_f >= args.success_reward:
                    first_success_step = int(step)
                reward_list.append(reward_f)
                action_list.append(action.astype(np.float32))
                model_action_list.append(action.astype(np.float32))
                base_list.append(get_base_pose(env))
                eef_list.append(get_eef_xyz(env))
                obj_list.append(get_body_xyz(env, obj_body))
                for body_name, body_list in extra_body_lists.items():
                    body_list.append(get_body_xyz(env, body_name))
                goal_list.append(get_body_xyz(env, "goal_region"))

                if writer is not None:
                    frame = make_vis_frame(
                        info=info,
                        base_xy=np.asarray(base_list)[:, :2],
                        ep_idx=ep_idx,
                        step_idx=step,
                        reward=reward_f,
                    )
                    writer.write(frame)

            if terminated or truncated or step >= args.max_steps:
                break

            action = model_action.copy()
            if args.gripper_filter != "none":
                raw_gripper_cmd = float(model_action[8])
                if step < int(args.gripper_filter_warmup_steps):
                    filtered_gripper_cmd = gripper_filter_value
                elif args.gripper_filter == "ema":
                    alpha = float(np.clip(args.gripper_ema_alpha, 0.0, 1.0))
                    filtered_gripper_cmd = (
                        alpha * raw_gripper_cmd + (1.0 - alpha) * gripper_filter_value
                    )
                    if abs(filtered_gripper_cmd - gripper_filter_value) > 1e-6:
                        gripper_filter_update_count += 1
                    gripper_filter_value = filtered_gripper_cmd
                elif args.gripper_filter == "hold_deadband":
                    gripper_filter_hold_count += 1
                    can_update = gripper_filter_hold_count >= int(args.gripper_min_hold_steps)
                    should_update = (
                        abs(raw_gripper_cmd - gripper_filter_value)
                        >= float(args.gripper_deadband)
                    )
                    if can_update and should_update:
                        old_gripper_cmd = gripper_filter_value
                        gripper_filter_value = raw_gripper_cmd
                        gripper_filter_hold_count = 0
                        gripper_filter_update_count += 1
                        gripper_filter_events.append(
                            {
                                "step": int(step),
                                "old_cmd": float(old_gripper_cmd),
                                "new_cmd": float(gripper_filter_value),
                                "model_cmd": float(raw_gripper_cmd),
                            }
                        )
                    filtered_gripper_cmd = gripper_filter_value
                elif args.gripper_filter == "binary_hysteresis":
                    gripper_filter_hold_count += 1
                    threshold = float(args.gripper_binary_threshold)
                    margin = max(0.0, float(args.gripper_hysteresis_margin))
                    open_threshold = threshold + margin
                    close_threshold = threshold - margin
                    can_update = gripper_filter_hold_count >= int(args.gripper_min_hold_steps)

                    requested_is_open = gripper_binary_is_open
                    if gripper_binary_is_open:
                        if raw_gripper_cmd <= close_threshold:
                            requested_is_open = False
                    else:
                        if raw_gripper_cmd >= open_threshold:
                            requested_is_open = True

                    if can_update and requested_is_open != gripper_binary_is_open:
                        old_gripper_cmd = gripper_filter_value
                        gripper_binary_is_open = requested_is_open
                        gripper_filter_value = (
                            float(args.gripper_open_cmd)
                            if gripper_binary_is_open
                            else float(args.gripper_close_cmd)
                        )
                        gripper_filter_hold_count = 0
                        gripper_filter_update_count += 1
                        gripper_filter_events.append(
                            {
                                "step": int(step),
                                "old_cmd": float(old_gripper_cmd),
                                "new_cmd": float(gripper_filter_value),
                                "model_cmd": float(raw_gripper_cmd),
                                "binary_is_open": bool(gripper_binary_is_open),
                                "open_threshold": float(open_threshold),
                                "close_threshold": float(close_threshold),
                            }
                        )
                    filtered_gripper_cmd = gripper_filter_value
                elif args.gripper_filter == "binary_task_latch":
                    threshold = float(args.gripper_binary_threshold)
                    margin = max(0.0, float(args.gripper_hysteresis_margin))
                    open_threshold = threshold + margin
                    close_threshold = threshold - margin
                    old_gripper_cmd = gripper_filter_value
                    previous_phase = gripper_task_phase

                    if gripper_task_phase == "open" and raw_gripper_cmd <= close_threshold:
                        gripper_task_phase = "closed"
                        gripper_task_close_step = int(step)
                        gripper_binary_is_open = False
                        gripper_filter_value = float(args.gripper_close_cmd)
                    elif (
                        gripper_task_phase == "closed"
                        and gripper_task_close_step is not None
                        and int(step) - int(gripper_task_close_step)
                        >= int(args.gripper_close_min_hold_steps)
                        and raw_gripper_cmd >= open_threshold
                    ):
                        gripper_task_phase = "released"
                        gripper_binary_is_open = True
                        gripper_filter_value = float(args.gripper_open_cmd)

                    if gripper_task_phase != previous_phase:
                        gripper_filter_update_count += 1
                        gripper_filter_events.append(
                            {
                                "step": int(step),
                                "old_cmd": float(old_gripper_cmd),
                                "new_cmd": float(gripper_filter_value),
                                "model_cmd": float(raw_gripper_cmd),
                                "task_phase": gripper_task_phase,
                                "closed_since_step": gripper_task_close_step,
                                "open_threshold": float(open_threshold),
                                "close_threshold": float(close_threshold),
                            }
                        )
                    filtered_gripper_cmd = gripper_filter_value
                else:
                    raise ValueError(f"Unsupported gripper_filter: {args.gripper_filter}")
                action[8] = float(filtered_gripper_cmd)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_f = float(reward)

            step += 1
            total_reward += reward_f
            max_reward = max(max_reward, reward_f)
            if first_success_step is None and reward_f >= args.success_reward:
                first_success_step = int(step)
            reward_list.append(reward_f)
            action_list.append(action.astype(np.float32))
            model_action_list.append(model_action.astype(np.float32))
            base_list.append(get_base_pose(env))
            eef_list.append(get_eef_xyz(env))
            obj_list.append(get_body_xyz(env, obj_body))
            for body_name, body_list in extra_body_lists.items():
                body_list.append(get_body_xyz(env, body_name))
            goal_list.append(get_body_xyz(env, "goal_region"))

            if writer is not None:
                frame = make_vis_frame(
                    info=info,
                    base_xy=np.asarray(base_list)[:, :2],
                    ep_idx=ep_idx,
                    step_idx=step,
                    reward=reward_f,
                )
                writer.write(frame)

            if terminated or truncated:
                break

    if writer is not None:
        writer.release()

    base_arr = np.asarray(base_list, dtype=np.float32)
    eef_arr = np.asarray(eef_list, dtype=np.float32)
    obj_arr = np.asarray(obj_list, dtype=np.float32)
    goal_arr = np.asarray(goal_list, dtype=np.float32)
    extra_body_arrs = {
        name: np.asarray(body_list, dtype=np.float32)
        for name, body_list in extra_body_lists.items()
    }
    rewards_arr = np.asarray(reward_list, dtype=np.float32)
    actions_arr = np.asarray(action_list, dtype=np.float32) if len(action_list) else np.zeros((0, 9), dtype=np.float32)
    model_actions_arr = (
        np.asarray(model_action_list, dtype=np.float32)
        if len(model_action_list)
        else np.zeros((0, 9), dtype=np.float32)
    )

    trajectory_payload = {
        "base_pose": base_arr,
        "eef_xyz": eef_arr,
        "obj_xyz": obj_arr,
        "goal_xyz": goal_arr,
        "obj_goal_abs": np.abs(obj_arr - goal_arr),
        "rewards": rewards_arr,
        "actions": actions_arr,
        "model_actions": model_actions_arr,
    }
    for body_name, body_arr in extra_body_arrs.items():
        trajectory_payload[f"{body_name}_xyz"] = body_arr
        trajectory_payload[f"{body_name}_goal_abs"] = np.abs(body_arr - goal_arr)
    if extra_body_arrs:
        trajectory_payload["extra_body_names"] = np.asarray(extra_body_names)
    np.savez_compressed(ep_dir / "trajectory_data.npz", **trajectory_payload)
    save_episode_plots(ep_dir, base_arr[:, :2], eef_arr, rewards_arr)

    episode_summary = {
        "episode": ep_idx,
        "task": args.task,
        "seed": int(episode_seed),
        "world_idx": int(world_idx),
        "steps": int(step),
        "total_reward": float(total_reward),
        "max_reward": float(max_reward if max_reward > -1e8 else 0.0),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": bool(max_reward >= args.success_reward),
        "first_success_step": first_success_step,
        "task_layout": task_layout,
        "initial_settle": settle_info,
        "slot_source": args.slot_source,
        "initial_base_xy": initial_base_xy.tolist(),
        "predicted_initial_slots": (
            predicted_initial_slots.tolist() if predicted_initial_slots is not None else None
        ),
        "tracked_object_body": obj_body,
        "final_obj_xyz": obj_arr[-1].tolist(),
        "final_goal_xyz": goal_arr[-1].tolist(),
        "final_obj_goal_abs": np.abs(obj_arr[-1] - goal_arr[-1]).tolist(),
        "final_extra_body_xyz": {
            name: body_arr[-1].tolist()
            for name, body_arr in extra_body_arrs.items()
        },
        "final_extra_body_goal_abs": {
            name: np.abs(body_arr[-1] - goal_arr[-1]).tolist()
            for name, body_arr in extra_body_arrs.items()
        },
        "close_guard_used": bool(close_guard_used),
        "close_guard_events": close_guard_events,
        "gripper_filter": args.gripper_filter,
        "gripper_filter_update_count": int(gripper_filter_update_count),
        "gripper_filter_events": gripper_filter_events,
        "gripper_task_final_phase": gripper_task_phase,
        "gripper_exec_spikes_gt005": count_gripper_spikes(actions_arr, threshold=0.05),
        "gripper_model_spikes_gt005": count_gripper_spikes(model_actions_arr, threshold=0.05),
        "gripper_exec_crossings_025": count_gripper_threshold_crossings(actions_arr, 0.25),
        "gripper_model_crossings_025": count_gripper_threshold_crossings(model_actions_arr, 0.25),
    }
    with open(ep_dir / "episode_summary.json", "w", encoding="utf-8") as f:
        json.dump(episode_summary, f, indent=2, ensure_ascii=False)

    return episode_summary, base_arr[:, :2]


async def run_batch(args):
    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("rollouts") / f"hsr_{args.task}_flash_{ts}"
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_config = {
        "task": args.task,
        "uri": args.uri,
        "env_id": args.env_id,
        "prompt": args.prompt,
        "max_steps": int(args.max_steps),
        "exec_horizon": int(args.exec_horizon),
        "history_len": int(args.history_len),
        "history_weight_mode": args.history_weight_mode,
        "inference_seed": args.inference_seed,
        "third_view": args.third_view,
        "image_mask_mode": args.image_mask_mode,
        "image_mask": image_mask_from_mode(args.image_mask_mode),
        "state_layout": args.state_layout,
        "state_dim": expected_state_dim_for_layout(args.state_layout),
        "mask_scene_slots": bool(args.mask_scene_slots),
        "slot_source": args.slot_source,
        "initial_scene_slot_ckpt": (
            str(args.initial_scene_slot_ckpt) if args.initial_scene_slot_ckpt is not None else None
        ),
        "slot_perceptor_device": args.slot_perceptor_device,
        "slot_valid_threshold": float(args.slot_valid_threshold),
        "hand_depth_min": args.hand_depth_min,
        "hand_depth_max": args.hand_depth_max,
        "hand_depth_encoding": args.hand_depth_encoding,
        "hand_depth_colormap": bool(args.hand_depth_colormap),
        "pre_inference_forward_vel": float(args.pre_inference_forward_vel),
        "initial_settle_steps": int(args.initial_settle_steps),
        "close_guard": bool(args.close_guard),
        "close_guard_forward_vel": float(args.close_guard_forward_vel),
        "close_guard_steps": int(args.close_guard_steps),
        "close_guard_min_step": int(args.close_guard_min_step),
        "close_guard_trigger_direction": args.close_guard_trigger_direction,
        "close_guard_base_arm_vx": float(args.close_guard_base_arm_vx),
        "close_guard_base_stop_vx": float(args.close_guard_base_stop_vx),
        "close_guard_base_stop_consecutive": int(args.close_guard_base_stop_consecutive),
        "close_guard_arm_below": float(args.close_guard_arm_below),
        "close_guard_arm_above": float(args.close_guard_arm_above),
        "close_guard_trigger_abs": float(args.close_guard_trigger_abs),
        "close_guard_trigger_below": float(args.close_guard_trigger_below),
        "close_guard_trigger_delta": float(args.close_guard_trigger_delta),
        "gripper_filter": args.gripper_filter,
        "gripper_deadband": float(args.gripper_deadband),
        "gripper_min_hold_steps": int(args.gripper_min_hold_steps),
        "gripper_close_min_hold_steps": int(args.gripper_close_min_hold_steps),
        "gripper_filter_warmup_steps": int(args.gripper_filter_warmup_steps),
        "gripper_ema_alpha": float(args.gripper_ema_alpha),
        "gripper_binary_threshold": float(args.gripper_binary_threshold),
        "gripper_hysteresis_margin": float(args.gripper_hysteresis_margin),
        "gripper_open_cmd": float(args.gripper_open_cmd),
        "gripper_close_cmd": float(args.gripper_close_cmd),
        "success_reward": float(args.success_reward),
        "expected_layout": (
            {
                "can_x": EXPECTED_CAN_X,
                "box_x": EXPECTED_BOX_X,
                "goal_x": EXPECTED_GOAL_X,
            }
            if args.task == "tomato_can_box"
            else {
                "can_x_range": [RANDOM_CAN_X_MIN, RANDOM_CAN_X_MAX],
                "can_y_range": [RANDOM_CAN_Y_MIN, RANDOM_CAN_Y_MAX],
                "box_x": EXPECTED_BOX_X,
                "goal_x": EXPECTED_GOAL_X,
            }
            if args.task == "tomato_can_box_random"
            else {
                "cup_x": EXPECTED_CUP_X,
                "cup_y": EXPECTED_CUP_Y,
                "cup_xy_jitter": CUP_XY_JITTER,
                "box_x": EXPECTED_CUP_BOX_X,
                "box_y": EXPECTED_CUP_BOX_Y,
            }
        ),
        "fixed_eval_seeds": [int(s) for s in args.eval_seeds],
        "world_indices": [int(w) for w in args.eval_world_indices],
    }
    with open(out_dir / "eval_config.json", "w", encoding="utf-8") as f:
        json.dump(eval_config, f, indent=2, ensure_ascii=False)

    if args.slot_source == "predicted" and args.state_layout != "no_wrench15_slots4x3":
        raise ValueError("--slot_source=predicted requires --state_layout=no_wrench15_slots4x3")
    slot_perceptor = None
    slot_perceptor_device = None
    if args.slot_source == "predicted":
        slot_perceptor, slot_perceptor_device = load_initial_scene_slot_perceptor(args)
        print(f"[Batch] loaded initial-scene slot perceptor from {args.initial_scene_slot_ckpt}")

    env = gym.make(args.env_id, render_mode="rgb_array")
    episode_metrics = []
    all_base_paths = []

    try:
        async with websockets.connect(args.uri, max_size=100_000_000) as ws:
            print(f"[Batch] connected to {args.uri}")
            print(f"[Batch] task={args.task}")
            print(f"[Batch] env_id={args.env_id}")
            print(f"[Batch] image_mask={eval_config['image_mask']} third_view={args.third_view}")
            for ep in range(args.episodes):
                ep_dir = out_dir / f"episode_{ep:03d}"
                ep_dir.mkdir(parents=True, exist_ok=True)
                metric, path_xy = await run_one_episode(
                    ws,
                    env,
                    args,
                    ep,
                    ep_dir,
                    episode_seed=args.eval_seeds[ep],
                    world_idx=args.eval_world_indices[ep],
                    slot_perceptor=slot_perceptor,
                    slot_perceptor_device=slot_perceptor_device,
                )
                episode_metrics.append(metric)
                all_base_paths.append(path_xy)
                print(
                    f"[Batch] ep={ep:03d} seed={metric['seed']} world_idx={metric['world_idx']} "
                    f"steps={metric['steps']} total_reward={metric['total_reward']:.3f} "
                    f"max_reward={metric['max_reward']:.3f} success={metric['success']}"
                )
    finally:
        try:
            env.close()
        except Exception:
            pass

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(episode_metrics, f, indent=2, ensure_ascii=False)

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "task",
                "seed",
                "world_idx",
                "steps",
                "total_reward",
                "max_reward",
                "terminated",
                "truncated",
                "success",
                "close_guard_used",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(episode_metrics)

    save_summary_plots(out_dir, episode_metrics, all_base_paths)

    success_rate = float(np.mean([m["success"] for m in episode_metrics])) if episode_metrics else 0.0
    print("")
    print(f"[Batch] output_dir={out_dir}")
    print(f"[Batch] episodes={len(episode_metrics)} success_rate={success_rate:.3f}")
    print(f"[Batch] summary_csv={out_dir / 'summary.csv'}")
    return out_dir


def main():
    args = parse_args()
    resolve_eval_schedule(args)
    asyncio.run(run_batch(args))


if __name__ == "__main__":
    main()
