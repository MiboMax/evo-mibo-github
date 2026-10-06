#!/usr/bin/env python3
"""
Convert RoboManipBaselines HSR `.rmb` episodes into an Evo-1 compatible
LeRobot v2.1-style local dataset directory.

Expected input layout:
  <rmb_root>/
    MujocoHsrTidyup_world0_000.rmb/
      main.rmb.hdf5
      head_rgb_image.rmb.mp4
      hand_rgb_image.rmb.mp4
      ...

Generated output layout:
  <out_root>/
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.head_rgb_image/episode_000000.mp4
    videos/chunk-000/observation.images.hand_rgb_image/episode_000000.mp4
    videos/chunk-000/observation.images.hand_depth_image/episode_000000.mp4  (optional)
    meta/tasks.jsonl
    meta/episodes.jsonl
    meta/episodes_stats.jsonl
    meta/stats.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import h5py
import numpy as np
import pandas as pd

try:
    import videoio
except ModuleNotFoundError:  # Optional: available in rmb_hsr, absent in some Evo envs.
    rmb_videoio_site = os.environ.get(
        "RMB_VIDEOIO_SITE",
        "/home/mibo/miniconda3/envs/rmb_hsr/lib/python3.10/site-packages",
    )
    if Path(rmb_videoio_site).exists():
        sys.path.insert(0, rmb_videoio_site)
        try:
            import videoio
        except ModuleNotFoundError:
            videoio = None
    else:
        videoio = None


STATE_KEYS_FULL21 = [
    "measured_joint_pos",       # 6
    "measured_joint_vel",       # 6
    "measured_eef_wrench",      # 6
    "measured_mobile_omni_vel", # 3
]

STATE_KEYS_NO_WRENCH15 = [
    "measured_joint_pos",       # 6
    "measured_joint_vel",       # 6
    "measured_mobile_omni_vel", # 3
]

STATE_KEYS_SCENE_SLOTS = [
    "measured_scene_slots",     # 4 * 7 flattened scene slots
]

ACTION_KEYS = [
    "command_mobile_omni_vel",  # 3: base vx, base vy, base yaw velocity
    "command_joint_pos",        # 6: 5 arm joints + hand_motor_joint
]

OPTIONAL_OBS_EEF_KEY = "measured_eef_pose"   # 7
OPTIONAL_CMD_EEF_KEY = "command_eef_pose"    # 7

RGB_VIDEO_MAP = {
    "observation.images.head_rgb_image": "head_rgb_image.rmb.mp4",
    "observation.images.hand_rgb_image": "hand_rgb_image.rmb.mp4",
}

HAND_DEPTH_VIEW = "observation.images.hand_depth_image"
HAND_DEPTH_SRC = "hand_depth_image.rmb.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RMB HSR dataset to LeRobot v2.1 style.")
    parser.add_argument(
        "--rmb-root",
        type=Path,
        required=True,
        help="Directory containing *.rmb episode folders.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output LeRobot-style dataset root.",
    )
    parser.add_argument(
        "--task-text",
        type=str,
        default="Pick up the bottles and place them into the matching containers.",
        help="Task text written into meta/tasks.jsonl.",
    )
    parser.add_argument(
        "--copy-videos",
        action="store_true",
        help="Copy videos instead of creating hard links/symlinks.",
    )
    parser.add_argument(
        "--normalize-video-fps",
        action="store_true",
        help=(
            "Rewrite RGB/depth videos using the fps implied by HDF5 timestamps. "
            "This keeps timestamp-based training decoders aligned across views."
        ),
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help=(
            "Force the exported dataset/video fps metadata to this value. "
            "When omitted, fps is inferred from HDF5 timestamps per episode."
        ),
    )
    parser.add_argument(
        "--skip-empty-episodes",
        action="store_true",
        help="Skip 0-frame episode folders instead of failing conversion.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    parser.add_argument(
        "--include-hand-depth",
        action="store_true",
        help="Export hand_depth_image as an optional third RGB image view.",
    )
    parser.add_argument(
        "--hand-depth-source",
        choices=("auto", "hdf5", "video"),
        default="auto",
        help=(
            "Metric hand-depth source. auto prefers HDF5 hand_depth_raw_mm when present "
            "and falls back to RMB uint16 video. Use hdf5 for real HSR raw recordings."
        ),
    )
    parser.add_argument(
        "--hand-depth-hdf5-key",
        type=str,
        default="hand_depth_raw_mm",
        help="HDF5 dataset key for real metric hand depth in uint16 millimeters.",
    )
    parser.add_argument(
        "--export-initial-hand-invdepth",
        action="store_true",
        help=(
            "Export one initial hand inverse-depth PNG and metric-depth NPZ per episode. "
            "These static per-episode assets are intended for an initial-scene vision/slot branch."
        ),
    )
    parser.add_argument(
        "--initial-hand-depth-frame-index",
        type=int,
        default=0,
        help="Recorded hand-depth frame index used for --export-initial-hand-invdepth.",
    )
    parser.add_argument(
        "--state-layout",
        choices=("no_wrench15", "full21", "no_wrench15_slots4x7", "no_wrench15_slots4x3"),
        default="no_wrench15",
        help=(
            "State vector layout. no_wrench15 = measured_joint_pos(6) + "
            "measured_joint_vel(6) + measured_mobile_omni_vel(3). "
            "full21 additionally includes measured_eef_wrench(6). "
            "no_wrench15_slots4x7 appends flattened measured_scene_slots(4*7). "
            "no_wrench15_slots4x3 appends only [dx,dy,valid] for each of four slots."
        ),
    )
    parser.add_argument(
        "--hand-depth-min",
        type=float,
        default=0.05,
        help=(
            "Depth near value [m]. For inverse encoding this maps to white. "
            "For linear encoding this maps to black. Default is tuned for HSR tomato hand depth."
        ),
    )
    parser.add_argument(
        "--hand-depth-max",
        type=float,
        default=2.0,
        help=(
            "Depth far value [m]. For inverse encoding this maps to black. "
            "For linear encoding this maps to white. Default suppresses far background."
        ),
    )
    parser.add_argument(
        "--hand-depth-encoding",
        choices=("inverse", "linear"),
        default="inverse",
        help=(
            "Depth-to-RGB encoding for the optional third view. inverse encodes closer pixels brighter "
            "using normalized inverse depth; linear keeps the previous farther-is-brighter visualization."
        ),
    )
    parser.add_argument(
        "--hand-depth-colormap",
        action="store_true",
        help="Apply OpenCV JET colormap after depth encoding instead of grayscale RGB.",
    )
    parser.add_argument(
        "--binarize-gripper-action",
        action="store_true",
        help="Binarize HSR action[8] gripper command after action concatenation.",
    )
    parser.add_argument(
        "--gripper-binary-threshold",
        type=float,
        default=0.25,
        help="Threshold for binarizing action[8]. Values above threshold map to open value.",
    )
    parser.add_argument(
        "--gripper-open-value",
        type=float,
        default=0.5,
        help="Binary open command value written to action[8]. HSR tomato data starts open at 0.5.",
    )
    parser.add_argument(
        "--gripper-close-value",
        type=float,
        default=-0.1,
        help="Binary close command value written to action[8]. For current HSR data low cluster median is about -0.1.",
    )
    return parser.parse_args()


def write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def reset_output_dir(out_root: Path, overwrite: bool) -> None:
    if out_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {out_root}. "
                f"Use --overwrite to replace it."
            )
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)


def list_episode_dirs(rmb_root: Path) -> List[Path]:
    episode_dirs = sorted(
        p for p in rmb_root.iterdir() if p.is_dir() and p.name.endswith(".rmb")
    )
    if not episode_dirs:
        raise FileNotFoundError(f"No *.rmb episode directories found under {rmb_root}")
    return episode_dirs


def get_episode_frame_count(episode_dir: Path) -> int:
    h5_path = episode_dir / "main.rmb.hdf5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing HDF5 file: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        if "time" not in f:
            raise KeyError(f"Missing HDF5 time key in {h5_path}")
        return int(f["time"].shape[0])


def ensure_same_length(arrs: List[np.ndarray], episode_dir: Path) -> int:
    lengths = [int(a.shape[0]) for a in arrs]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"Inconsistent frame lengths in {episode_dir}: {lengths}"
        )
    return lengths[0]


def link_or_copy(src: Path, dst: Path, copy_videos: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if copy_videos:
        shutil.copy2(src, dst)
        return

    # Prefer hard-link to save space and avoid broken symlink issues.
    try:
        os.link(src, dst)
        return
    except OSError:
        pass

    try:
        os.symlink(src, dst)
        return
    except OSError:
        pass

    # Final fallback.
    shutil.copy2(src, dst)


def infer_fps_from_timestamps(time_arr: np.ndarray) -> float:
    diffs = np.diff(np.asarray(time_arr, dtype=np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 30.0
    return float(1.0 / np.median(diffs))


def read_rgb_video(src_video: Path, expected_frames: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(src_video))
    frames = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"Failed to decode RGB video: {src_video}")
    if len(frames) != int(expected_frames):
        raise ValueError(
            f"RGB video length mismatch for {src_video}: "
            f"video={len(frames)}, expected={expected_frames}"
        )
    return np.stack(frames, axis=0)


def convert_rgb_video(
    src_video: Path,
    dst_video: Path,
    expected_frames: int,
    fps: float,
) -> None:
    frames_rgb = read_rgb_video(src_video, expected_frames=expected_frames)
    write_rgb_video(dst_video, frames_rgb, fps=fps)


def _read_uint16_depth_video(src_video: Path) -> np.ndarray:
    """Read RMB depth video and return metric depth in meters."""
    if videoio is not None:
        return (1e-3 * videoio.uint16read(str(src_video))).astype(np.float32)

    raise RuntimeError(
        "videoio is required for metric hand-depth conversion. "
        "Run with RMB_VIDEOIO_SITE pointing to the rmb_hsr site-packages directory, "
        "or use an environment where `import videoio` works."
    )


def _read_depth_scale_from_metadata(episode_dir: Path) -> float:
    metadata_path = episode_dir / "metadata.json"
    if not metadata_path.exists():
        return 1e-3

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    for section in ("d435", "hand_depth", "camera", "cameras"):
        value = metadata.get(section)
        if isinstance(value, dict) and "depth_scale" in value:
            return float(value["depth_scale"])
    if "depth_scale" in metadata:
        return float(metadata["depth_scale"])
    return 1e-3


def _read_hdf5_depth_mm(
    episode_dir: Path,
    hdf5_key: str,
    expected_frames: int,
) -> np.ndarray:
    """Read real HSR raw D405 depth from HDF5 and return metric depth in meters."""
    h5_path = episode_dir / "main.rmb.hdf5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing HDF5 file: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        if hdf5_key not in f:
            raise KeyError(f"Missing HDF5 hand-depth key `{hdf5_key}` in {h5_path}")
        raw_mm = f[hdf5_key][:].astype(np.uint16)

    if int(raw_mm.shape[0]) != int(expected_frames):
        raise ValueError(
            f"HDF5 hand depth length mismatch in {episode_dir}: "
            f"depth={raw_mm.shape[0]}, expected={expected_frames}"
        )

    depth_scale = _read_depth_scale_from_metadata(episode_dir)
    depth_m = raw_mm.astype(np.float32) * float(depth_scale)

    # RealSense raw depth uses 0 for invalid pixels; 65535 can appear as a saturated
    # sentinel in the raw stream and should not become a far metric point.
    invalid = (raw_mm == 0) | (raw_mm >= np.iinfo(np.uint16).max)
    depth_m[invalid] = 0.0
    depth_m[~np.isfinite(depth_m)] = 0.0
    return depth_m


def read_metric_hand_depth(
    episode_dir: Path,
    expected_frames: int,
    source: str,
    hdf5_key: str,
) -> np.ndarray:
    if source in ("auto", "hdf5"):
        h5_path = episode_dir / "main.rmb.hdf5"
        has_hdf5_depth = False
        if h5_path.exists():
            with h5py.File(h5_path, "r") as f:
                has_hdf5_depth = hdf5_key in f
        if has_hdf5_depth:
            return _read_hdf5_depth_mm(
                episode_dir=episode_dir,
                hdf5_key=hdf5_key,
                expected_frames=expected_frames,
            )
        if source == "hdf5":
            raise KeyError(f"Missing required HDF5 hand-depth key `{hdf5_key}` in {h5_path}")

    src_video = episode_dir / HAND_DEPTH_SRC
    if not src_video.exists():
        raise FileNotFoundError(f"Missing hand depth video: {src_video}")
    depth_seq = _read_uint16_depth_video(src_video)
    if int(depth_seq.shape[0]) != int(expected_frames):
        raise ValueError(
            f"Hand depth length mismatch in {episode_dir}: "
            f"depth={depth_seq.shape[0]}, expected={expected_frames}"
        )
    return depth_seq.astype(np.float32)


def _depth_to_rgb_frames(
    depth_seq: np.ndarray,
    depth_min: float | None,
    depth_max: float | None,
    encoding: str,
    use_colormap: bool,
) -> np.ndarray:
    valid = np.isfinite(depth_seq) & (depth_seq > 0)
    if not valid.any():
        raise ValueError("Depth sequence has no valid positive finite values")

    if depth_min is None:
        depth_min = float(np.percentile(depth_seq[valid], 1.0))
    if depth_max is None:
        depth_max = float(np.percentile(depth_seq[valid], 99.0))
    if depth_max <= depth_min:
        depth_max = depth_min + 1e-6

    clipped = np.clip(depth_seq, depth_min, depth_max)
    if encoding == "linear":
        norm = (clipped - depth_min) / (depth_max - depth_min)
    elif encoding == "inverse":
        inv = np.zeros_like(clipped, dtype=np.float32)
        inv[valid] = 1.0 / clipped[valid]
        inv_near = 1.0 / depth_min
        inv_far = 1.0 / depth_max
        norm = (inv - inv_far) / (inv_near - inv_far)
    else:
        raise ValueError(f"Unsupported hand depth encoding: {encoding}")

    norm = np.clip(norm, 0.0, 1.0)
    norm[~valid] = 0.0
    encoded = (norm * 255.0).astype(np.uint8)

    rgb_frames = []
    for frame in encoded:
        if use_colormap:
            # cv2 returns BGR; convert to RGB before writing through VideoWriter wrapper.
            colored = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
            rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        else:
            rgb = np.repeat(frame[..., None], 3, axis=2)
        rgb_frames.append(rgb)
    return np.stack(rgb_frames, axis=0)


def write_rgb_video(dst_video: Path, frames_rgb: np.ndarray, fps: float = 30.0) -> None:
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    if dst_video.exists() or dst_video.is_symlink():
        dst_video.unlink()
    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB frames with shape (T,H,W,3), got {frames_rgb.shape}")

    height, width = frames_rgb.shape[1:3]
    writer = cv2.VideoWriter(
        str(dst_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {dst_video}")
    try:
        for frame_rgb in frames_rgb:
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def convert_hand_depth_video(
    episode_dir: Path,
    episode_stem: str,
    chunk_name: str,
    out_root: Path,
    depth_source: str,
    depth_hdf5_key: str,
    depth_min: float | None,
    depth_max: float | None,
    depth_encoding: str,
    use_colormap: bool,
    expected_frames: int,
    fps: float,
) -> None:
    depth_seq = read_metric_hand_depth(
        episode_dir=episode_dir,
        expected_frames=expected_frames,
        source=depth_source,
        hdf5_key=depth_hdf5_key,
    )
    frames_rgb = _depth_to_rgb_frames(
        depth_seq=depth_seq,
        depth_min=depth_min,
        depth_max=depth_max,
        encoding=depth_encoding,
        use_colormap=use_colormap,
    )
    dst_video = out_root / "videos" / chunk_name / HAND_DEPTH_VIEW / f"{episode_stem}.mp4"
    write_rgb_video(dst_video, frames_rgb, fps=fps)


def export_initial_hand_invdepth(
    episode_dir: Path,
    episode_stem: str,
    out_root: Path,
    frame_index: int,
    depth_source: str,
    depth_hdf5_key: str,
    depth_min: float | None,
    depth_max: float | None,
    depth_encoding: str,
    use_colormap: bool,
    expected_frames: int,
) -> Dict:
    """Write one static initial-scene depth asset without duplicating it per action frame."""
    depth_seq = read_metric_hand_depth(
        episode_dir=episode_dir,
        expected_frames=expected_frames,
        source=depth_source,
        hdf5_key=depth_hdf5_key,
    )
    if not 0 <= int(frame_index) < int(depth_seq.shape[0]):
        raise IndexError(
            f"Initial hand depth frame index {frame_index} is outside "
            f"[0, {depth_seq.shape[0] - 1}] for {episode_dir}"
        )

    depth_frame_m = depth_seq[int(frame_index)].astype(np.float32)
    invdepth_rgb = _depth_to_rgb_frames(
        depth_seq=depth_frame_m[None, ...],
        depth_min=depth_min,
        depth_max=depth_max,
        encoding=depth_encoding,
        use_colormap=use_colormap,
    )[0]

    asset_root = out_root / "initial_scene"
    invdepth_path = asset_root / "hand_invdepth" / f"{episode_stem}.png"
    metric_depth_path = asset_root / "hand_depth_m" / f"{episode_stem}.npz"
    invdepth_path.parent.mkdir(parents=True, exist_ok=True)
    metric_depth_path.parent.mkdir(parents=True, exist_ok=True)

    # cv2.imwrite expects BGR; invdepth_rgb is RGB like the regular depth view.
    if not cv2.imwrite(str(invdepth_path), cv2.cvtColor(invdepth_rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write initial inverse-depth image: {invdepth_path}")
    np.savez_compressed(metric_depth_path, depth_m=depth_frame_m)

    return {
        "source_frame_index": int(frame_index),
        "hand_invdepth_file": str(invdepth_path.relative_to(out_root)),
        "hand_depth_m_file": str(metric_depth_path.relative_to(out_root)),
        "shape": [int(depth_frame_m.shape[0]), int(depth_frame_m.shape[1])],
        "encoding": depth_encoding,
        "depth_min": depth_min,
        "depth_max": depth_max,
        "colormap": bool(use_colormap),
        "depth_source": depth_source,
        "depth_hdf5_key": depth_hdf5_key if depth_source in ("auto", "hdf5") else None,
    }


def convert_episode(
    episode_dir: Path,
    episode_index: int,
    chunk_name: str,
    out_root: Path,
    copy_videos: bool,
    include_hand_depth: bool,
    export_initial_hand_invdepth_assets: bool,
    initial_hand_depth_frame_index: int,
    hand_depth_min: float | None,
    hand_depth_max: float | None,
    hand_depth_encoding: str,
    hand_depth_colormap: bool,
    hand_depth_source: str,
    hand_depth_hdf5_key: str,
    state_layout: str,
    normalize_video_fps: bool,
    target_fps: float | None,
    binarize_gripper_action: bool,
    gripper_binary_threshold: float,
    gripper_open_value: float,
    gripper_close_value: float,
) -> Dict:
    h5_path = episode_dir / "main.rmb.hdf5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing HDF5 file: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        time_arr = f["time"][:].astype(np.float32)
        if int(time_arr.shape[0]) == 0:
            raise ValueError(f"Empty episode with 0 frames: {episode_dir}")
        initial_scene_slots = (
            f["measured_scene_slots"][0].astype(np.float32)
            if "measured_scene_slots" in f
            else None
        )
        initial_mobile_pose = (
            f["measured_mobile_omni_pose"][0].astype(np.float32)
            if "measured_mobile_omni_pose" in f
            else None
        )
        if state_layout == "full21":
            state_keys = STATE_KEYS_FULL21
            expected_state_dim = 21
        elif state_layout == "no_wrench15":
            state_keys = STATE_KEYS_NO_WRENCH15
            expected_state_dim = 15
        elif state_layout == "no_wrench15_slots4x7":
            state_keys = STATE_KEYS_NO_WRENCH15 + STATE_KEYS_SCENE_SLOTS
            expected_state_dim = 43
        elif state_layout == "no_wrench15_slots4x3":
            state_keys = STATE_KEYS_NO_WRENCH15 + STATE_KEYS_SCENE_SLOTS
            expected_state_dim = 27
        else:
            raise ValueError(f"Unsupported state layout: {state_layout}")

        state_parts = []
        for key in state_keys:
            arr = f[key][:].astype(np.float32)
            if key == "measured_scene_slots":
                if arr.ndim != 3 or arr.shape[1:] != (4, 7):
                    raise ValueError(
                        f"Expected measured_scene_slots shape (T,4,7) in {episode_dir}, got {arr.shape}"
                    )
                if state_layout == "no_wrench15_slots4x3":
                    arr = arr[:, :, [0, 1, 6]].reshape(arr.shape[0], -1)
                else:
                    arr = arr.reshape(arr.shape[0], -1)
            state_parts.append(arr)
        action_parts = [f[k][:].astype(np.float32) for k in ACTION_KEYS]
        obs_eef = (
            f[OPTIONAL_OBS_EEF_KEY][:].astype(np.float32)
            if OPTIONAL_OBS_EEF_KEY in f
            else None
        )
        cmd_eef = (
            f[OPTIONAL_CMD_EEF_KEY][:].astype(np.float32)
            if OPTIONAL_CMD_EEF_KEY in f
            else None
        )

    extra_parts = [time_arr]
    if obs_eef is not None:
        extra_parts.append(obs_eef)
    if cmd_eef is not None:
        extra_parts.append(cmd_eef)
    n_frames = ensure_same_length(state_parts + action_parts + extra_parts, episode_dir)

    state = np.concatenate(state_parts, axis=1)   # (T, expected_state_dim)
    action = np.concatenate(action_parts, axis=1) # (T, 9)

    if state.shape[1] != expected_state_dim:
        raise ValueError(f"Unexpected state dim in {episode_dir}: {state.shape}")
    if action.shape[1] != 9:
        raise ValueError(f"Unexpected action dim in {episode_dir}: {action.shape}")
    if binarize_gripper_action:
        gripper_raw = action[:, 8].copy()
        action[:, 8] = np.where(
            gripper_raw > float(gripper_binary_threshold),
            float(gripper_open_value),
            float(gripper_close_value),
        ).astype(np.float32)
    video_fps = float(target_fps) if target_fps is not None else infer_fps_from_timestamps(time_arr)

    episode_stem = f"episode_{episode_index:06d}"
    parquet_path = out_root / "data" / chunk_name / f"{episode_stem}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    table = {
        "episode_index": np.full(n_frames, episode_index, dtype=np.int32),
        "frame_index": np.arange(n_frames, dtype=np.int32),
        "timestamp": time_arr.tolist(),
        "task_index": np.zeros(n_frames, dtype=np.int32),
        "observation.state": state.tolist(),
        "action": action.tolist(),
    }
    if obs_eef is not None:
        table["observation.eef_pose"] = obs_eef.tolist()
    if cmd_eef is not None:
        table["command.eef_pose"] = cmd_eef.tolist()

    df = pd.DataFrame(table)
    df.to_parquet(parquet_path, index=False)

    for view_folder, src_name in RGB_VIDEO_MAP.items():
        src_video = episode_dir / src_name
        if not src_video.exists():
            raise FileNotFoundError(f"Missing video file: {src_video}")
        dst_video = (
            out_root
            / "videos"
            / chunk_name
            / view_folder
            / f"{episode_stem}.mp4"
        )
        if normalize_video_fps:
            convert_rgb_video(
                src_video=src_video,
                dst_video=dst_video,
                expected_frames=n_frames,
                fps=video_fps,
            )
        else:
            link_or_copy(src_video, dst_video, copy_videos=copy_videos)

    if include_hand_depth:
        convert_hand_depth_video(
            episode_dir=episode_dir,
            episode_stem=episode_stem,
            chunk_name=chunk_name,
            out_root=out_root,
            depth_source=hand_depth_source,
            depth_hdf5_key=hand_depth_hdf5_key,
            depth_min=hand_depth_min,
            depth_max=hand_depth_max,
            depth_encoding=hand_depth_encoding,
            use_colormap=hand_depth_colormap,
            expected_frames=n_frames,
            fps=video_fps,
        )

    initial_scene = None
    if export_initial_hand_invdepth_assets:
        initial_scene = export_initial_hand_invdepth(
            episode_dir=episode_dir,
            episode_stem=episode_stem,
            out_root=out_root,
            frame_index=initial_hand_depth_frame_index,
            depth_source=hand_depth_source,
            depth_hdf5_key=hand_depth_hdf5_key,
            depth_min=hand_depth_min,
            depth_max=hand_depth_max,
            depth_encoding=hand_depth_encoding,
            use_colormap=hand_depth_colormap,
            expected_frames=n_frames,
        )

    state_min = state.min(axis=0).tolist()
    state_max = state.max(axis=0).tolist()
    action_min = action.min(axis=0).tolist()
    action_max = action.max(axis=0).tolist()
    obs_eef_min = obs_eef.min(axis=0).tolist() if obs_eef is not None else None
    obs_eef_max = obs_eef.max(axis=0).tolist() if obs_eef is not None else None
    cmd_eef_min = cmd_eef.min(axis=0).tolist() if cmd_eef is not None else None
    cmd_eef_max = cmd_eef.max(axis=0).tolist() if cmd_eef is not None else None

    return {
        "episode_index": episode_index,
        "episode_stem": episode_stem,
        "length": int(n_frames),
        "state_min": state_min,
        "state_max": state_max,
        "action_min": action_min,
        "action_max": action_max,
        "obs_eef_min": obs_eef_min,
        "obs_eef_max": obs_eef_max,
        "cmd_eef_min": cmd_eef_min,
        "cmd_eef_max": cmd_eef_max,
        "video_fps": video_fps,
        "initial_scene": initial_scene,
        "initial_scene_slots": (
            initial_scene_slots.tolist() if initial_scene_slots is not None else None
        ),
        "initial_mobile_pose": (
            initial_mobile_pose.tolist() if initial_mobile_pose is not None else None
        ),
    }


def main() -> None:
    args = parse_args()
    rmb_root: Path = args.rmb_root.expanduser().resolve()
    out_root: Path = args.out_root.expanduser().resolve()
    chunk_name = "chunk-000"

    reset_output_dir(out_root, overwrite=args.overwrite)
    episode_dirs = list_episode_dirs(rmb_root)

    episode_infos = []
    skipped_empty = []
    for source_i, ep_dir in enumerate(episode_dirs):
        if args.skip_empty_episodes and get_episode_frame_count(ep_dir) == 0:
            skipped_empty.append(ep_dir.name)
            print(f"[skip] empty episode: {ep_dir.name}")
            continue
        info = convert_episode(
            episode_dir=ep_dir,
            episode_index=len(episode_infos),
            chunk_name=chunk_name,
            out_root=out_root,
            copy_videos=args.copy_videos,
            include_hand_depth=args.include_hand_depth,
            export_initial_hand_invdepth_assets=args.export_initial_hand_invdepth,
            initial_hand_depth_frame_index=args.initial_hand_depth_frame_index,
            hand_depth_min=args.hand_depth_min,
            hand_depth_max=args.hand_depth_max,
            hand_depth_encoding=args.hand_depth_encoding,
            hand_depth_colormap=args.hand_depth_colormap,
            hand_depth_source=args.hand_depth_source,
            hand_depth_hdf5_key=args.hand_depth_hdf5_key,
            state_layout=args.state_layout,
            normalize_video_fps=args.normalize_video_fps,
            target_fps=args.target_fps,
            binarize_gripper_action=args.binarize_gripper_action,
            gripper_binary_threshold=args.gripper_binary_threshold,
            gripper_open_value=args.gripper_open_value,
            gripper_close_value=args.gripper_close_value,
        )
        episode_infos.append(info)
        converted = len(episode_infos)
        if converted % 10 == 0 or (source_i + 1) == len(episode_dirs):
            print(f"[{source_i + 1}/{len(episode_dirs)}] converted {ep_dir.name}")

    if not episode_infos:
        raise RuntimeError(f"No valid episodes were converted from {rmb_root}")

    tasks_records = [
        {"task_index": 0, "task": args.task_text},
    ]
    write_jsonl(out_root / "meta" / "tasks.jsonl", tasks_records)

    episodes_records = []
    episodes_stats_records = []
    initial_scene_records = []

    state_mins = []
    state_maxs = []
    action_mins = []
    action_maxs = []
    obs_eef_mins = []
    obs_eef_maxs = []
    cmd_eef_mins = []
    cmd_eef_maxs = []

    for info in episode_infos:
        episodes_records.append(
            {
                "episode_index": info["episode_index"],
                "task_index": 0,
                "length": info["length"],
                "data_file": f"data/{chunk_name}/{info['episode_stem']}.parquet",
            }
        )
        stats_obj = {
            "observation.state": {
                "min": info["state_min"],
                "max": info["state_max"],
            },
            "action": {
                "min": info["action_min"],
                "max": info["action_max"],
            },
        }
        if info["obs_eef_min"] is not None and info["obs_eef_max"] is not None:
            stats_obj["observation.eef_pose"] = {
                "min": info["obs_eef_min"],
                "max": info["obs_eef_max"],
            }
        if info["cmd_eef_min"] is not None and info["cmd_eef_max"] is not None:
            stats_obj["command.eef_pose"] = {
                "min": info["cmd_eef_min"],
                "max": info["cmd_eef_max"],
            }
        episodes_stats_records.append(
            {
                "episode_index": info["episode_index"],
                "stats": stats_obj,
            }
        )
        if info["initial_scene"] is not None:
            record = {
                "episode_index": info["episode_index"],
                "hand_rgb_source": (
                    f"videos/{chunk_name}/observation.images.hand_rgb_image/"
                    f"{info['episode_stem']}.mp4"
                ),
                "head_rgb_source": (
                    f"videos/{chunk_name}/observation.images.head_rgb_image/"
                    f"{info['episode_stem']}.mp4"
                ),
                **info["initial_scene"],
            }
            if info["initial_scene_slots"] is not None:
                record["scene_slots_target"] = info["initial_scene_slots"]
                record["scene_slots_layout"] = (
                    "dx,dy,size_x,size_y,height,confidence,valid"
                )
                record["scene_slots_target_2d"] = (
                    np.asarray(info["initial_scene_slots"], dtype=np.float32)[:, [0, 1, 6]].tolist()
                )
                record["scene_slots_layout_2d"] = "dx,dy,valid"
            if info["initial_mobile_pose"] is not None:
                record["mobile_pose_world"] = info["initial_mobile_pose"]
            initial_scene_records.append(record)
        state_mins.append(np.asarray(info["state_min"], dtype=np.float32))
        state_maxs.append(np.asarray(info["state_max"], dtype=np.float32))
        action_mins.append(np.asarray(info["action_min"], dtype=np.float32))
        action_maxs.append(np.asarray(info["action_max"], dtype=np.float32))
        if info["obs_eef_min"] is not None and info["obs_eef_max"] is not None:
            obs_eef_mins.append(np.asarray(info["obs_eef_min"], dtype=np.float32))
            obs_eef_maxs.append(np.asarray(info["obs_eef_max"], dtype=np.float32))
        if info["cmd_eef_min"] is not None and info["cmd_eef_max"] is not None:
            cmd_eef_mins.append(np.asarray(info["cmd_eef_min"], dtype=np.float32))
            cmd_eef_maxs.append(np.asarray(info["cmd_eef_max"], dtype=np.float32))

    write_jsonl(out_root / "meta" / "episodes.jsonl", episodes_records)
    write_jsonl(out_root / "meta" / "episodes_stats.jsonl", episodes_stats_records)
    if initial_scene_records:
        write_jsonl(out_root / "meta" / "initial_scene_frames.jsonl", initial_scene_records)

    global_stats = {
        "observation.state": {
            "min": np.min(np.stack(state_mins, axis=0), axis=0).tolist(),
            "max": np.max(np.stack(state_maxs, axis=0), axis=0).tolist(),
        },
        "action": {
            "min": np.min(np.stack(action_mins, axis=0), axis=0).tolist(),
            "max": np.max(np.stack(action_maxs, axis=0), axis=0).tolist(),
        },
    }
    if obs_eef_mins and obs_eef_maxs:
        global_stats["observation.eef_pose"] = {
            "min": np.min(np.stack(obs_eef_mins, axis=0), axis=0).tolist(),
            "max": np.max(np.stack(obs_eef_maxs, axis=0), axis=0).tolist(),
        }
    if cmd_eef_mins and cmd_eef_maxs:
        global_stats["command.eef_pose"] = {
            "min": np.min(np.stack(cmd_eef_mins, axis=0), axis=0).tolist(),
            "max": np.max(np.stack(cmd_eef_maxs, axis=0), axis=0).tolist(),
        }
    with (out_root / "meta" / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(global_stats, f, indent=2, ensure_ascii=False)

    total_frames = int(sum(info["length"] for info in episode_infos))
    print("")
    print("Conversion completed.")
    print(f"Input episodes : {len(episode_infos)}")
    print(f"Skipped empty  : {len(skipped_empty)}")
    if skipped_empty:
        print(f"Skipped names  : {', '.join(skipped_empty)}")
    print(f"Total frames   : {total_frames}")
    print(f"State dim      : {len(global_stats['observation.state']['min'])}")
    print(f"State layout   : {args.state_layout}")
    print(f"Action dim     : {len(global_stats['action']['min'])}")
    print(f"Gripper binary : {args.binarize_gripper_action}")
    if args.binarize_gripper_action:
        print(
            "Gripper bin cfg: "
            f"threshold={args.gripper_binary_threshold}, "
            f"close={args.gripper_close_value}, open={args.gripper_open_value}"
        )
    print(f"Video fps norm : {args.normalize_video_fps}")
    print(f"Target fps     : {args.target_fps if args.target_fps is not None else 'timestamp-inferred'}")
    if episode_infos:
        fps_values = np.asarray([info["video_fps"] for info in episode_infos], dtype=np.float64)
        print(f"Video fps range: [{fps_values.min():.6g}, {fps_values.max():.6g}]")
    print(f"Hand depth view: {args.include_hand_depth}")
    if args.include_hand_depth:
        print(f"Depth source   : {args.hand_depth_source}")
        print(f"Depth HDF5 key : {args.hand_depth_hdf5_key}")
        print(f"Depth encoding : {args.hand_depth_encoding}")
        print(f"Depth range    : [{args.hand_depth_min}, {args.hand_depth_max}] m")
        print(f"Depth visual   : {'JET colormap' if args.hand_depth_colormap else 'grayscale RGB'}")
    print(f"Initial scene depth: {args.export_initial_hand_invdepth}")
    if args.export_initial_hand_invdepth:
        print(f"Initial depth frame: {args.initial_hand_depth_frame_index}")
        print("Initial scene manifest: meta/initial_scene_frames.jsonl")
    print(f"Output path    : {out_root}")


if __name__ == "__main__":
    main()
