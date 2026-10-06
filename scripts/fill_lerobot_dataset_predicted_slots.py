#!/usr/bin/env python3
"""Create a State27 LeRobot dataset by appending predicted 2D slots to State15.

The source dataset is left untouched. Videos and initial-scene assets are
hard-linked by default, while parquet/meta stats are regenerated.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.initial_scene_slots import InitialSceneSlotConfig, InitialSceneSlotPerceptor, slots_from_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append predicted 2D slots to a State15 HSR LeRobot dataset.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--perceptor-ckpt", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--valid-threshold", type=float, default=0.5)
    parser.add_argument("--hand-depth-min", type=float, default=0.05)
    parser.add_argument("--hand-depth-max", type=float, default=2.0)
    parser.add_argument("--copy-media", action="store_true", help="Copy media/assets instead of hard-linking.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def reset_out_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, copy_media: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_media:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def mirror_tree(source_root: Path, out_root: Path, relative_dir: str, copy_media: bool) -> None:
    src_dir = source_root / relative_dir
    if not src_dir.exists():
        return
    for src in src_dir.rglob("*"):
        if src.is_dir():
            continue
        dst = out_root / src.relative_to(source_root)
        link_or_copy(src, dst, copy_media=copy_media)


def read_first_rgb(video_path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    try:
        ok, frame_bgr = capture.read()
    finally:
        capture.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Unable to decode first frame from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def inverse_depth_from_meters(depth_m: np.ndarray, depth_min: float, depth_max: float) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    clipped = np.clip(depth, float(depth_min), float(depth_max))
    inv = np.zeros_like(clipped, dtype=np.float32)
    inv[valid] = 1.0 / clipped[valid]
    inv_near = 1.0 / float(depth_min)
    inv_far = 1.0 / float(depth_max)
    invdepth = np.clip((inv - inv_far) / (inv_near - inv_far), 0.0, 1.0)
    invdepth[~valid] = 0.0
    return invdepth


def load_perceptor(ckpt_path: Path, device: torch.device) -> InitialSceneSlotPerceptor:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Perceptor checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device)
    config = InitialSceneSlotConfig(**payload["model_config"])
    model = InitialSceneSlotPerceptor(config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model


def predict_slots_for_record(
    model: InitialSceneSlotPerceptor,
    device: torch.device,
    source_root: Path,
    record: Dict,
    valid_threshold: float,
    depth_min: float,
    depth_max: float,
) -> np.ndarray:
    hand_rgb = read_first_rgb(source_root / record["hand_rgb_source"]).astype(np.float32) / 255.0
    head_rgb = read_first_rgb(source_root / record["head_rgb_source"]).astype(np.float32) / 255.0
    depth_m = np.load(source_root / record["hand_depth_m_file"])["depth_m"].astype(np.float32)
    invdepth = inverse_depth_from_meters(depth_m, depth_min=depth_min, depth_max=depth_max)

    hand_tensor = torch.from_numpy(hand_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    head_tensor = torch.from_numpy(head_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    depth_tensor = torch.from_numpy(invdepth).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        prediction = model(hand_tensor, depth_tensor, head_tensor)
        slots = slots_from_prediction(prediction, valid_threshold=valid_threshold)[0]
        invalid = slots[:, 2] <= 0.5
        slots[invalid, :2] = 0.0
    return slots.float().cpu().numpy().astype(np.float32)


def recompute_episode_stats(df: pd.DataFrame) -> Dict:
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    action = np.stack(df["action"].to_numpy()).astype(np.float32)
    stats = {
        "observation.state": {"min": state.min(axis=0).tolist(), "max": state.max(axis=0).tolist()},
        "action": {"min": action.min(axis=0).tolist(), "max": action.max(axis=0).tolist()},
    }
    if "observation.eef_pose" in df.columns:
        obs_eef = np.stack(df["observation.eef_pose"].to_numpy()).astype(np.float32)
        stats["observation.eef_pose"] = {"min": obs_eef.min(axis=0).tolist(), "max": obs_eef.max(axis=0).tolist()}
    if "command.eef_pose" in df.columns:
        cmd_eef = np.stack(df["command.eef_pose"].to_numpy()).astype(np.float32)
        stats["command.eef_pose"] = {"min": cmd_eef.min(axis=0).tolist(), "max": cmd_eef.max(axis=0).tolist()}
    return stats


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested but CUDA is unavailable")
    reset_out_dir(out_root, overwrite=args.overwrite)

    for required in ("meta/tasks.jsonl", "meta/episodes.jsonl", "meta/initial_scene_frames.jsonl"):
        if not (source_root / required).exists():
            raise FileNotFoundError(f"Missing required source file: {source_root / required}")

    mirror_tree(source_root, out_root, "videos", copy_media=args.copy_media)
    mirror_tree(source_root, out_root, "initial_scene", copy_media=args.copy_media)
    (out_root / "meta").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / "meta" / "tasks.jsonl", out_root / "meta" / "tasks.jsonl")

    model = load_perceptor(args.perceptor_ckpt.expanduser().resolve(), device=device)
    episodes = read_jsonl(source_root / "meta" / "episodes.jsonl")
    initial_records = read_jsonl(source_root / "meta" / "initial_scene_frames.jsonl")
    initial_by_episode = {int(record["episode_index"]): record for record in initial_records}

    out_episode_stats = []
    out_initial_records = []
    state_mins = []
    state_maxs = []
    action_mins = []
    action_maxs = []
    valid_counts = np.zeros(4, dtype=np.int64)
    xy_values: list[np.ndarray] = []

    for idx, episode in enumerate(episodes):
        episode_index = int(episode["episode_index"])
        if episode_index not in initial_by_episode:
            raise KeyError(f"Missing initial-scene record for episode_index={episode_index}")
        slots = predict_slots_for_record(
            model=model,
            device=device,
            source_root=source_root,
            record=initial_by_episode[episode_index],
            valid_threshold=args.valid_threshold,
            depth_min=args.hand_depth_min,
            depth_max=args.hand_depth_max,
        )
        valid_counts += (slots[:, 2] > 0.5).astype(np.int64)
        if np.any(slots[:, 2] > 0.5):
            xy_values.append(slots[slots[:, 2] > 0.5, :2])

        src_parquet = source_root / episode["data_file"]
        dst_parquet = out_root / episode["data_file"]
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(src_parquet)
        if df.empty:
            raise ValueError(f"Unexpected empty parquet: {src_parquet}")
        first_state = np.asarray(df["observation.state"].iloc[0], dtype=np.float32)
        if first_state.shape != (15,):
            raise ValueError(f"Expected State15 source, got state shape {first_state.shape} in {src_parquet}")
        slot_flat = slots.reshape(-1)
        df["observation.state"] = [
            np.concatenate([np.asarray(state, dtype=np.float32), slot_flat], axis=0).tolist()
            for state in df["observation.state"].to_numpy()
        ]
        df.to_parquet(dst_parquet, index=False)

        stats = recompute_episode_stats(df)
        out_episode_stats.append({"episode_index": episode_index, "stats": stats})
        state_mins.append(np.asarray(stats["observation.state"]["min"], dtype=np.float32))
        state_maxs.append(np.asarray(stats["observation.state"]["max"], dtype=np.float32))
        action_mins.append(np.asarray(stats["action"]["min"], dtype=np.float32))
        action_maxs.append(np.asarray(stats["action"]["max"], dtype=np.float32))

        out_record = dict(initial_by_episode[episode_index])
        out_record["scene_slots_predicted_2d"] = slots.tolist()
        out_record["scene_slots_target_2d"] = slots.tolist()
        out_record["scene_slots_layout_2d"] = "dx,dy,valid"
        out_record["scene_slots_source"] = "initial_scene_slot_perceptor"
        out_record["scene_slots_perceptor_ckpt"] = str(args.perceptor_ckpt.expanduser().resolve())
        out_record["scene_slots_valid_threshold"] = float(args.valid_threshold)
        out_initial_records.append(out_record)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(episodes):
            print(f"[{idx + 1}/{len(episodes)}] filled {src_parquet.name}")

    shutil.copy2(source_root / "meta" / "episodes.jsonl", out_root / "meta" / "episodes.jsonl")
    write_jsonl(out_root / "meta" / "episodes_stats.jsonl", out_episode_stats)
    write_jsonl(out_root / "meta" / "initial_scene_frames.jsonl", out_initial_records)

    stats = {
        "observation.state": {
            "min": np.min(np.stack(state_mins, axis=0), axis=0).tolist(),
            "max": np.max(np.stack(state_maxs, axis=0), axis=0).tolist(),
        },
        "action": {
            "min": np.min(np.stack(action_mins, axis=0), axis=0).tolist(),
            "max": np.max(np.stack(action_maxs, axis=0), axis=0).tolist(),
        },
    }
    with (out_root / "meta" / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)

    all_xy = np.concatenate(xy_values, axis=0) if xy_values else np.zeros((0, 2), dtype=np.float32)
    summary = {
        "source_root": str(source_root),
        "out_root": str(out_root),
        "perceptor_ckpt": str(args.perceptor_ckpt.expanduser().resolve()),
        "valid_threshold": float(args.valid_threshold),
        "episodes": len(episodes),
        "state_dim": 27,
        "valid_counts_by_slot": valid_counts.tolist(),
        "valid_xy_min": all_xy.min(axis=0).tolist() if all_xy.size else None,
        "valid_xy_max": all_xy.max(axis=0).tolist() if all_xy.size else None,
        "valid_xy_mean": all_xy.mean(axis=0).tolist() if all_xy.size else None,
    }
    with (out_root / "meta" / "predicted_slots_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
