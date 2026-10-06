#!/usr/bin/env python3
"""Train the reset-time RGBD -> compact 2D scene-slot perceptor.

This is intentionally separate from VLA imitation training. It gives direct
meters and valid-slot metrics before predicted slots are fed to the policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.initial_scene_slots import InitialSceneSlotConfig, InitialSceneSlotPerceptor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HSR initial-scene 2D slot perceptor.")
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--env_id", type=str, default=None)
    parser.add_argument("--env_train_seed_start", type=int, default=0)
    parser.add_argument("--env_train_seed_count", type=int, default=0)
    parser.add_argument("--env_val_seed_start", type=int, default=10000)
    parser.add_argument("--env_val_seed_count", type=int, default=0)
    parser.add_argument("--initial_settle_steps", type=int, default=30)
    parser.add_argument("--hand_depth_min", type=float, default=0.05)
    parser.add_argument("--hand_depth_max", type=float, default=2.0)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument(
        "--architecture_version",
        type=str,
        default="spatial_softargmax_v2",
        choices=["query_attention_v1", "spatial_softargmax_v2", "spatial_softargmax_headrgb_v3"],
    )
    parser.add_argument("--max_slot_residual_m", type=float, default=0.35)
    parser.add_argument("--heatmap_temperature", type=float, default=0.35)
    parser.add_argument(
        "--slot_xy_weights",
        type=str,
        default="3.0,1.0,0.0,0.0",
        help="Comma-separated xy loss weights for slot0..slot3. Use a larger slot0 weight for tomato can localization.",
    )
    parser.add_argument("--no_augment", action="store_true", help="Disable photometric/depth training augmentation.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--valid_loss_weight", type=float, default=0.25)
    parser.add_argument("--heatmap_loss_weight", type=float, default=0.0)
    parser.add_argument("--heatmap_sigma", type=float, default=0.035)
    parser.add_argument("--val_modulo", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_interval", type=int, default=10, help="Write a train metric every N optimizer steps.")
    parser.add_argument("--log_dir", type=Path, default=None, help="Directory for training.log, metrics.csv and metrics.jsonl.")
    parser.add_argument("--resume_path", type=Path, default=None, help="Optional last.pt or best.pt checkpoint to continue from.")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing initial-scene manifest: {path}")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"Manifest has no records: {path}")
    return records


class InitialSceneSlotDataset(Dataset):
    def __init__(self, root: Path, records: List[Dict], image_size: int, augment: bool = False):
        self.root = root
        self.records = records
        self.image_size = int(image_size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _read_first_rgb(video_path: Path) -> np.ndarray:
        capture = cv2.VideoCapture(str(video_path))
        try:
            ok, frame_bgr = capture.read()
        finally:
            capture.release()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Unable to decode first RGB frame from {video_path}")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        if "hand_rgb_array" in record:
            rgb = np.asarray(record["hand_rgb_array"], dtype=np.uint8)
            head_rgb = np.asarray(record["head_rgb_array"], dtype=np.uint8)
            depth = np.asarray(record["hand_invdepth_array"], dtype=np.float32)
        else:
            rgb = self._read_first_rgb(self.root / record["hand_rgb_source"])
            head_rgb_source = record.get("head_rgb_source", record["hand_rgb_source"])
            head_rgb = self._read_first_rgb(self.root / head_rgb_source)
            depth_bgr = cv2.imread(str(self.root / record["hand_invdepth_file"]), cv2.IMREAD_COLOR)
            if depth_bgr is None:
                raise FileNotFoundError(f"Unable to read {record['hand_invdepth_file']}")
            depth_rgb = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)
            depth = depth_rgb.mean(axis=2, dtype=np.float32) / 255.0
        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        head_rgb = cv2.resize(head_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        rgb = rgb.astype(np.float32) / 255.0
        head_rgb = head_rgb.astype(np.float32) / 255.0
        if self.augment:
            brightness = np.random.uniform(-0.08, 0.08)
            contrast = np.random.uniform(0.85, 1.15)
            rgb = np.clip((rgb - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
            head_rgb = np.clip((head_rgb - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
            rgb = np.clip(rgb + np.random.normal(0.0, 0.01, rgb.shape).astype(np.float32), 0.0, 1.0)
            head_rgb = np.clip(
                head_rgb + np.random.normal(0.0, 0.01, head_rgb.shape).astype(np.float32), 0.0, 1.0
            )
            depth = np.clip(depth + np.random.normal(0.0, 0.008, depth.shape).astype(np.float32), 0.0, 1.0)

        target = np.asarray(
            record.get("scene_slots_target_2d", record.get("scene_slots_target")), dtype=np.float32
        )
        if target.shape == (4, 7):
            target = target[:, [0, 1, 6]]
        if target.shape != (4, 3):
            raise ValueError(f"Expected 4x3 2D slots for episode {record.get('episode_index')}, got {target.shape}")
        pixel_target = np.asarray(record.get("scene_slots_head_pixel_xy", np.zeros((4, 3))), dtype=np.float32)
        if pixel_target.shape != (4, 3):
            raise ValueError(
                f"Expected 4x3 head pixel targets for episode {record.get('episode_index')}, got {pixel_target.shape}"
            )
        return {
            "hand_rgb": torch.from_numpy(rgb.copy()).permute(2, 0, 1).float(),
            "hand_invdepth": torch.from_numpy(depth.copy()).unsqueeze(0).float(),
            "head_rgb": torch.from_numpy(head_rgb.copy()).permute(2, 0, 1).float(),
            "target_xy": torch.from_numpy(target[:, :2].copy()),
            "target_valid": torch.from_numpy(target[:, 2].copy()),
            "target_pixel_xy": torch.from_numpy(pixel_target[:, :2].copy()),
            "target_pixel_valid": torch.from_numpy(pixel_target[:, 2].copy()),
            "episode_index": torch.tensor(int(record["episode_index"]), dtype=torch.long),
        }


def parse_slot_weights(text: str) -> torch.Tensor:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(values) != 4:
        raise ValueError(f"--slot_xy_weights must contain 4 comma-separated values, got {text!r}")
    if any(value < 0.0 for value in values):
        raise ValueError("--slot_xy_weights values must be non-negative")
    return torch.tensor(values, dtype=torch.float32)


def extract_2d_target(record: Dict) -> np.ndarray:
    target = np.asarray(record.get("scene_slots_target_2d", record.get("scene_slots_target")), dtype=np.float32)
    if target.shape == (4, 7):
        target = target[:, [0, 1, 6]]
    if target.shape != (4, 3):
        raise ValueError(f"Expected 4x3 2D slots for episode {record.get('episode_index')}, got {target.shape}")
    return target


def compute_slot_anchors(records: List[Dict]) -> tuple[tuple[float, float], ...]:
    targets = np.stack([extract_2d_target(record) for record in records], axis=0)
    anchors: list[tuple[float, float]] = []
    for slot_idx in range(targets.shape[1]):
        valid = targets[:, slot_idx, 2] > 0.5
        if np.any(valid):
            xy = targets[valid, slot_idx, :2].mean(axis=0)
            anchors.append((float(xy[0]), float(xy[1])))
        else:
            anchors.append((0.0, 0.0))
    return tuple(anchors)


def inverse_depth_from_meters(depth_m: np.ndarray, depth_min: float, depth_max: float) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    clipped = np.clip(depth, depth_min, depth_max)
    inv = np.zeros_like(clipped, dtype=np.float32)
    inv[valid] = 1.0 / clipped[valid]
    inv_near = 1.0 / float(depth_min)
    inv_far = 1.0 / float(depth_max)
    inverse_depth = np.clip((inv - inv_far) / (inv_near - inv_far), 0.0, 1.0)
    inverse_depth[~valid] = 0.0
    return inverse_depth


def collect_env_initial_scene_records(args: argparse.Namespace, seeds: Iterable[int]) -> List[Dict]:
    import gymnasium as gym
    import mujoco
    import robo_manip_baselines  # noqa: F401

    def project_body_to_head_pixel(env, body_name: str) -> list[float]:
        model = env.unwrapped.model
        data = env.unwrapped.data
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head")
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if cam_id < 0 or body_id < 0:
            return [0.0, 0.0, 0.0]
        point = data.xpos[body_id]
        cam_pos = data.cam_xpos[cam_id]
        cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
        camera_point = cam_mat.T @ (point - cam_pos)
        if camera_point[2] >= -1e-6:
            return [0.0, 0.0, 0.0]
        height, width = 480.0, 640.0
        focal = 0.5 * height / math.tan(math.radians(float(model.cam_fovy[cam_id])) / 2.0)
        u = width / 2.0 + focal * (camera_point[0] / (-camera_point[2]))
        v = height / 2.0 - focal * (camera_point[1] / (-camera_point[2]))
        valid = 0.0 <= u < width and 0.0 <= v < height
        return [float(np.clip(u / width, 0.0, 1.0)), float(np.clip(v / height, 0.0, 1.0)), float(valid)]

    records: List[Dict] = []
    env = gym.make(args.env_id)
    try:
        for seed in seeds:
            if hasattr(env.unwrapped, "modify_world"):
                env.unwrapped.modify_world(cumulative_idx=int(seed))
            obs, info = env.reset(seed=int(seed))
            settle_action = np.zeros(env.action_space.shape, dtype=np.float32)
            settle_action[3:9] = np.asarray(obs["joint_pos"], dtype=np.float32)
            for _ in range(max(0, int(args.initial_settle_steps))):
                obs, _, terminated, truncated, info = env.step(settle_action.copy())
                if terminated or truncated:
                    break
            info = env.unwrapped._get_info()
            rgb_images = info.get("rgb_images", {})
            depth_images = info.get("depth_images", {})
            if "hand" not in rgb_images or "head" not in rgb_images or "hand" not in depth_images:
                raise KeyError("Env initial-scene records require head RGB, hand RGB, and hand depth")
            slots = env.unwrapped.get_scene_slots()
            slots_2d = np.asarray(slots, dtype=np.float32)[:, [0, 1, 6]]
            pixel_slots = np.zeros((4, 3), dtype=np.float32)
            for slot_idx, spec in enumerate(env.unwrapped.get_scene_slot_specs()[:4]):
                pixel_slots[slot_idx] = np.asarray(
                    project_body_to_head_pixel(env, spec["body_name"]),
                    dtype=np.float32,
                )
            records.append(
                {
                    "episode_index": int(seed),
                    "hand_rgb_array": np.asarray(rgb_images["hand"], dtype=np.uint8).copy(),
                    "head_rgb_array": np.asarray(rgb_images["head"], dtype=np.uint8).copy(),
                    "hand_invdepth_array": inverse_depth_from_meters(
                        depth_images["hand"],
                        args.hand_depth_min,
                        args.hand_depth_max,
                    ).copy(),
                    "scene_slots_target_2d": slots_2d.tolist(),
                    "scene_slots_head_pixel_xy": pixel_slots.tolist(),
                }
            )
    finally:
        try:
            env.close()
        except Exception as exc:
            logging.getLogger("initial_scene_slot_perceptor").warning("env.close warning: %s", exc)
    if not records:
        raise ValueError("No env initial-scene records were collected")
    return records


def evaluate(
    model,
    loader,
    device: torch.device,
    valid_loss_weight: float,
    slot_xy_weights: torch.Tensor | None,
    heatmap_loss_weight: float,
    heatmap_sigma: float,
) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            prediction = model(
                batch["hand_rgb"].to(device),
                batch["hand_invdepth"].to(device),
                batch["head_rgb"].to(device),
            )
            metrics = model.loss(
                prediction,
                batch["target_xy"].to(device),
                batch["target_valid"].to(device),
                valid_loss_weight=valid_loss_weight,
                slot_xy_weights=slot_xy_weights.to(device) if slot_xy_weights is not None else None,
                target_pixel_xy=batch["target_pixel_xy"].to(device),
                target_pixel_valid=batch["target_pixel_valid"].to(device),
                heatmap_loss_weight=heatmap_loss_weight,
                heatmap_sigma=heatmap_sigma,
            )
            batch_size = int(batch["hand_rgb"].shape[0])
            count += batch_size
            for key in metrics:
                sums.setdefault(key, 0.0)
                sums[key] += float(metrics[key]) * batch_size
    if count == 0:
        raise ValueError("Validation loader is empty")
    return {key: value / count for key, value in sums.items()}


def setup_logger(log_dir: Path) -> logging.Logger:
    logger = logging.getLogger("initial_scene_slot_perceptor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_dir / "training.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def append_metric(log_dir: Path, record: Dict) -> None:
    with (log_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    csv_path = log_dir / "metrics.csv"
    write_header = not csv_path.exists()
    fieldnames = [
        "phase", "epoch", "global_step", "learning_rate", "loss",
        "xy_loss", "valid_loss", "heatmap_loss", "xy_mae_m", "valid_accuracy",
        "slot0_xy_mae_m", "slot1_xy_mae_m", "slot2_xy_mae_m", "slot3_xy_mae_m",
    ]
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: record.get(key) for key in fieldnames})


def main() -> None:
    args = parse_args()
    if args.val_modulo < 2:
        raise ValueError("val_modulo must be at least 2")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested but CUDA is unavailable")

    if args.env_id is not None:
        if args.env_train_seed_count <= 0 or args.env_val_seed_count <= 0:
            raise ValueError("--env_train_seed_count and --env_val_seed_count must be positive when --env_id is used")
        train_seeds = range(args.env_train_seed_start, args.env_train_seed_start + args.env_train_seed_count)
        val_seeds = range(args.env_val_seed_start, args.env_val_seed_start + args.env_val_seed_count)
        train_records = collect_env_initial_scene_records(args, train_seeds)
        val_records = collect_env_initial_scene_records(args, val_seeds)
    else:
        if args.dataset_root is None:
            raise ValueError("Either --dataset_root or --env_id must be provided")
        records = load_jsonl(args.dataset_root / "meta" / "initial_scene_frames.jsonl")
        train_records = [r for r in records if int(r["episode_index"]) % args.val_modulo != 0]
        val_records = [r for r in records if int(r["episode_index"]) % args.val_modulo == 0]
    if not train_records or not val_records:
        raise ValueError("Deterministic train/validation split produced an empty subset")
    dataset_root = args.dataset_root or Path(".")
    train_dataset = InitialSceneSlotDataset(dataset_root, train_records, args.image_size, augment=not args.no_augment)
    val_dataset = InitialSceneSlotDataset(dataset_root, val_records, args.image_size, augment=False)
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    slot_xy_weights = parse_slot_weights(args.slot_xy_weights)
    slot_xy_anchors = compute_slot_anchors(train_records)
    config = InitialSceneSlotConfig(
        image_size=args.image_size,
        hidden_dim=args.hidden_dim,
        architecture_version=args.architecture_version,
        max_slot_residual_m=args.max_slot_residual_m,
        heatmap_temperature=args.heatmap_temperature,
        slot_xy_anchors=slot_xy_anchors,
    )
    model = InitialSceneSlotPerceptor(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir or (args.save_dir / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir)
    config_snapshot = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    with (log_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config_snapshot, handle, indent=2)
    best_val_mae = float("inf")
    start_epoch = 1
    global_step = 0

    if args.resume_path is not None:
        if not args.resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_path}")
        payload = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        if payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload.get("global_step", (start_epoch - 1) * len(train_loader)))
        best_val_mae = float(payload.get("validation", {}).get("xy_mae_m", best_val_mae))
        logger.info(
            "Resumed checkpoint=%s epoch=%d next_epoch=%d global_step=%d lr=%g",
            args.resume_path, int(payload["epoch"]), start_epoch, global_step, args.lr,
        )

    logger.info(
        "dataset=%s env_id=%s train_episodes=%d val_episodes=%d batches_per_epoch=%d device=%s",
        args.dataset_root, args.env_id, len(train_records), len(val_records), len(train_loader), device,
    )
    logger.info("architecture=%s slot_xy_anchors=%s slot_xy_weights=%s augment=%s",
                args.architecture_version, slot_xy_anchors, slot_xy_weights.tolist(), not args.no_augment)
    logger.info("heatmap_loss_weight=%g heatmap_sigma=%g", args.heatmap_loss_weight, args.heatmap_sigma)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            prediction = model(
                batch["hand_rgb"].to(device),
                batch["hand_invdepth"].to(device),
                batch["head_rgb"].to(device),
            )
            metrics = model.loss(
                prediction,
                batch["target_xy"].to(device),
                batch["target_valid"].to(device),
                valid_loss_weight=args.valid_loss_weight,
                slot_xy_weights=slot_xy_weights.to(device),
                target_pixel_xy=batch["target_pixel_xy"].to(device),
                target_pixel_valid=batch["target_pixel_valid"].to(device),
                heatmap_loss_weight=args.heatmap_loss_weight,
                heatmap_sigma=args.heatmap_sigma,
            )
            optimizer.zero_grad(set_to_none=True)
            metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            batch_size = int(batch["hand_rgb"].shape[0])
            running_loss += float(metrics["loss"].detach()) * batch_size
            seen += batch_size
            if global_step % args.log_interval == 0:
                record = {
                    "phase": "train",
                    "epoch": epoch,
                    "global_step": global_step,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    **{key: float(value.detach()) if torch.is_tensor(value) else float(value) for key, value in metrics.items()},
                }
                append_metric(log_dir, record)
                logger.info(
                    "step=%d epoch=%d train_loss=%.6f xy_mae_m=%.4f slot0_mae=%.4f slot1_mae=%.4f valid_acc=%.3f lr=%g",
                    global_step, epoch, record["loss"], record["xy_mae_m"],
                    record["slot0_xy_mae_m"], record["slot1_xy_mae_m"],
                    record["valid_accuracy"], record["learning_rate"],
                )

        val = evaluate(
            model,
            val_loader,
            device,
            args.valid_loss_weight,
            slot_xy_weights,
            args.heatmap_loss_weight,
            args.heatmap_sigma,
        )
        train_loss = running_loss / max(seen, 1)
        val_record = {
            "phase": "validation",
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **val,
        }
        append_metric(log_dir, val_record)
        logger.info(
            "epoch=%03d global_step=%d train_loss=%.6f val_loss=%.6f val_xy_mae_m=%.4f val_slot0_mae=%.4f val_slot1_mae=%.4f val_valid_acc=%.3f",
            epoch, global_step, train_loss, val["loss"], val["xy_mae_m"],
            val["slot0_xy_mae_m"], val["slot1_xy_mae_m"], val["valid_accuracy"],
        )
        payload = {
            "epoch": epoch,
            "global_step": global_step,
            "model_config": model.export_config(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation": val,
            "dataset_root": str(args.dataset_root),
            "env_id": args.env_id,
            "train_episodes": [int(r["episode_index"]) for r in train_records],
            "val_episodes": [int(r["episode_index"]) for r in val_records],
            "slot_xy_weights": slot_xy_weights.tolist(),
        }
        torch.save(payload, args.save_dir / "last.pt")
        if val["xy_mae_m"] < best_val_mae:
            best_val_mae = val["xy_mae_m"]
            torch.save(payload, args.save_dir / "best.pt")

    logger.info("Best validation xy MAE: %.4f m", best_val_mae)


if __name__ == "__main__":
    main()
