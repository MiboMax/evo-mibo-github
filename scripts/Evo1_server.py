# evo1_server_json.py

import sys
import os
import asyncio
import websockets
import numpy as np
import cv2
import json
import torch
import contextlib
import argparse
from typing import Dict, List
from PIL import Image
from torchvision import transforms


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.Evo1 import EVO1
from config import EvoConfig


class Normalizer:
    def __init__(
        self,
        stats_or_path,
        state_dim: int,
        action_dim: int,
        stats_mode: str = "auto",
        stats_dataset_key: str | None = None,
    ):
        if isinstance(stats_or_path, str):
            with open(stats_or_path, "r") as f:
                stats = json.load(f)
        else:
            stats = stats_or_path

        self.state_dim = state_dim
        self.action_dim = action_dim

        if len(stats) != 1:
            raise ValueError(f"norm_stats.json should contain only one robot key, but: {list(stats.keys())}")

        robot_key = list(stats.keys())[0]
        robot_stats = stats[robot_key]

        selected_stats, selection_desc = self._select_stats(
            robot_key=robot_key,
            robot_stats=robot_stats,
            stats_mode=stats_mode,
            stats_dataset_key=stats_dataset_key,
        )
        print(f"[Normalizer] Selected stats: {selection_desc}")

        self.state_min = self._pad_to_dim(selected_stats["observation.state"]["min"], self.state_dim)
        self.state_max = self._pad_to_dim(selected_stats["observation.state"]["max"], self.state_dim)
        self.action_min = self._pad_to_dim(selected_stats["action"]["min"], self.action_dim)
        self.action_max = self._pad_to_dim(selected_stats["action"]["max"], self.action_dim)

    @staticmethod
    def _pad_to_dim(values, target_dim: int):
        tensor = torch.tensor(values, dtype=torch.float32)
        if tensor.shape[0] < target_dim:
            pad = torch.zeros(target_dim - tensor.shape[0], dtype=torch.float32)
            tensor = torch.cat([tensor, pad], dim=0)
        elif tensor.shape[0] > target_dim:
            tensor = tensor[:target_dim]
        return tensor

    @staticmethod
    def _reduce_vectors(vectors: List[np.ndarray], mode: str, name: str) -> np.ndarray:
        if not vectors:
            raise ValueError(f"No vectors to merge for {name}")
        base_shape = vectors[0].shape
        for v in vectors[1:]:
            if v.shape != base_shape:
                raise ValueError(f"Shape mismatch while merging {name}: {base_shape} vs {v.shape}")

        stacked = np.stack(vectors, axis=0)
        if mode == "min":
            return np.min(stacked, axis=0)
        if mode == "max":
            return np.max(stacked, axis=0)
        if mode == "mean":
            return np.mean(stacked, axis=0)
        raise ValueError(f"Unsupported reduce mode: {mode}")

    def _merge_dataset_stats(self, dataset_stats_map: Dict[str, Dict]) -> Dict:
        dataset_keys = list(dataset_stats_map.keys())
        merged = {"observation.state": {}, "action": {}}

        for field in ("observation.state", "action"):
            if not all(field in dataset_stats_map[k] for k in dataset_keys):
                raise ValueError(f"Missing '{field}' in one or more dataset stats")

            stat_key_union = set()
            for k in dataset_keys:
                stat_key_union.update(dataset_stats_map[k][field].keys())

            for stat_name in sorted(stat_key_union):
                vecs = []
                for k in dataset_keys:
                    if stat_name in dataset_stats_map[k][field]:
                        vecs.append(np.asarray(dataset_stats_map[k][field][stat_name], dtype=np.float32))
                if not vecs:
                    continue

                if stat_name in ("min", "q01"):
                    reduced = self._reduce_vectors(vecs, "min", f"{field}.{stat_name}")
                elif stat_name in ("max", "q99"):
                    reduced = self._reduce_vectors(vecs, "max", f"{field}.{stat_name}")
                else:
                    reduced = self._reduce_vectors(vecs, "mean", f"{field}.{stat_name}")
                merged[field][stat_name] = reduced.tolist()

        # Bounds fallback safety.
        for field in ("observation.state", "action"):
            if "min" not in merged[field] or "max" not in merged[field]:
                raise ValueError(f"Merged stats for {field} missing min/max")

        return merged

    def _select_stats(
        self,
        robot_key: str,
        robot_stats: Dict,
        stats_mode: str,
        stats_dataset_key: str | None,
    ):
        # Flat format: {"observation.state": ..., "action": ...}
        if "observation.state" in robot_stats and "action" in robot_stats:
            return robot_stats, f"robot={robot_key}, mode=flat"

        # Nested format: {dataset_key: {"observation.state": ..., "action": ...}, ...}
        dataset_keys = list(robot_stats.keys())
        if not dataset_keys:
            raise ValueError(f"No dataset stats found under robot key: {robot_key}")

        if stats_dataset_key is not None:
            if stats_dataset_key not in robot_stats:
                raise ValueError(
                    f"stats_dataset_key '{stats_dataset_key}' not found under robot '{robot_key}'. "
                    f"Available: {dataset_keys}"
                )
            return robot_stats[stats_dataset_key], (
                f"robot={robot_key}, mode=dataset_key, dataset_key={stats_dataset_key}"
            )

        resolved_mode = stats_mode
        if resolved_mode == "auto":
            resolved_mode = "merged" if len(dataset_keys) > 1 else "first"

        if resolved_mode == "first":
            chosen = dataset_keys[0]
            return robot_stats[chosen], f"robot={robot_key}, mode=first, dataset_key={chosen}"

        if resolved_mode == "merged":
            merged = self._merge_dataset_stats(robot_stats)
            return merged, f"robot={robot_key}, mode=merged, dataset_keys={dataset_keys}"

        raise ValueError(f"Invalid stats_mode: {stats_mode}")

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        state_min = self.state_min.to(state.device, dtype=state.dtype)
        state_max = self.state_max.to(state.device, dtype=state.dtype)
        normalized = torch.clamp(2 * (state - state_min) / (state_max - state_min + 1e-8) - 1, -1.0, 1.0)
        if normalized.shape[-1] >= 43 and state.shape[-1] >= 43:
            valid_indices = [21, 28, 35, 42]  # no_wrench15 + 4 * [..., valid]
        elif normalized.shape[-1] >= 27 and state.shape[-1] >= 27:
            valid_indices = [17, 20, 23, 26]  # no_wrench15 + 4 * [dx, dy, valid]
        else:
            return normalized
        normalized = normalized.clone()
        valid_indices = torch.tensor(valid_indices, device=normalized.device)
        normalized[..., valid_indices] = state[..., valid_indices].to(normalized.dtype).clamp(0.0, 1.0)
        return normalized

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        action_min = self.action_min.to(action.device, dtype=action.dtype)
        action_max = self.action_max.to(action.device, dtype=action.dtype)
        if action.ndim == 1:
            action = action.view(1, -1)
        return (action + 1.0) / 2.0 * (action_max - action_min + 1e-8) + action_min


def load_model_and_normalizer(
    ckpt_dir,
    stats_mode: str = "auto",
    stats_dataset_key: str | None = None,
    image_size_override: int | None = None,
    num_inference_timesteps: int | None = None,
):
    config = json.load(open(os.path.join(ckpt_dir, "config.json")))
    stats = json.load(open(os.path.join(ckpt_dir, "norm_stats.json")))

    state_dim = int(config.get("state_dim", 24))
    action_dim = int(config.get("per_action_dim", 24))
    horizon_raw = config.get("horizon", None)
    if horizon_raw is None:
        horizon_raw = config.get("action_horizon", 16)
    horizon = int(horizon_raw if horizon_raw is not None else 16)
    image_size = int(config.get("image_size", 448))
    if image_size_override is not None:
        image_size = int(image_size_override)
    # Some checkpoints store action_horizon=None; EVO1 prefers action_horizon first.
    config["horizon"] = horizon
    config["action_horizon"] = horizon

    config["finetune_vlm"] = False
    config["finetune_action_head"] = False
    if num_inference_timesteps is not None:
        config["num_inference_timesteps"] = int(num_inference_timesteps)
    else:
        # Preserve historical server behavior unless explicitly overridden.
        config["num_inference_timesteps"] = 32

    model = EVO1(EvoConfig.from_dict(config)).eval()
    ds_ckpt_path = os.path.join(ckpt_dir, "mp_rank_00_model_states.pt")
    pt_ckpt_path = os.path.join(ckpt_dir, "pytorch_model.pt")
    std_ckpt_path = os.path.join(ckpt_dir, "checkpoint.pt")

    if os.path.exists(ds_ckpt_path):
        checkpoint = torch.load(ds_ckpt_path, map_location="cpu")
        state_dict = checkpoint["module"]
    elif os.path.exists(pt_ckpt_path):
        checkpoint = torch.load(pt_ckpt_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "module" in checkpoint:
            state_dict = checkpoint["module"]
        else:
            state_dict = checkpoint
    elif os.path.exists(std_ckpt_path):
        checkpoint = torch.load(std_ckpt_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "module" in checkpoint:
            state_dict = checkpoint["module"]
        else:
            state_dict = checkpoint
    else:
        raise FileNotFoundError(
            f"No supported checkpoint found under {ckpt_dir}. "
            f"Expected one of: mp_rank_00_model_states.pt, pytorch_model.pt, checkpoint.pt"
        )

    # The inference model does not include optional training-only heads
    # (e.g., eef_aux_head), so drop these keys when present.
    filtered_state_dict = {
        k: v for k, v in state_dict.items() if not k.startswith("eef_aux_head.")
    }
    dropped = len(state_dict) - len(filtered_state_dict)
    if dropped > 0:
        print(f"[load_model] Dropped {dropped} training-only keys (eef_aux_head.*).")

    incompatible = model.load_state_dict(filtered_state_dict, strict=False)
    if incompatible.unexpected_keys:
        print(f"[load_model] Unexpected keys: {incompatible.unexpected_keys[:20]}")
    if incompatible.missing_keys:
        print(f"[load_model] Missing keys: {incompatible.missing_keys[:20]}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    normalizer = Normalizer(
        stats,
        state_dim=state_dim,
        action_dim=action_dim,
        stats_mode=stats_mode,
        stats_dataset_key=stats_dataset_key,
    )
    return model, normalizer, state_dim, action_dim, horizon, image_size, device



def decode_image_from_list(img_list, device: str, image_size: int):
    img_array = np.array(img_list, dtype=np.uint8)
    img = cv2.resize(img_array, (image_size, image_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    return transforms.ToTensor()(pil).to(device)



def infer_from_json_dict(
    data: dict,
    model,
    normalizer,
    state_dim: int,
    action_dim: int,
    horizon: int,
    image_size: int,
    device: str,
    clip_action_to_stats: bool = False,
):

    inference_seed = data.get("inference_seed")
    if inference_seed is not None:
        inference_seed = int(inference_seed)
        torch.manual_seed(inference_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(inference_seed)

    images = [decode_image_from_list(img, device=device, image_size=image_size) for img in data["image"]]
    assert len(images) == 3, "Must provide exactly 3 images."
    for img in images:
        assert img.shape == (3, image_size, image_size), f"image_size must be (3,{image_size},{image_size})"

    state = torch.tensor(data["state"], dtype=torch.float32, device=device)
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.shape[1] < state_dim:
        state = torch.cat([state, torch.zeros((1, state_dim - state.shape[1]), device=device)], dim=1)
    elif state.shape[1] > state_dim:
        state = state[:, :state_dim]
    norm_state = normalizer.normalize_state(state).to(dtype=torch.float32)

    prompt = data["prompt"]
    image_mask = torch.tensor(data["image_mask"], dtype=torch.int32, device=device)
    action_mask = torch.tensor(data["action_mask"], dtype=torch.int32, device=device)
    if action_mask.ndim == 1:
        action_mask = action_mask.unsqueeze(0)
    elif action_mask.ndim == 3 and action_mask.shape[1] == 1:
        action_mask = action_mask.squeeze(1)
    if action_mask.shape[-1] != action_dim:
        raise ValueError(
            f"action_mask last dim must be {action_dim}, got {action_mask.shape}"
        )

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.startswith("cuda")
        else contextlib.nullcontext()
    )
    with torch.no_grad(), autocast_ctx:
        action = model.run_inference(
            images=images,
            image_mask=image_mask,
            prompt=prompt,
            state_input=norm_state,
            action_mask=action_mask
        )
        action = action.reshape(1, horizon, action_dim)
        action = normalizer.denormalize_action(action[0])
        if clip_action_to_stats:
            action_min = normalizer.action_min.to(action.device, dtype=action.dtype)
            action_max = normalizer.action_max.to(action.device, dtype=action.dtype)
            action = torch.clamp(action, min=action_min, max=action_max)
        return action.cpu().numpy().tolist()


async def handle_request(
    websocket,
    model,
    normalizer,
    state_dim: int,
    action_dim: int,
    horizon: int,
    image_size: int,
    device: str,
    clip_action_to_stats: bool = False,
):
    try:
        async for message in websocket:
            json_data = json.loads(message)
            actions = infer_from_json_dict(
                json_data,
                model=model,
                normalizer=normalizer,
                state_dim=state_dim,
                action_dim=action_dim,
                horizon=horizon,
                image_size=image_size,
                device=device,
                clip_action_to_stats=clip_action_to_stats,
            )
            await websocket.send(json.dumps(actions))

    except websockets.exceptions.ConnectionClosed:
        pass


# === Start server ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evo-1 inference websocket server")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to checkpoint dir (e.g., step_best or step_final)")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument(
        "--stats_mode",
        type=str,
        default="auto",
        choices=["auto", "first", "merged"],
        help="How to choose stats when norm_stats has multiple dataset groups.",
    )
    parser.add_argument(
        "--stats_dataset_key",
        type=str,
        default=None,
        help="Explicit dataset key inside norm_stats to use for (de)normalization.",
    )
    parser.add_argument(
        "--image_size_override",
        type=int,
        default=None,
        help="Optional image size override; defaults to checkpoint config image_size.",
    )
    parser.add_argument(
        "--num_inference_timesteps",
        type=int,
        default=None,
        help="Override inference diffusion steps. Keep unset to preserve legacy server default (32).",
    )
    parser.add_argument(
        "--clip_action_to_stats",
        action="store_true",
        help="Clip denormalized actions to norm_stats min/max bounds before sending to client.",
    )
    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir
    port = args.port

    print("Loading EVO_1 model...")
    model, normalizer, state_dim, action_dim, horizon, image_size, device = load_model_and_normalizer(
        ckpt_dir,
        stats_mode=args.stats_mode,
        stats_dataset_key=args.stats_dataset_key,
        image_size_override=args.image_size_override,
        num_inference_timesteps=args.num_inference_timesteps,
    )

    async def main():
        print(f"EVO_1 server running at ws://0.0.0.0:{port}")
        print(f"Inference image_size={image_size}")
        print(f"clip_action_to_stats={args.clip_action_to_stats}")
        print(
            f"num_inference_timesteps={getattr(model.action_head.config, 'num_inference_timesteps', 'unknown')}"
        )
        async with websockets.serve(
            lambda ws: handle_request(
                ws,
                model,
                normalizer,
                state_dim,
                action_dim,
                horizon,
                image_size,
                device,
                args.clip_action_to_stats,
            ),
            "0.0.0.0", port, max_size=100_000_000
        ):
            await asyncio.Future()

    asyncio.run(main())
