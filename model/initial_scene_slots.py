"""Initial-scene perception for compact 2D HSR scene slots.

The action policy consumes slots as [dx, dy, valid]. This module predicts the
reset-time dx/dy values from the hand RGB image and its inverse-depth map.
During rollout, odometry updates dx/dy after the one-time prediction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class InitialSceneSlotConfig:
    slot_count: int = 4
    image_size: int = 224
    hidden_dim: int = 192
    num_heads: int = 4
    coordinate_scale_m: float = 1.5
    architecture_version: str = "query_attention_v1"
    max_slot_residual_m: float = 0.35
    heatmap_temperature: float = 0.35
    slot_xy_anchors: tuple[tuple[float, float], ...] | None = None


class InitialSceneSlotPerceptor(nn.Module):
    """Predict canonical-order reset-time 2D slots from RGB plus inverse depth."""

    def __init__(self, config: InitialSceneSlotConfig | None = None):
        super().__init__()
        self.config = config or InitialSceneSlotConfig()
        hidden_dim = self.config.hidden_dim

        self._init_anchor_buffer()
        if self.config.architecture_version == "query_attention_v1":
            self._init_query_attention_v1(hidden_dim)
        elif self.config.architecture_version == "spatial_softargmax_v2":
            self._init_spatial_softargmax_v2(hidden_dim)
        elif self.config.architecture_version == "spatial_softargmax_headrgb_v3":
            self._init_spatial_softargmax_headrgb_v3(hidden_dim)
        else:
            raise ValueError(f"Unsupported initial-scene slot architecture: {self.config.architecture_version}")

    def _init_anchor_buffer(self) -> None:
        anchors = self.config.slot_xy_anchors
        if anchors is None:
            anchors = tuple((0.0, 0.0) for _ in range(self.config.slot_count))
        if len(anchors) != self.config.slot_count:
            raise ValueError(f"slot_xy_anchors must contain {self.config.slot_count} entries")
        self.register_buffer("slot_xy_anchors", torch.tensor(anchors, dtype=torch.float32), persistent=False)

    def _init_query_attention_v1(self, hidden_dim: int) -> None:
        self.backbone = nn.Sequential(
            nn.Conv2d(6, 48, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(6, 48),
            nn.GELU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Conv2d(96, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.slot_queries = nn.Parameter(torch.empty(self.config.slot_count, hidden_dim))
        nn.init.normal_(self.slot_queries, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=self.config.num_heads,
            batch_first=True,
        )
        self.slot_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def _init_spatial_softargmax_v2(self, hidden_dim: int) -> None:
        self.backbone = nn.Sequential(
            nn.Conv2d(6, 48, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(6, 48),
            nn.GELU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Conv2d(96, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.heatmap_head = nn.Conv2d(hidden_dim, self.config.slot_count, kernel_size=1)
        self.slot_id_embedding = nn.Parameter(torch.empty(self.config.slot_count, hidden_dim))
        nn.init.normal_(self.slot_id_embedding, std=0.02)
        self.slot_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def _init_spatial_softargmax_headrgb_v3(self, hidden_dim: int) -> None:
        self.hand_backbone = self._make_conv_stem(input_channels=6, hidden_dim=hidden_dim)
        self.head_backbone = self._make_conv_stem(input_channels=5, hidden_dim=hidden_dim)
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.heatmap_head = nn.Conv2d(hidden_dim, self.config.slot_count, kernel_size=1)
        self.slot_id_embedding = nn.Parameter(torch.empty(self.config.slot_count, hidden_dim))
        nn.init.normal_(self.slot_id_embedding, std=0.02)
        self.slot_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    @staticmethod
    def _make_conv_stem(input_channels: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(input_channels, 48, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(6, 48),
            nn.GELU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Conv2d(96, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        hand_rgb: torch.Tensor,
        hand_invdepth: torch.Tensor,
        head_rgb: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if hand_rgb.ndim != 4 or hand_rgb.shape[1] != 3:
            raise ValueError(f"hand_rgb must have shape [B,3,H,W], got {tuple(hand_rgb.shape)}")
        if hand_invdepth.ndim != 4 or hand_invdepth.shape[1] != 1:
            raise ValueError(
                f"hand_invdepth must have shape [B,1,H,W], got {tuple(hand_invdepth.shape)}"
            )
        if hand_rgb.shape[0] != hand_invdepth.shape[0]:
            raise ValueError("hand_rgb and hand_invdepth batch sizes must match")

        target_size = (self.config.image_size, self.config.image_size)
        hand_rgb = F.interpolate(hand_rgb.float(), size=target_size, mode="bilinear", align_corners=False)
        hand_invdepth = F.interpolate(hand_invdepth.float(), size=target_size, mode="bilinear", align_corners=False)
        batch_size, _, height, width = hand_rgb.shape
        coords = self._coordinate_planes(batch_size, height, width, hand_rgb.device, hand_rgb.dtype)
        if self.config.architecture_version == "spatial_softargmax_headrgb_v3":
            if head_rgb is None:
                raise ValueError("head_rgb is required for spatial_softargmax_headrgb_v3")
            if head_rgb.ndim != 4 or head_rgb.shape[1] != 3:
                raise ValueError(f"head_rgb must have shape [B,3,H,W], got {tuple(head_rgb.shape)}")
            if head_rgb.shape[0] != batch_size:
                raise ValueError("head_rgb and hand_rgb batch sizes must match")
            head_rgb = F.interpolate(head_rgb.float(), size=target_size, mode="bilinear", align_corners=False)
            head_coords = self._coordinate_planes(batch_size, height, width, head_rgb.device, head_rgb.dtype)
            hand_features = self.hand_backbone(torch.cat([hand_rgb, hand_invdepth, coords], dim=1))
            head_features = self.head_backbone(torch.cat([head_rgb, head_coords], dim=1))
            features = self.fusion(torch.cat([hand_features, head_features], dim=1))
            return self._forward_spatial_softargmax_v2(features)
        features = self.backbone(torch.cat([hand_rgb, hand_invdepth, coords], dim=1))
        if self.config.architecture_version == "spatial_softargmax_v2":
            return self._forward_spatial_softargmax_v2(features)
        return self._forward_query_attention_v1(features)

    @staticmethod
    def _coordinate_planes(
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack([xx, yy], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)

    def _forward_query_attention_v1(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = features.shape[0]
        tokens = features.flatten(2).transpose(1, 2)
        queries = self.slot_queries.unsqueeze(0).expand(batch_size, -1, -1)
        slot_tokens, _ = self.cross_attention(queries, tokens, tokens, need_weights=False)
        raw = self.head(self.slot_norm(slot_tokens))
        xy = torch.tanh(raw[..., :2]) * float(self.config.coordinate_scale_m)
        return {"xy": xy, "valid_logits": raw[..., 2]}

    def _forward_spatial_softargmax_v2(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, channels, height, width = features.shape
        heat_logits = self.heatmap_head(features)
        temperature = max(float(self.config.heatmap_temperature), 1e-3)
        heat_probs = torch.softmax((heat_logits / temperature).flatten(2), dim=-1).view(
            batch_size, self.config.slot_count, height, width
        )
        pooled = torch.einsum("bshw,bchw->bsc", heat_probs, features)
        slot_tokens = self.slot_norm(pooled + self.slot_id_embedding.unsqueeze(0))

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=features.device, dtype=features.dtype),
            torch.linspace(-1.0, 1.0, width, device=features.device, dtype=features.dtype),
            indexing="ij",
        )
        expected_x = (heat_probs * xx[None, None]).sum(dim=(2, 3))
        expected_y = (heat_probs * yy[None, None]).sum(dim=(2, 3))
        expected_xy = torch.stack([expected_x, expected_y], dim=-1)

        anchors = self.slot_xy_anchors.to(device=features.device, dtype=features.dtype)
        anchors = anchors.unsqueeze(0).expand(batch_size, -1, -1)
        raw = self.head(torch.cat([slot_tokens, expected_xy, anchors], dim=-1))
        residual = torch.tanh(raw[..., :2]) * float(self.config.max_slot_residual_m)
        xy = anchors + residual
        return {"xy": xy, "valid_logits": raw[..., 2], "heat_probs": heat_probs}

    def loss(
        self,
        prediction: Dict[str, torch.Tensor],
        target_xy: torch.Tensor,
        target_valid: torch.Tensor,
        valid_loss_weight: float = 0.25,
        slot_xy_weights: torch.Tensor | None = None,
        target_pixel_xy: torch.Tensor | None = None,
        target_pixel_valid: torch.Tensor | None = None,
        heatmap_loss_weight: float = 0.0,
        heatmap_sigma: float = 0.035,
    ) -> Dict[str, torch.Tensor]:
        if target_xy.shape != prediction["xy"].shape:
            raise ValueError(
                f"target_xy shape {tuple(target_xy.shape)} does not match predicted {tuple(prediction['xy'].shape)}"
            )
        if target_valid.shape != prediction["valid_logits"].shape:
            raise ValueError("target_valid shape does not match valid_logits")

        valid_mask = target_valid > 0.5
        xy_error = F.smooth_l1_loss(prediction["xy"], target_xy, reduction="none").mean(dim=-1)
        xy_weights = valid_mask.float()
        if slot_xy_weights is not None:
            if slot_xy_weights.ndim != 1 or slot_xy_weights.shape[0] != target_valid.shape[-1]:
                raise ValueError("slot_xy_weights must be a 1D tensor with one value per slot")
            xy_weights = xy_weights * slot_xy_weights.to(device=target_valid.device, dtype=xy_error.dtype).view(1, -1)
        xy_loss = (xy_error * xy_weights).sum() / xy_weights.sum().clamp_min(1e-6)
        valid_loss = F.binary_cross_entropy_with_logits(prediction["valid_logits"], target_valid.float())
        heatmap_loss = xy_loss.new_tensor(0.0)
        if (
            heatmap_loss_weight > 0.0
            and target_pixel_xy is not None
            and target_pixel_valid is not None
            and "heat_probs" in prediction
        ):
            heatmap_loss = self._heatmap_alignment_loss(
                prediction["heat_probs"],
                target_pixel_xy,
                target_pixel_valid,
                sigma=float(heatmap_sigma),
            )
        total = xy_loss + float(valid_loss_weight) * valid_loss + float(heatmap_loss_weight) * heatmap_loss
        xy_mae = ((prediction["xy"] - target_xy).abs().mean(dim=-1) * valid_mask).sum()
        xy_mae = xy_mae / valid_mask.sum().clamp_min(1)
        valid_accuracy = ((prediction["valid_logits"] >= 0) == valid_mask).float().mean()
        metrics = {
            "loss": total,
            "xy_loss": xy_loss.detach(),
            "valid_loss": valid_loss.detach(),
            "heatmap_loss": heatmap_loss.detach(),
            "xy_mae_m": xy_mae.detach(),
            "valid_accuracy": valid_accuracy.detach(),
        }
        per_slot_mae = (prediction["xy"] - target_xy).abs().mean(dim=-1)
        for slot_idx in range(target_valid.shape[-1]):
            slot_valid = valid_mask[:, slot_idx]
            if slot_valid.any():
                metrics[f"slot{slot_idx}_xy_mae_m"] = per_slot_mae[:, slot_idx][slot_valid].mean().detach()
            else:
                metrics[f"slot{slot_idx}_xy_mae_m"] = per_slot_mae.new_tensor(0.0)
        return metrics

    @staticmethod
    def _heatmap_alignment_loss(
        heat_probs: torch.Tensor,
        target_pixel_xy: torch.Tensor,
        target_pixel_valid: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        if heat_probs.ndim != 4:
            raise ValueError(f"heat_probs must have shape [B,S,H,W], got {tuple(heat_probs.shape)}")
        if target_pixel_xy.shape[:2] != heat_probs.shape[:2] or target_pixel_xy.shape[-1] != 2:
            raise ValueError("target_pixel_xy must have shape [B,S,2]")
        if target_pixel_valid.shape != heat_probs.shape[:2]:
            raise ValueError("target_pixel_valid must have shape [B,S]")
        batch_size, slot_count, height, width = heat_probs.shape
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, height, device=heat_probs.device, dtype=heat_probs.dtype),
            torch.linspace(0.0, 1.0, width, device=heat_probs.device, dtype=heat_probs.dtype),
            indexing="ij",
        )
        target_x = target_pixel_xy[..., 0].to(device=heat_probs.device, dtype=heat_probs.dtype)
        target_y = target_pixel_xy[..., 1].to(device=heat_probs.device, dtype=heat_probs.dtype)
        dist2 = (xx[None, None] - target_x[:, :, None, None]) ** 2
        dist2 = dist2 + (yy[None, None] - target_y[:, :, None, None]) ** 2
        target_heat = torch.exp(-dist2 / (2.0 * max(sigma, 1e-4) ** 2))
        target_heat = target_heat / target_heat.flatten(2).sum(dim=-1).clamp_min(1e-8)[:, :, None, None]
        per_slot = -(target_heat * torch.log(heat_probs.clamp_min(1e-8))).flatten(2).sum(dim=-1)
        valid = target_pixel_valid.to(device=heat_probs.device, dtype=heat_probs.dtype)
        return (per_slot * valid).sum() / valid.sum().clamp_min(1.0)

    def export_config(self) -> Dict[str, int | float | str | tuple[tuple[float, float], ...] | None]:
        return asdict(self.config)


def slots_from_prediction(
    prediction: Dict[str, torch.Tensor], valid_threshold: float = 0.5
) -> torch.Tensor:
    """Return policy-compatible `[dx, dy, valid]` slots."""
    valid = (torch.sigmoid(prediction["valid_logits"]) >= float(valid_threshold)).to(prediction["xy"].dtype)
    return torch.cat([prediction["xy"], valid.unsqueeze(-1)], dim=-1)
