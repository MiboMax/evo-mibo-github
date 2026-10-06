import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, seq_len: int):
        if seq_len > self.pe.size(1):
            self._extend_pe(seq_len)
        return self.pe[:, :seq_len, :]

    def _extend_pe(self, new_max_len):
        old_max_len, dim = self.pe.size(1), self.pe.size(2)
        if new_max_len <= old_max_len:
            return
        extra_positions = torch.arange(old_max_len, new_max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        extra_pe = torch.zeros(new_max_len - old_max_len, dim)
        extra_pe[:, 0::2] = torch.sin(extra_positions * div_term)
        extra_pe[:, 1::2] = torch.cos(extra_positions * div_term)
        extra_pe = extra_pe.unsqueeze(0)
        new_pe = torch.cat([self.pe, extra_pe.to(self.pe.device)], dim=1)
        self.pe = new_pe

class CategorySpecificLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_categories: int = 1):
        super().__init__()
        self.num_categories = num_categories
        if num_categories <= 1:
            self.linear = nn.Linear(in_dim, out_dim)
        else:
            self.weight = nn.Parameter(torch.empty(num_categories, in_dim, out_dim))
            self.bias = nn.Parameter(torch.zeros(num_categories, out_dim))
            nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, category_id: torch.LongTensor):

        if self.num_categories <= 1:
            if x.dtype != self.linear.weight.dtype:
                x = x.to(dtype=self.linear.weight.dtype)
            return self.linear(x)

        if x.dtype != self.weight.dtype:
            x = x.to(dtype=self.weight.dtype)

        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1]) 
        if category_id.dim() == 0:
       
            cid = category_id.item()
            out = x_flat @ self.weight[cid] + self.bias[cid]
        else:
           
            category_id = category_id.reshape(-1)
            if category_id.numel() != x_flat.size(0):
                raise ValueError(
                    f"category_id length {category_id.numel()} does not match flattened batch {x_flat.size(0)}"
                )
            weight_selected = self.weight[category_id]        
            bias_selected = self.bias[category_id]        
            out = torch.bmm(x_flat.unsqueeze(1), weight_selected).squeeze(1) + bias_selected
        out_shape = orig_shape[:-1] + (out.shape[-1],)
        return out.view(out_shape)

class CategorySpecificMLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_categories: int = 1):
        super().__init__()
        self.fc1 = CategorySpecificLinear(input_dim, hidden_dim, num_categories)
        self.fc2 = CategorySpecificLinear(hidden_dim, output_dim, num_categories)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, category_id: torch.LongTensor):
        out = self.activation(self.fc1(x, category_id))
        out = self.fc2(out, category_id)
        return out

class MultiEmbodimentActionEncoder(nn.Module):

    def __init__(self, action_dim: int, embed_dim: int, hidden_dim: int, horizon: int, num_categories: int = 1):
        super().__init__()
        self.horizon = horizon
        self.embed_dim = embed_dim
        self.num_categories = num_categories
        
        self.W1 = CategorySpecificLinear(action_dim, hidden_dim, num_categories)
        self.W2 = CategorySpecificLinear(hidden_dim, hidden_dim, num_categories)
        self.W3 = CategorySpecificLinear(hidden_dim, embed_dim, num_categories)
   
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim, max_len=horizon)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, action_seq: torch.Tensor, category_id: torch.LongTensor):

        B, H, D = action_seq.shape
        assert H == self.horizon, "Action sequence length must match horizon"
       
        x = action_seq.reshape(B * H, D) 
      
        if category_id.dim() == 0:
           
            cat_ids = category_id.expand(H * B)
        else:
            cat_ids = category_id.unsqueeze(1).expand(B, H).reshape(B * H)
        out = self.activation(self.W1(x, cat_ids))            
    
        pos_enc = self.pos_encoding(H).to(device=out.device, dtype=out.dtype)    
        out = out.view(B, H, -1) + pos_enc
        out = out.view(B * H, -1)
        out = self.activation(self.W2(out, cat_ids))         
        out = self.W3(out, cat_ids)                        
        out = out.view(B, H, self.embed_dim)
        return out


class SlotConditionedStateEncoder(nn.Module):
    """Encode HSR robot state and scene slots as structured context tokens."""

    def __init__(
        self,
        robot_state_dim: int,
        slot_count: int,
        slot_dim: int,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_categories: int = 1,
    ):
        super().__init__()
        self.robot_state_dim = int(robot_state_dim)
        self.slot_count = int(slot_count)
        self.slot_dim = int(slot_dim)
        self.embed_dim = int(embed_dim)

        self.robot_encoder = CategorySpecificMLP(
            input_dim=self.robot_state_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            num_categories=num_categories,
        )
        self.slot_encoder = CategorySpecificMLP(
            input_dim=self.slot_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            num_categories=num_categories,
        )
        self.slot_index_embedding = nn.Parameter(torch.zeros(self.slot_count, embed_dim))
        nn.init.normal_(self.slot_index_embedding, std=0.02)
        self.slot_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.robot_norm = nn.LayerNorm(embed_dim)
        self.slot_norm = nn.LayerNorm(embed_dim)

    def forward(self, state: torch.Tensor, category_id: torch.LongTensor) -> torch.Tensor:
        expected_dim = self.robot_state_dim + self.slot_count * self.slot_dim
        if state.shape[-1] != expected_dim:
            raise ValueError(
                f"SlotConditionedStateEncoder expected state dim {expected_dim}, got {state.shape[-1]}"
            )

        robot_state = state[:, : self.robot_state_dim]
        slots = state[:, self.robot_state_dim :].reshape(
            state.shape[0], self.slot_count, self.slot_dim
        )

        robot_token = self.robot_encoder(robot_state, category_id)
        robot_token = self.robot_norm(robot_token).unsqueeze(1)

        if category_id.dim() == 0:
            slot_cat_ids = category_id.expand(state.shape[0] * self.slot_count)
        else:
            slot_cat_ids = category_id.unsqueeze(1).expand(
                state.shape[0], self.slot_count
            ).reshape(-1)

        slot_tokens = self.slot_encoder(
            slots.reshape(state.shape[0] * self.slot_count, self.slot_dim),
            slot_cat_ids,
        ).reshape(state.shape[0], self.slot_count, self.embed_dim)
        slot_tokens = slot_tokens + self.slot_index_embedding.to(
            device=slot_tokens.device, dtype=slot_tokens.dtype
        ).unsqueeze(0)

        valid = slots[..., -1] > 0.5
        key_padding_mask = ~valid
        all_invalid = key_padding_mask.all(dim=1)
        if all_invalid.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_invalid, 0] = False

        slot_context, _ = self.slot_attention(
            robot_token,
            slot_tokens,
            slot_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        slot_context = self.slot_norm(slot_context)
        return torch.cat([robot_token, slot_context], dim=1)


class VisualSceneSlotPredictor(nn.Module):
    """Predict compact scene slots from VLM tokens for end-to-end slot memory."""

    def __init__(
        self,
        slot_count: int,
        slot_dim: int,
        slot_position_dim: int,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.slot_count = int(slot_count)
        self.slot_dim = int(slot_dim)
        self.slot_position_dim = int(slot_position_dim)
        if self.slot_dim < 3:
            raise ValueError("VisualSceneSlotPredictor requires slot_dim >= 3: [pos..., valid]")
        if not (1 <= self.slot_position_dim <= self.slot_dim - 1):
            raise ValueError(
                f"slot_position_dim={self.slot_position_dim} is incompatible with slot_dim={self.slot_dim}"
            )

        self.slot_queries = nn.Parameter(torch.zeros(self.slot_count, embed_dim))
        nn.init.normal_(self.slot_queries, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.slot_position_dim + 1),
        )

    def forward(self, visual_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if visual_tokens.dim() == 2:
            visual_tokens = visual_tokens.unsqueeze(1)
        if visual_tokens.dim() != 3:
            raise ValueError(f"visual_tokens must be [B,D] or [B,N,D], got {tuple(visual_tokens.shape)}")

        batch_size = visual_tokens.shape[0]
        queries = self.slot_queries.to(
            device=visual_tokens.device,
            dtype=visual_tokens.dtype,
        ).unsqueeze(0).expand(batch_size, -1, -1)
        slot_features, _ = self.cross_attention(
            queries,
            visual_tokens,
            visual_tokens,
            need_weights=False,
        )
        raw = self.head(self.norm(slot_features))
        xy = torch.tanh(raw[..., : self.slot_position_dim])
        valid_logits = raw[..., self.slot_position_dim]
        valid_prob = torch.sigmoid(valid_logits).unsqueeze(-1)
        if self.slot_dim > self.slot_position_dim + 1:
            pad_dim = self.slot_dim - self.slot_position_dim - 1
            pad = torch.zeros(*xy.shape[:-1], pad_dim, device=xy.device, dtype=xy.dtype)
            slots = torch.cat([xy, pad, valid_prob], dim=-1)
        else:
            slots = torch.cat([xy, valid_prob], dim=-1)
        return {
            "slots": slots,
            "xy": xy,
            "valid_logits": valid_logits,
        }

    def loss(
        self,
        prediction: dict[str, torch.Tensor],
        target_slots: torch.Tensor | None,
        target_mask: torch.Tensor | None,
        valid_loss_weight: float,
    ) -> dict[str, torch.Tensor]:
        xy = prediction["xy"]
        valid_logits = prediction["valid_logits"]
        zero = xy.new_tensor(0.0)
        if target_slots is None or target_mask is None:
            return {
                "loss": zero,
                "xy_loss": zero,
                "valid_loss": zero,
                "xy_mae": zero,
                "valid_accuracy": zero,
                "has_target": torch.zeros((), device=xy.device, dtype=torch.bool),
            }

        target_slots = target_slots.to(device=xy.device, dtype=xy.dtype)
        target_mask = target_mask.to(device=xy.device, dtype=torch.bool)
        if target_slots.shape[:2] != xy.shape[:2]:
            raise ValueError(
                f"target_slots shape {tuple(target_slots.shape)} does not match prediction {tuple(xy.shape)}"
            )
        if target_slots.shape[-1] < self.slot_position_dim + 1:
            raise ValueError(
                f"target_slots last dim {target_slots.shape[-1]} is too small for slot_position_dim={self.slot_position_dim}"
            )

        target_xy = target_slots[..., : self.slot_position_dim]
        target_valid = target_slots[..., -1].clamp(0.0, 1.0)
        supervised = target_mask & (target_valid > 0.5)

        xy_error = F.smooth_l1_loss(xy.float(), target_xy.float(), reduction="none").mean(dim=-1)
        xy_weights = supervised.to(dtype=xy_error.dtype)
        xy_loss = (xy_error * xy_weights).sum() / xy_weights.sum().clamp_min(1.0)

        valid_targets = target_valid.float()
        valid_loss_all = F.binary_cross_entropy_with_logits(
            valid_logits.float(),
            valid_targets,
            reduction="none",
        )
        valid_weights = target_mask.to(dtype=valid_loss_all.dtype)
        valid_loss = (valid_loss_all * valid_weights).sum() / valid_weights.sum().clamp_min(1.0)

        xy_mae = ((xy - target_xy).abs().mean(dim=-1) * xy_weights).sum() / xy_weights.sum().clamp_min(1.0)
        valid_accuracy = (((valid_logits >= 0) == (target_valid > 0.5)).float() * valid_weights).sum()
        valid_accuracy = valid_accuracy / valid_weights.sum().clamp_min(1.0)
        total = xy_loss + float(valid_loss_weight) * valid_loss
        return {
            "loss": total,
            "xy_loss": xy_loss.detach(),
            "valid_loss": valid_loss.detach(),
            "xy_mae": xy_mae.detach(),
            "valid_accuracy": valid_accuracy.detach(),
            "has_target": target_mask.any(),
        }


class GeometricMemoryStateEncoder(nn.Module):
    """Encode robot state and slots with a learned geometric memory bank.

    The input slot layout is assumed to keep position in the first coordinates
    and valid in the last coordinate. Current State27 uses [dx, dy, valid], and
    future 3D slots can use [dx, dy, dz, valid].
    """

    def __init__(
        self,
        robot_state_dim: int,
        slot_count: int,
        slot_dim: int,
        slot_position_dim: int,
        memory_slot_count: int,
        use_visual_tokens: bool,
        relation_min_distance: float,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_categories: int = 1,
    ):
        super().__init__()
        self.robot_state_dim = int(robot_state_dim)
        self.slot_count = int(slot_count)
        self.slot_dim = int(slot_dim)
        self.slot_position_dim = int(slot_position_dim)
        self.memory_slot_count = int(memory_slot_count)
        self.use_visual_tokens = bool(use_visual_tokens)
        self.relation_min_distance = float(relation_min_distance)
        self.embed_dim = int(embed_dim)

        if self.slot_dim < 3:
            raise ValueError("GeometricMemoryStateEncoder requires slot_dim >= 3: [pos..., valid]")
        if not (1 <= self.slot_position_dim <= self.slot_dim - 1):
            raise ValueError(
                "slot_position_dim must be in [1, slot_dim - 1], got "
                f"{self.slot_position_dim} for slot_dim={self.slot_dim}"
            )
        if self.relation_min_distance < 0:
            raise ValueError(f"relation_min_distance must be >= 0, got {self.relation_min_distance}")

        self.robot_encoder = CategorySpecificMLP(
            input_dim=self.robot_state_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            num_categories=num_categories,
        )
        self.slot_encoder = CategorySpecificMLP(
            input_dim=self.slot_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            num_categories=num_categories,
        )

        relation_input_dim = self.slot_dim * 2 + 2 * self.slot_position_dim + 2
        self.relation_encoder = CategorySpecificMLP(
            input_dim=relation_input_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            num_categories=num_categories,
        )
        self.memory_candidate = CategorySpecificMLP(
            input_dim=embed_dim * (5 if self.use_visual_tokens else 4),
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            num_categories=num_categories,
        )
        self.memory_gate = CategorySpecificMLP(
            input_dim=embed_dim * (5 if self.use_visual_tokens else 4),
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            num_categories=num_categories,
        )
        self.association_query_proj = CategorySpecificLinear(
            in_dim=embed_dim,
            out_dim=embed_dim,
            num_categories=num_categories,
        )
        self.relation_query_proj = CategorySpecificLinear(
            in_dim=embed_dim,
            out_dim=embed_dim,
            num_categories=num_categories,
        )

        self.slot_index_embedding = nn.Parameter(torch.zeros(self.slot_count, embed_dim))
        self.memory_query = nn.Parameter(torch.zeros(self.memory_slot_count, embed_dim))
        nn.init.normal_(self.slot_index_embedding, std=0.02)
        nn.init.normal_(self.memory_query, std=0.02)

        self.slot_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.association_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.visual_memory_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.memory_read_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.robot_norm = nn.LayerNorm(embed_dim)
        self.slot_norm = nn.LayerNorm(embed_dim)
        self.relation_norm = nn.LayerNorm(embed_dim)
        self.geometry_query_norm = nn.LayerNorm(embed_dim)
        self.visual_context_norm = nn.LayerNorm(embed_dim)
        self.memory_norm = nn.LayerNorm(embed_dim)
        self.memory_context_norm = nn.LayerNorm(embed_dim)

    def _expand_category_ids(
        self,
        category_id: torch.LongTensor,
        batch_size: int,
        repeat_count: int,
    ) -> torch.LongTensor:
        if category_id.dim() == 0:
            return category_id.expand(batch_size * repeat_count)
        return category_id.unsqueeze(1).expand(batch_size, repeat_count).reshape(-1)

    def _masked_attention(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        valid: torch.Tensor,
        attention: nn.MultiheadAttention,
    ) -> torch.Tensor:
        key_padding_mask = ~valid
        all_invalid = key_padding_mask.all(dim=1)
        if all_invalid.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_invalid, 0] = False
        out, _ = attention(
            query,
            key_value,
            key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        if all_invalid.any():
            out = out.clone()
            out[all_invalid] = 0
        return out

    def _relation_token(
        self,
        slots: torch.Tensor,
        valid: torch.Tensor,
        category_id: torch.LongTensor,
    ) -> torch.Tensor:
        batch_size = slots.shape[0]
        slot_i = slots.unsqueeze(2).expand(batch_size, self.slot_count, self.slot_count, self.slot_dim)
        slot_j = slots.unsqueeze(1).expand(batch_size, self.slot_count, self.slot_count, self.slot_dim)

        delta_pos = slot_j[..., : self.slot_position_dim] - slot_i[..., : self.slot_position_dim]
        dist = torch.linalg.norm(delta_pos, dim=-1, keepdim=True)
        min_distance = torch.as_tensor(
            self.relation_min_distance,
            device=slots.device,
            dtype=slots.dtype,
        )
        non_self = ~torch.eye(self.slot_count, dtype=torch.bool, device=slots.device).unsqueeze(0)
        far_enough = dist.squeeze(-1) > min_distance
        pair_valid_bool = valid.unsqueeze(2) & valid.unsqueeze(1) & non_self & far_enough
        pair_valid = pair_valid_bool.to(dtype=slots.dtype).unsqueeze(-1)
        safe_dist = dist.clamp_min(max(self.relation_min_distance, 1e-6))
        unit_pos = torch.where(pair_valid_bool.unsqueeze(-1), delta_pos / safe_dist, torch.zeros_like(delta_pos))

        relation_features = torch.cat(
            [slot_i, slot_j, delta_pos, dist, unit_pos, pair_valid],
            dim=-1,
        )
        relation_count = self.slot_count * self.slot_count
        relation_cat_ids = self._expand_category_ids(category_id, batch_size, relation_count)
        relation_tokens = self.relation_encoder(
            relation_features.reshape(batch_size * relation_count, -1),
            relation_cat_ids,
        ).reshape(batch_size, relation_count, self.embed_dim)

        weights = pair_valid.reshape(batch_size, relation_count, 1)
        relation_sum = (relation_tokens * weights).sum(dim=1, keepdim=True)
        denom = weights.sum(dim=1, keepdim=True)
        has_relation = denom > 0
        relation_mean = relation_sum / denom.clamp_min(1.0)
        relation_mean = torch.where(has_relation, relation_mean, torch.zeros_like(relation_mean))
        relation_token = self.relation_norm(relation_mean)
        return torch.where(has_relation, relation_token, torch.zeros_like(relation_token))

    def _visual_context(
        self,
        query_tokens: torch.Tensor,
        visual_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.use_visual_tokens or visual_tokens is None:
            return torch.zeros_like(query_tokens)
        if visual_tokens.dim() == 2:
            visual_tokens = visual_tokens.unsqueeze(1)
        if visual_tokens.dim() != 3:
            raise ValueError(
                "visual_tokens must have shape [B, D] or [B, N, D], got "
                f"{tuple(visual_tokens.shape)}"
            )
        visual_tokens = visual_tokens.to(device=query_tokens.device, dtype=query_tokens.dtype)
        if visual_tokens.shape[0] != query_tokens.shape[0]:
            raise ValueError(
                f"visual_tokens batch {visual_tokens.shape[0]} != state batch {query_tokens.shape[0]}"
            )
        if visual_tokens.shape[-1] != self.embed_dim:
            raise ValueError(
                f"visual_tokens dim {visual_tokens.shape[-1]} != embed_dim {self.embed_dim}"
            )
        visual_context, _ = self.visual_memory_attention(
            query_tokens,
            visual_tokens,
            visual_tokens,
            need_weights=False,
        )
        return self.visual_context_norm(visual_context)

    def _geometry_conditioned_query(
        self,
        memory_prior: torch.Tensor,
        associated_slots: torch.Tensor,
        relation_token: torch.Tensor,
        category_id: torch.LongTensor,
    ) -> torch.Tensor:
        batch_size = memory_prior.shape[0]
        memory_cat_ids = self._expand_category_ids(category_id, batch_size, self.memory_slot_count)
        relation_per_memory = relation_token.expand(-1, self.memory_slot_count, -1)
        associated_delta = self.association_query_proj(
            associated_slots.reshape(batch_size * self.memory_slot_count, self.embed_dim),
            memory_cat_ids,
        ).reshape(batch_size, self.memory_slot_count, self.embed_dim)
        relation_delta = self.relation_query_proj(
            relation_per_memory.reshape(batch_size * self.memory_slot_count, self.embed_dim),
            memory_cat_ids,
        ).reshape(batch_size, self.memory_slot_count, self.embed_dim)
        return self.geometry_query_norm(memory_prior + associated_delta + relation_delta)

    def forward(
        self,
        state: torch.Tensor,
        category_id: torch.LongTensor,
        visual_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected_dim = self.robot_state_dim + self.slot_count * self.slot_dim
        if state.shape[-1] != expected_dim:
            raise ValueError(
                f"GeometricMemoryStateEncoder expected state dim {expected_dim}, got {state.shape[-1]}"
            )

        batch_size = state.shape[0]
        robot_state = state[:, : self.robot_state_dim]
        slots = state[:, self.robot_state_dim :].reshape(
            batch_size, self.slot_count, self.slot_dim
        )
        valid = slots[..., -1] > 0.5

        robot_token = self.robot_encoder(robot_state, category_id)
        robot_token = self.robot_norm(robot_token).unsqueeze(1)

        slot_cat_ids = self._expand_category_ids(category_id, batch_size, self.slot_count)
        slot_tokens = self.slot_encoder(
            slots.reshape(batch_size * self.slot_count, self.slot_dim),
            slot_cat_ids,
        ).reshape(batch_size, self.slot_count, self.embed_dim)
        slot_tokens = slot_tokens + self.slot_index_embedding.to(
            device=slot_tokens.device, dtype=slot_tokens.dtype
        ).unsqueeze(0)

        slot_context = self._masked_attention(robot_token, slot_tokens, valid, self.slot_attention)
        slot_context = self.slot_norm(slot_context)

        relation_token = self._relation_token(slots, valid, category_id)

        memory_prior = self.memory_query.to(
            device=slot_tokens.device, dtype=slot_tokens.dtype
        ).unsqueeze(0).expand(batch_size, -1, -1)
        associated_slots = self._masked_attention(
            memory_prior,
            slot_tokens,
            valid,
            self.association_attention,
        )
        geometry_query = self._geometry_conditioned_query(
            memory_prior,
            associated_slots,
            relation_token,
            category_id,
        )
        visual_context = self._visual_context(geometry_query, visual_tokens)

        memory_input_parts = [
            geometry_query,
            associated_slots,
            relation_token.expand(-1, self.memory_slot_count, -1),
            robot_token.expand(-1, self.memory_slot_count, -1),
        ]
        if self.use_visual_tokens:
            memory_input_parts.append(visual_context)
        memory_input = torch.cat(memory_input_parts, dim=-1)
        memory_cat_ids = self._expand_category_ids(category_id, batch_size, self.memory_slot_count)
        candidate = torch.tanh(
            self.memory_candidate(
                memory_input.reshape(batch_size * self.memory_slot_count, -1),
                memory_cat_ids,
            ).reshape(batch_size, self.memory_slot_count, self.embed_dim)
        )
        gate = torch.sigmoid(
            self.memory_gate(
                memory_input.reshape(batch_size * self.memory_slot_count, -1),
                memory_cat_ids,
            ).reshape(batch_size, self.memory_slot_count, self.embed_dim)
        )
        memory_tokens = self.memory_norm((1.0 - gate) * memory_prior + gate * candidate)

        memory_context, _ = self.memory_read_attention(
            robot_token,
            memory_tokens,
            memory_tokens,
            need_weights=False,
        )
        memory_context = self.memory_context_norm(memory_context)

        return torch.cat(
            [robot_token, slot_context, relation_token, memory_context, memory_tokens],
            dim=1,
        )

class BasicTransformerBlock(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, action_tokens: torch.Tensor, context_tokens: torch.Tensor, time_emb: torch.Tensor):

        x = self.norm1(action_tokens)
        attn_out, _ = self.attn(x, context_tokens, context_tokens)

        x = action_tokens + attn_out

        x2 = self.norm2(x)

        if time_emb is not None:
            x2 = x2 + time_emb.unsqueeze(1)
        ff_out = self.ff(x2)
        x = x + ff_out
        return x

class FlowmatchingActionHead(nn.Module):

    def __init__(self, config=None,
                 embed_dim: int = 896, 
                 hidden_dim: int = 1024,
                 action_dim: int = 16*7,
                 horizon: int = 16,
                 per_action_dim: int = 7,
                 num_heads: int = 8,
                 num_layers: int = 8,
                 dropout: float = 0.0,
                 num_inference_timesteps: int = 20,
                 num_categories: int = 1):
        super().__init__()

        if config is not None:
      
            embed_dim = getattr(config, "embed_dim", embed_dim)
            hidden_dim = getattr(config, "hidden_dim", hidden_dim)
            action_dim = getattr(config, "action_dim", action_dim)
            horizon = getattr(config, "horizon", horizon)
            num_heads = getattr(config, "num_heads", num_heads)
            num_layers = getattr(config, "num_layers", num_layers)
            dropout = getattr(config, "dropout", dropout)
            num_inference_timesteps = getattr(config, "num_inference_timesteps", num_inference_timesteps)
            num_categories = getattr(config, "num_categories", num_categories)
            self.config = config
        else:
            from config import EvoConfig
            self.config = EvoConfig(
                embed_dim=embed_dim, hidden_dim=hidden_dim,
                action_horizon=horizon, per_action_dim=per_action_dim,
                num_heads=num_heads, num_layers=num_layers,
                dropout=dropout, num_inference_timesteps=num_inference_timesteps,
                num_categories=num_categories
            )
        logger.info("FlowmatchingActionHead num_inference_timesteps=%s", num_inference_timesteps)
        self.embed_dim = embed_dim
        self.horizon = horizon
        self.per_action_dim = getattr(self.config, "per_action_dim", per_action_dim)
        self.action_dim = getattr(self.config, "action_dim", action_dim)


        self.time_pos_enc = SinusoidalPositionalEncoding(embed_dim, max_len=1000)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(embed_dim=embed_dim, num_heads=num_heads,
                                   hidden_dim=embed_dim*4, dropout=dropout)
            for _ in range(num_layers)
        ])
       
        self.norm_out = nn.LayerNorm(embed_dim)
        self.seq_pool_proj = nn.Linear(self.horizon * self.embed_dim, self.embed_dim)

        self.mlp_head = CategorySpecificMLP(input_dim=embed_dim, hidden_dim=hidden_dim,
                                            output_dim=action_dim, num_categories=num_categories)

        self.state_encoder = None
        self.state_encoder_type = getattr(self.config, "state_encoder_type", "flat")
        self.predict_scene_slots = bool(getattr(self.config, "predict_scene_slots", False))
        self.slot_aux_loss_weight = float(getattr(self.config, "slot_aux_loss_weight", 0.0))
        self.slot_valid_loss_weight = float(getattr(self.config, "slot_valid_loss_weight", 0.25))
        self.scene_slot_predictor = None
        if hasattr(self.config, "state_dim") and self.config.state_dim is not None:
       
            state_hidden = getattr(self.config, "state_hidden_dim", embed_dim)
            if self.state_encoder_type == "flat":
                self.state_encoder = CategorySpecificMLP(input_dim=self.config.state_dim,
                                                        hidden_dim=state_hidden,
                                                        output_dim=embed_dim,
                                                        num_categories=num_categories)
            elif self.state_encoder_type == "slot_attention":
                robot_state_dim = getattr(self.config, "robot_state_dim", 15)
                slot_count = getattr(self.config, "slot_count", 4)
                slot_dim = getattr(self.config, "slot_dim", 7)
                expected_state_dim = robot_state_dim + slot_count * slot_dim
                if int(self.config.state_dim) != int(expected_state_dim):
                    raise ValueError(
                        "slot_attention state encoder requires "
                        f"state_dim={expected_state_dim}, got {self.config.state_dim}"
                    )
                self.state_encoder = SlotConditionedStateEncoder(
                    robot_state_dim=robot_state_dim,
                    slot_count=slot_count,
                    slot_dim=slot_dim,
                    embed_dim=embed_dim,
                    hidden_dim=state_hidden,
                    num_heads=num_heads,
                    num_categories=num_categories,
                )
            elif self.state_encoder_type == "geometric_memory":
                robot_state_dim = getattr(self.config, "robot_state_dim", 15)
                slot_count = getattr(self.config, "slot_count", 4)
                slot_dim = getattr(self.config, "slot_dim", 3)
                slot_position_dim = getattr(self.config, "slot_position_dim", 2)
                memory_slot_count = getattr(self.config, "memory_slot_count", slot_count)
                memory_use_visual_tokens = getattr(self.config, "memory_use_visual_tokens", True)
                slot_relation_min_distance = getattr(self.config, "slot_relation_min_distance", 1e-3)
                expected_state_dim = robot_state_dim + slot_count * slot_dim
                if (not self.predict_scene_slots) and int(self.config.state_dim) != int(expected_state_dim):
                    raise ValueError(
                        "geometric_memory state encoder requires "
                        f"state_dim={expected_state_dim}, got {self.config.state_dim}"
                    )
                if self.predict_scene_slots:
                    if int(self.config.state_dim) not in {int(robot_state_dim), int(expected_state_dim)}:
                        raise ValueError(
                            "predict_scene_slots geometric_memory accepts state_dim="
                            f"{robot_state_dim} or {expected_state_dim}, got {self.config.state_dim}"
                        )
                    self.scene_slot_predictor = VisualSceneSlotPredictor(
                        slot_count=slot_count,
                        slot_dim=slot_dim,
                        slot_position_dim=slot_position_dim,
                        embed_dim=embed_dim,
                        hidden_dim=getattr(self.config, "slot_predictor_hidden_dim", 512),
                        num_heads=num_heads,
                    )
                self.state_encoder = GeometricMemoryStateEncoder(
                    robot_state_dim=robot_state_dim,
                    slot_count=slot_count,
                    slot_dim=slot_dim,
                    slot_position_dim=slot_position_dim,
                    memory_slot_count=memory_slot_count,
                    use_visual_tokens=memory_use_visual_tokens,
                    relation_min_distance=slot_relation_min_distance,
                    embed_dim=embed_dim,
                    hidden_dim=state_hidden,
                    num_heads=num_heads,
                    num_categories=num_categories,
                )
            else:
                raise ValueError(f"Unsupported state_encoder_type: {self.state_encoder_type}")

        self.action_encoder = None
        if horizon > 1:
          
            per_action_dim = getattr(self.config, "per_action_dim", None)
            if per_action_dim is None:
            
                per_action_dim = action_dim // horizon if action_dim % horizon == 0 else action_dim
            self.action_encoder = MultiEmbodimentActionEncoder(action_dim=per_action_dim,
                                                               embed_dim=embed_dim,
                                                               hidden_dim=embed_dim,  
                                                               horizon=horizon,
                                                               num_categories=num_categories)
            self.single_action_proj = None
        else:
            self.action_encoder = None
            self.single_action_proj = nn.Linear(self.per_action_dim, self.embed_dim)


    def _state_context_tokens(
        self,
        state: torch.Tensor,
        embodiment_id: torch.LongTensor,
        visual_tokens: torch.Tensor | None = None,
        slot_targets: torch.Tensor | None = None,
        slot_target_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, dict]:
        if state is None or self.state_encoder is None:
            return None, {}
        if self.state_encoder_type == "flat":
            return self.state_encoder(state, embodiment_id).unsqueeze(1), {}
        if self.state_encoder_type == "slot_attention":
            return self.state_encoder(state, embodiment_id), {}
        if self.state_encoder_type == "geometric_memory":
            state_for_encoder = state
            aux_outputs = {}
            if self.predict_scene_slots:
                if self.scene_slot_predictor is None:
                    raise RuntimeError("predict_scene_slots=True but scene_slot_predictor is not initialized")
                if visual_tokens is None:
                    raise ValueError("predict_scene_slots requires visual_tokens")
                slot_prediction = self.scene_slot_predictor(self._ensure_context_sequence(visual_tokens))
                robot_state = state[:, : self.state_encoder.robot_state_dim]
                predicted_slots = slot_prediction["slots"].to(dtype=robot_state.dtype)
                state_for_encoder = torch.cat(
                    [robot_state, predicted_slots.reshape(state.shape[0], -1)],
                    dim=-1,
                )
                slot_loss = self.scene_slot_predictor.loss(
                    slot_prediction,
                    target_slots=slot_targets,
                    target_mask=slot_target_mask,
                    valid_loss_weight=self.slot_valid_loss_weight,
                )
                aux_outputs = {
                    "predicted_scene_slots": predicted_slots,
                    "slot_valid_logits": slot_prediction["valid_logits"],
                    "slot_loss": slot_loss["loss"],
                    "slot_xy_loss": slot_loss["xy_loss"],
                    "slot_valid_loss": slot_loss["valid_loss"],
                    "slot_xy_mae": slot_loss["xy_mae"],
                    "slot_valid_accuracy": slot_loss["valid_accuracy"],
                    "slot_has_target": slot_loss["has_target"],
                }
            return self.state_encoder(state_for_encoder, embodiment_id, visual_tokens=visual_tokens), aux_outputs
        raise ValueError(f"Unsupported state_encoder_type: {self.state_encoder_type}")

    def _ensure_context_sequence(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        if fused_tokens.dim() == 2:
            return fused_tokens.unsqueeze(1)
        if fused_tokens.dim() != 3:
            raise ValueError(
                "fused_tokens must have shape [B, D] or [B, N, D], got "
                f"{tuple(fused_tokens.shape)}"
            )
        return fused_tokens

    def _project_actions(self, action_seq: torch.Tensor, embodiment_id: torch.LongTensor) -> torch.Tensor:
        if self.horizon > 1 and self.action_encoder is not None:
            return self.action_encoder(action_seq, embodiment_id)
        if self.single_action_proj is None:
            raise RuntimeError("single_action_proj is not initialized for horizon <= 1.")
        return self.single_action_proj(action_seq)

    def _expand_action_mask(self, action_mask: torch.Tensor, batch_size: int, per_action_dim: int, device, dtype):
        if action_mask is None:
            raise ValueError("action_mask must be provided for flow matching inference.")

        if action_mask.dim() == 2:
            expected_last_dim = self.horizon * per_action_dim
            if action_mask.shape == (batch_size, expected_last_dim):
                expanded_mask = action_mask.reshape(batch_size, self.horizon, per_action_dim)
            elif action_mask.shape == (batch_size, per_action_dim):
                expanded_mask = action_mask.unsqueeze(1).expand(batch_size, self.horizon, per_action_dim)
            else:
                raise ValueError(
                    f"Expected action_mask shape {(batch_size, expected_last_dim)} or {(batch_size, per_action_dim)}, got {tuple(action_mask.shape)}"
                )
        elif action_mask.dim() == 3:
            expected_shape = (batch_size, self.horizon, per_action_dim)
            if tuple(action_mask.shape) != expected_shape:
                raise ValueError(f"Expected action_mask shape {expected_shape}, got {tuple(action_mask.shape)}")
            expanded_mask = action_mask
        else:
            raise ValueError(f"Unsupported action_mask rank: {action_mask.dim()}")

        return expanded_mask.to(device=device, dtype=dtype)

    def forward(self, fused_tokens: torch.Tensor, state: torch.Tensor = None,
                actions_gt: torch.Tensor = None, embodiment_id: torch.LongTensor = None, 
                state_mask: torch.Tensor = None, action_mask: torch.Tensor = None,
                slot_targets: torch.Tensor | None = None,
                slot_target_mask: torch.Tensor | None = None):

        if actions_gt is None:
            return self.get_action(fused_tokens, state=state, embodiment_id=embodiment_id, action_mask=action_mask)
        B = fused_tokens.size(0)
        device = fused_tokens.device

        if embodiment_id is None:
            embodiment_id = torch.zeros(B, dtype=torch.long, device=device)

        context_tokens = self._ensure_context_sequence(fused_tokens)
        aux_outputs = {}
        if state is not None and self.state_encoder is not None:
            state_tokens, aux_outputs = self._state_context_tokens(
                state,
                embodiment_id,
                visual_tokens=fused_tokens,
                slot_targets=slot_targets,
                slot_target_mask=slot_target_mask,
            )
            context_tokens = torch.cat([context_tokens, state_tokens], dim=1) 

        t = torch.distributions.Beta(2, 2).sample((B,)).clamp(0.02, 0.98).to(device).to(dtype=self.dtype)

        
                    
        time_index = (t * 999).long().clamp_(0, 999)  
        time_emb = self.time_pos_enc(1000)[:, time_index, :].squeeze(0).to(dtype=context_tokens.dtype)
    
        action_shape = actions_gt.shape[1]  
    

        actions_gt_seq = actions_gt  


        noise = torch.rand_like(actions_gt) * 2 - 1  

        if action_mask is not None:
            action_mask = action_mask.to(dtype=noise.dtype, device=noise.device)
            assert action_mask.shape == noise.shape, f"action_mask shape {action_mask.shape} != noise shape {noise.shape}"
            noise = noise * action_mask


        if self.horizon > 1:
            noise_seq = noise.view(B, self.horizon, self.per_action_dim)
            
        else:
            noise_seq = noise.unsqueeze(1)

        if self.horizon > 1:
            t_broadcast = t.view(B, 1, 1)
        else:
            t_broadcast = t.view(B, 1)
        action_intermediate_seq = (1 - t_broadcast) * noise_seq + t_broadcast * actions_gt_seq  

        action_tokens = self._project_actions(action_intermediate_seq, embodiment_id)
        
        target_dtype = self.dtype
        action_tokens = action_tokens.to(dtype=target_dtype)
        context_tokens = context_tokens.to(dtype=target_dtype)
        time_emb = time_emb.to(dtype=target_dtype)

        x = action_tokens  
        for block in self.transformer_blocks:
            x = block(x, context_tokens, time_emb)

        x = self.norm_out(x)  

        if self.horizon > 1:
 
            x_flat = x.reshape(B, -1)  
            x_pooled = self.seq_pool_proj(x_flat)  
        else:
          
            x_pooled = x.squeeze(1) 

        pred_velocity = self.mlp_head(x_pooled, embodiment_id) 

        if aux_outputs:
            return pred_velocity, noise, aux_outputs
        return pred_velocity, noise

    def get_action(self, fused_tokens: torch.Tensor, state: torch.Tensor = None, embodiment_id: torch.LongTensor = None, action_mask: torch.Tensor = None):

        B = fused_tokens.size(0)
        device = fused_tokens.device
        if embodiment_id is None:
            embodiment_id = torch.zeros(B, dtype=torch.long, device=device)

        context_tokens = self._ensure_context_sequence(fused_tokens)
        if state is not None and self.state_encoder is not None:
            state_tokens, _ = self._state_context_tokens(state, embodiment_id, visual_tokens=fused_tokens)
            context_tokens = torch.cat([context_tokens, state_tokens], dim=1)

        action_dim_total = getattr(self.config, "action_dim", None)
        if action_dim_total is None:
          
            action_dim_total = self.action_dim
       
        if self.horizon > 1:
            per_action_dim = getattr(self.config, "per_action_dim", action_dim_total // self.horizon)
        else:
            per_action_dim = action_dim_total

        action = (torch.rand(B, action_dim_total, device=device, dtype=context_tokens.dtype) * 2 - 1)

        if self.horizon > 1:
            action_seq = action.view(B, self.horizon, per_action_dim)
        else:
            action_seq = action.view(B, 1, per_action_dim)

        action_mask = self._expand_action_mask(
            action_mask, batch_size=B, per_action_dim=per_action_dim, device=action_seq.device, dtype=action_seq.dtype
        )
        action_seq = action_seq * action_mask
        
        target_dtype = self.dtype
        context_tokens = context_tokens.to(dtype=target_dtype)
        
        N = int(getattr(self.config, "num_inference_timesteps", 32))
        if N <= 0:
            raise ValueError(f"num_inference_timesteps must be positive, got {N}")
        dt = 1.0 / N
        for i in range(N):
            t = i / N 

            time_index = min(int(t * 999), 999)
            time_emb = self.time_pos_enc(1000)[:, time_index, :].to(device).squeeze(0).to(dtype=context_tokens.dtype)
            time_emb = time_emb.unsqueeze(0).repeat(B, 1)  


            action_seq = action_seq * action_mask
            action_tokens = self._project_actions(action_seq, embodiment_id)
            action_tokens = action_tokens.to(dtype=target_dtype)
            time_emb = time_emb.to(dtype=target_dtype)

            x = action_tokens
            for block in self.transformer_blocks:
                x = block(x, context_tokens, time_emb)
            x = self.norm_out(x)

            if self.horizon > 1:
                x_flat = x.reshape(B, -1)
                x_pooled = self.seq_pool_proj(x_flat)
            else:
                x_pooled = x.squeeze(1)
         
            pred = self.mlp_head(x_pooled, embodiment_id)  
  
            action = action + dt * pred
          
            if self.horizon > 1:
                action_seq = action.view(B, self.horizon, per_action_dim)
            else:
                action_seq = action.view(B, 1, per_action_dim)
      
        action_seq = action_seq * action_mask
        return action_seq.reshape(B, -1)

    @property
    def device(self):
      
        return next(self.parameters()).device

    @property
    def dtype(self):
        
        return next(self.parameters()).dtype
