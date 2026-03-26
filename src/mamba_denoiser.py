import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


def apply_mask(hidden, valid_mask):
    if valid_mask is None:
        return hidden
    return hidden * valid_mask.unsqueeze(-1)


def modulate(hidden, shift, scale):
    return hidden * (1 + scale) + shift


def masked_tensor_stats(tensor, valid_mask):
    if valid_mask is None:
        flat_tensor = tensor.reshape(-1)
    else:
        valid_positions = valid_mask > 0
        if valid_positions.any():
            flat_tensor = tensor[valid_positions]
        else:
            flat_tensor = tensor.new_zeros(1)

    return {
        "mean": flat_tensor.mean().detach(),
        "std": flat_tensor.std(unbiased=False).detach(),
        "min": flat_tensor.min().detach(),
        "max": flat_tensor.max().detach(),
    }


def prefixed_stats(prefix, stats):
    return {
        f"{prefix}_mean": stats["mean"],
        f"{prefix}_std": stats["std"],
        f"{prefix}_min": stats["min"],
        f"{prefix}_max": stats["max"],
    }


class LocalSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=2, window_size=5, attn_dim=None, causal=True):
        super().__init__()
        attn_dim = hidden_size if attn_dim is None else attn_dim
        if attn_dim % num_heads != 0:
            raise ValueError("local attention dim must be divisible by num_heads")
        if window_size < 1:
            raise ValueError("local attention window_size must be >= 1")

        self.num_heads = num_heads
        self.window_size = window_size
        self.causal = causal
        self.attn_dim = attn_dim
        self.head_dim = attn_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden_size, attn_dim * 3)
        self.out_proj = nn.Linear(attn_dim, hidden_size)

        nn.init.normal_(self.qkv.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.qkv.bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, hidden, valid_mask=None):
        batch_size, seq_len, _ = hidden.shape
        qkv = self.qkv(hidden).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.causal:
            left_pad = self.window_size - 1
            right_pad = 0
            effective_window = self.window_size
        else:
            radius = self.window_size // 2
            left_pad = radius
            right_pad = radius
            effective_window = left_pad + right_pad + 1

        k_padded = F.pad(k, (0, 0, left_pad, right_pad))
        v_padded = F.pad(v, (0, 0, left_pad, right_pad))
        k_windows = k_padded.unfold(dimension=2, size=effective_window, step=1).permute(0, 1, 2, 4, 3)
        v_windows = v_padded.unfold(dimension=2, size=effective_window, step=1).permute(0, 1, 2, 4, 3)

        if valid_mask is None:
            window_valid = hidden.new_ones(batch_size, seq_len, effective_window, dtype=torch.bool)
        else:
            mask_padded = F.pad(valid_mask.bool(), (left_pad, right_pad), value=False)
            window_valid = mask_padded.unfold(dimension=1, size=effective_window, step=1)

        attn_scores = torch.einsum("bhld,bhlwd->bhlw", q, k_windows) * self.scale
        attn_scores = attn_scores.masked_fill(~window_valid.unsqueeze(1), -1e4)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = attn_probs * window_valid.unsqueeze(1).to(attn_probs.dtype)
        attn_probs = attn_probs / attn_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        context = torch.einsum("bhlw,bhlwd->bhld", attn_probs, v_windows)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.attn_dim)
        return self.out_proj(context)


class MambaBlock(nn.Module):
    def __init__(self, hidden_size, dropout, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "dif_decoder='mamba' requires the optional package 'mamba_ssm'. "
                "Install it before running the Mamba-denoiser experiment."
            )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.mamba = Mamba(
            d_model=hidden_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, valid_mask=None):
        hidden = apply_mask(hidden, valid_mask)

        residual = hidden
        hidden = self.norm1(hidden)
        hidden = self.mamba(hidden)
        hidden = residual + self.dropout(hidden)

        residual = hidden
        hidden = self.norm2(hidden)
        hidden = self.ffn(hidden)
        hidden = residual + self.dropout(hidden)

        hidden = apply_mask(hidden, valid_mask)
        return hidden


class MambaDenoiser(nn.Module):
    def __init__(self, args, num_blocks):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    hidden_size=args.hidden_size,
                    dropout=args.dropout,
                    d_state=args.mamba_d_state,
                    d_conv=args.mamba_d_conv,
                    expand=args.mamba_expand,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, hidden, valid_mask):
        for layer in self.layers:
            hidden = layer(hidden, valid_mask)
        return hidden


class AdaLNConditionalMambaBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        dropout,
        alpha_max=0.1,
        use_input_coupling=False,
        use_local_attention=False,
        local_attn_window=5,
        local_attn_heads=2,
        local_attn_dim=None,
        causal=True,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "AdaLN-conditioned Mamba decoders require the optional package 'mamba_ssm'."
            )

        self.use_input_coupling = use_input_coupling
        self.use_local_attention = use_local_attention
        self.alpha_max = alpha_max
        self.norm1 = nn.LayerNorm(hidden_size)
        self.mamba = Mamba(
            d_model=hidden_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm_attn = nn.LayerNorm(hidden_size)
        if self.use_local_attention:
            self.local_attention = LocalSelfAttention(
                hidden_size=hidden_size,
                num_heads=local_attn_heads,
                window_size=local_attn_window,
                attn_dim=local_attn_dim,
                causal=causal,
            )
        else:
            self.local_attention = None
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        modulation_chunks = 9 if self.use_local_attention else 6
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * modulation_chunks),
        )
        self.input_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2),
        )
        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        nn.init.constant_(self.modulation[-1].bias[2 * hidden_size:3 * hidden_size], -4.0)
        if self.use_local_attention:
            nn.init.constant_(self.modulation[-1].bias[5 * hidden_size:6 * hidden_size], -4.0)
            nn.init.constant_(self.modulation[-1].bias[8 * hidden_size:9 * hidden_size], -4.0)
        else:
            nn.init.constant_(self.modulation[-1].bias[5 * hidden_size:6 * hidden_size], -4.0)
        nn.init.zeros_(self.input_modulation[-1].weight)
        nn.init.zeros_(self.input_modulation[-1].bias)

    def forward(self, hidden, cond_ctx, valid_mask=None):
        hidden = apply_mask(hidden, valid_mask)
        stats = {}

        if self.use_input_coupling:
            input_shift, input_scale = self.input_modulation(cond_ctx).chunk(2, dim=-1)
            hidden = modulate(hidden, input_shift, input_scale)
            hidden = apply_mask(hidden, valid_mask)
            stats.update(prefixed_stats("adaln_input_shift", masked_tensor_stats(input_shift, valid_mask)))
            stats.update(prefixed_stats("adaln_input_scale", masked_tensor_stats(input_scale, valid_mask)))

        if self.use_local_attention:
            shift_m, scale_m, gate_m_raw, shift_a, scale_a, gate_a_raw, shift_f, scale_f, gate_f_raw = self.modulation(cond_ctx).chunk(9, dim=-1)
            gate_a = self.alpha_max * torch.sigmoid(gate_a_raw)
        else:
            shift_m, scale_m, gate_m_raw, shift_f, scale_f, gate_f_raw = self.modulation(cond_ctx).chunk(6, dim=-1)
            gate_a = None
        gate_m = self.alpha_max * torch.sigmoid(gate_m_raw)
        gate_f = self.alpha_max * torch.sigmoid(gate_f_raw)

        residual = hidden
        hidden_m = modulate(self.norm1(hidden), shift_m, scale_m)
        hidden_m = self.mamba(hidden_m)
        hidden = residual + gate_m * self.dropout(hidden_m)

        if self.use_local_attention:
            residual = hidden
            hidden_a = modulate(self.norm_attn(hidden), shift_a, scale_a)
            hidden_a = self.local_attention(hidden_a, valid_mask)
            hidden = residual + gate_a * self.dropout(hidden_a)

        residual = hidden
        hidden_f = modulate(self.norm2(hidden), shift_f, scale_f)
        hidden_f = self.ffn(hidden_f)
        hidden = residual + gate_f * self.dropout(hidden_f)

        hidden = apply_mask(hidden, valid_mask)
        stats.update(prefixed_stats("adaln_gate_m", masked_tensor_stats(gate_m, valid_mask)))
        if self.use_local_attention:
            stats.update(prefixed_stats("adaln_gate_a", masked_tensor_stats(gate_a, valid_mask)))
            stats.update(prefixed_stats("adaln_scale_a", masked_tensor_stats(scale_a, valid_mask)))
        stats.update(prefixed_stats("adaln_gate_f", masked_tensor_stats(gate_f, valid_mask)))
        stats.update(prefixed_stats("adaln_scale_m", masked_tensor_stats(scale_m, valid_mask)))
        stats.update(prefixed_stats("adaln_scale_f", masked_tensor_stats(scale_f, valid_mask)))
        return hidden, stats


class AdaLNConditionalMambaDenoiser(nn.Module):
    def __init__(self, args, num_blocks, mode="adaln_only"):
        super().__init__()
        self.latest_stats = {}
        self.mode = mode
        self.decoder_mode = mode
        self.use_input_coupling = mode in {
            "tcond_input_adaln",
            "tcond_input_adaln_localattn_last",
            "tcond_input_adaln_localattn_all",
        }
        self.local_attention_mode = {
            "adaln_only": "none",
            "tcond_input_adaln": "none",
            "tcond_input_adaln_localattn_last": "last",
            "tcond_input_adaln_localattn_all": "all",
        }[mode]
        self.active_components = "mamba_state+adaln_modulation"
        if self.local_attention_mode != "none":
            self.active_components += f"+local_attention_{self.local_attention_mode}"
        self.cond_proj = nn.Linear(args.hidden_size, args.hidden_size)
        self.time_proj = nn.Linear(args.hidden_size, args.hidden_size)
        self.input_proj = nn.Linear(args.hidden_size, args.hidden_size)
        self.local_attn_window = getattr(args, "local_attn_window", 5)
        self.local_attn_heads = getattr(args, "local_attn_heads", 2)
        self.local_attn_dim = getattr(args, "local_attn_dim", 0) or None
        self.layers = nn.ModuleList()
        for layer_index in range(num_blocks):
            use_local_attention = self.local_attention_mode == "all" or (
                self.local_attention_mode == "last" and layer_index == num_blocks - 1
            )
            self.layers.append(
                AdaLNConditionalMambaBlock(
                    hidden_size=args.hidden_size,
                    dropout=args.dropout,
                    alpha_max=args.tcond_gate_alpha_max,
                    use_input_coupling=self.use_input_coupling,
                    use_local_attention=use_local_attention,
                    local_attn_window=self.local_attn_window,
                    local_attn_heads=self.local_attn_heads,
                    local_attn_dim=self.local_attn_dim,
                    causal=getattr(args, "is_causal", True),
                    d_state=args.mamba_d_state,
                    d_conv=args.mamba_d_conv,
                    expand=args.mamba_expand,
                )
            )

        nn.init.normal_(self.cond_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.cond_proj.bias)
        nn.init.normal_(self.time_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.time_proj.bias)

    def forward(self, hidden_input, rep_item, valid_mask, time_emb):
        hidden = apply_mask(self.input_proj(hidden_input), valid_mask)
        cond_ctx = apply_mask(self.cond_proj(rep_item) + self.time_proj(time_emb), valid_mask)
        per_key_values = {}
        block_stats = {}

        for index, layer in enumerate(self.layers):
            hidden, layer_stats = layer(hidden, cond_ctx, valid_mask)
            for key, value in layer_stats.items():
                per_key_values.setdefault(key, []).append(value)
                block_stats[f"block{index}_{key}"] = float(value.item())

        self.latest_stats = {
            "adaln_mode": self.mode,
            "adaln_active_components": self.active_components,
            "adaln_local_attention_mode": self.local_attention_mode,
            "adaln_local_attention_window": self.local_attn_window,
            "adaln_local_attention_heads": self.local_attn_heads,
            "adaln_local_attention_dim": self.input_proj.out_features if self.local_attn_dim is None else self.local_attn_dim,
        }
        self.latest_stats.update(block_stats)
        self.latest_stats.update(
            {
                key: float(torch.stack(values).mean().item())
                for key, values in per_key_values.items()
            }
        )
        return hidden

    def get_latest_stats(self):
        return dict(self.latest_stats)


class StatePreservingConditionalMambaBlock(nn.Module):
    def __init__(self, hidden_size, dropout, alpha_max=0.1, mode="gated", d_state=16, d_conv=4, expand=2):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "state-preserving conditional Mamba decoders require the optional package 'mamba_ssm'."
            )

        self.mode = mode
        self.alpha_max = alpha_max
        self.norm_state = nn.LayerNorm(hidden_size)
        self.mamba = Mamba(
            d_model=hidden_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm_refine = nn.LayerNorm(hidden_size)
        self.cond_to_refine = nn.Linear(hidden_size, hidden_size)
        self.time_to_refine = nn.Linear(hidden_size, hidden_size)
        self.refine_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.gate_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.cond_to_refine.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.cond_to_refine.bias)
        nn.init.normal_(self.time_to_refine.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.time_to_refine.bias)
        nn.init.zeros_(self.gate_proj[-1].weight)
        nn.init.constant_(self.gate_proj[-1].bias, -4.0)

    def forward(self, hidden, cond_stream, time_emb, valid_mask=None):
        hidden = apply_mask(hidden, valid_mask)
        stats = {}

        residual = hidden
        state_hidden = self.norm_state(hidden)
        state_hidden = self.mamba(state_hidden)
        hidden = residual + self.dropout(state_hidden)

        residual = hidden
        cond_hidden = self.cond_to_refine(cond_stream)
        time_hidden = self.time_to_refine(time_emb)
        refine_hidden = self.norm_refine(hidden) + cond_hidden + time_hidden
        delta = self.refine_ffn(refine_hidden)
        stats.update(prefixed_stats("spc_delta", masked_tensor_stats(delta, valid_mask)))

        if self.mode == "nogate":
            hidden = residual + self.dropout(delta)
        else:
            gate = self.alpha_max * torch.sigmoid(self.gate_proj(refine_hidden))
            hidden = residual + gate * self.dropout(delta)
            stats.update(prefixed_stats("spc_gate", masked_tensor_stats(gate, valid_mask)))

        hidden = apply_mask(hidden, valid_mask)
        return hidden, stats


class StatePreservingConditionalMambaDenoiser(nn.Module):
    def __init__(self, args, num_blocks, mode="gated"):
        super().__init__()
        self.latest_stats = {}
        self.mode = mode
        self.decoder_mode = mode
        self.active_components = "plain_mamba_state+conditional_refinement"
        self.state_proj = nn.Linear(args.hidden_size, args.hidden_size)
        self.cond_proj = nn.Linear(args.hidden_size, args.hidden_size)
        self.layers = nn.ModuleList(
            [
                StatePreservingConditionalMambaBlock(
                    hidden_size=args.hidden_size,
                    dropout=args.dropout,
                    alpha_max=args.tcond_gate_alpha_max,
                    mode=mode,
                    d_state=args.mamba_d_state,
                    d_conv=args.mamba_d_conv,
                    expand=args.mamba_expand,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, state_input, cond_input, valid_mask, time_emb):
        hidden = apply_mask(self.state_proj(state_input), valid_mask)
        cond_stream = apply_mask(self.cond_proj(cond_input), valid_mask)
        stats_buffer = {}

        for layer in self.layers:
            hidden, layer_stats = layer(hidden, cond_stream, time_emb, valid_mask)
            for key, value in layer_stats.items():
                stats_buffer.setdefault(key, []).append(value)

        self.latest_stats = {
            "spc_mode": self.mode,
            "spc_active_components": self.active_components,
        }
        self.latest_stats.update(
            {
                key: float(torch.stack(values).mean().item())
                for key, values in stats_buffer.items()
            }
        )
        return hidden

    def get_latest_stats(self):
        return dict(self.latest_stats)


class TimestepConditionedMambaBlock(nn.Module):
    def __init__(self, hidden_size, dropout, alpha_max=0.1, placement="full", d_state=16, d_conv=4, expand=2):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "timestep-conditioned Mamba decoders require the optional package 'mamba_ssm'."
            )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.mamba = Mamba(
            d_model=hidden_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.time_to_ssm = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 3),
        )
        self.time_to_ffn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 3),
        )
        self.time_to_input = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2),
        )
        self.alpha_max = alpha_max
        self.placement = placement
        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.time_to_ssm[-1].weight)
        nn.init.zeros_(self.time_to_ssm[-1].bias)
        nn.init.zeros_(self.time_to_ffn[-1].weight)
        nn.init.zeros_(self.time_to_ffn[-1].bias)
        nn.init.zeros_(self.time_to_input[-1].weight)
        nn.init.zeros_(self.time_to_input[-1].bias)

    def forward(self, hidden, time_emb, valid_mask=None):
        hidden = apply_mask(hidden, valid_mask)
        stats = {}

        if self.placement == "input":
            input_shift, input_scale = self.time_to_input(time_emb).chunk(2, dim=-1)
            hidden = modulate(hidden, input_shift, input_scale)
            hidden = apply_mask(hidden, valid_mask)
            stats.update(prefixed_stats("tcond_input_shift", masked_tensor_stats(input_shift, valid_mask)))
            stats.update(prefixed_stats("tcond_input_scale", masked_tensor_stats(input_scale, valid_mask)))

        residual = hidden
        ssm_hidden = self.norm1(hidden)
        if self.placement in ("full", "ssm"):
            ssm_shift, ssm_scale, ssm_gate_raw = self.time_to_ssm(time_emb).chunk(3, dim=-1)
            ssm_gate = 1.0 + self.alpha_max * torch.tanh(ssm_gate_raw)
            ssm_hidden = modulate(ssm_hidden, ssm_shift, ssm_scale)
            ssm_hidden = self.mamba(ssm_hidden)
            hidden = residual + ssm_gate * self.dropout(ssm_hidden)
            stats.update(prefixed_stats("tcond_ssm_gate", masked_tensor_stats(ssm_gate, valid_mask)))
        else:
            ssm_hidden = self.mamba(ssm_hidden)
            hidden = residual + self.dropout(ssm_hidden)

        residual = hidden
        ffn_hidden = self.norm2(hidden)
        if self.placement in ("full", "ffn"):
            ffn_shift, ffn_scale, ffn_gate_raw = self.time_to_ffn(time_emb).chunk(3, dim=-1)
            ffn_gate = 1.0 + self.alpha_max * torch.tanh(ffn_gate_raw)
            ffn_hidden = modulate(ffn_hidden, ffn_shift, ffn_scale)
            ffn_hidden = self.ffn(ffn_hidden)
            hidden = residual + ffn_gate * self.dropout(ffn_hidden)
            stats.update(prefixed_stats("tcond_ffn_gate", masked_tensor_stats(ffn_gate, valid_mask)))
        else:
            ffn_hidden = self.ffn(ffn_hidden)
            hidden = residual + self.dropout(ffn_hidden)

        hidden = apply_mask(hidden, valid_mask)
        return hidden, stats


class TimestepConditionedMambaDenoiser(nn.Module):
    def __init__(self, args, num_blocks, placement="full"):
        super().__init__()
        self.latest_stats = {}
        self.placement = placement
        self.layers = nn.ModuleList(
            [
                TimestepConditionedMambaBlock(
                    hidden_size=args.hidden_size,
                    dropout=args.dropout,
                    alpha_max=args.tcond_gate_alpha_max,
                    placement=placement,
                    d_state=args.mamba_d_state,
                    d_conv=args.mamba_d_conv,
                    expand=args.mamba_expand,
                )
                for _ in range(num_blocks)
            ]
        )
        self.active_branches = {
            "full": "ssm,ffn",
            "ssm": "ssm_only",
            "ffn": "ffn_only",
            "input": "input_only",
        }[placement]

    def forward(self, hidden, valid_mask, time_emb):
        stats_buffer = {}
        for layer in self.layers:
            hidden, layer_stats = layer(hidden, time_emb, valid_mask)
            for key, value in layer_stats.items():
                stats_buffer.setdefault(key, []).append(value)

        self.latest_stats = {
            "tcond_placement": self.placement,
            "tcond_active_branches": self.active_branches,
        }
        self.latest_stats.update(
            {
                key: float(torch.stack(values).mean().item())
                for key, values in stats_buffer.items()
            }
        )
        return hidden

    def get_latest_stats(self):
        return dict(self.latest_stats)
