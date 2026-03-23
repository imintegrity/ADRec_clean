import torch
import torch.nn as nn


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
