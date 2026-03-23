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


def masked_gate_stats(gate, valid_mask):
    if valid_mask is None:
        flat_gate = gate.reshape(-1)
    else:
        valid_positions = valid_mask > 0
        if valid_positions.any():
            flat_gate = gate[valid_positions]
        else:
            flat_gate = gate.new_zeros(1)

    return {
        "mean": flat_gate.mean().detach(),
        "std": flat_gate.std(unbiased=False).detach(),
        "min": flat_gate.min().detach(),
        "max": flat_gate.max().detach(),
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
        if valid_mask is not None:
            hidden = apply_mask(hidden, valid_mask)

        residual = hidden
        hidden = self.norm1(hidden)
        hidden = self.mamba(hidden)
        hidden = residual + self.dropout(hidden)

        residual = hidden
        hidden = self.norm2(hidden)
        hidden = self.ffn(hidden)
        hidden = residual + self.dropout(hidden)

        if valid_mask is not None:
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
    def __init__(self, hidden_size, dropout, alpha_max=0.1, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "dif_decoder='mamba_tcond' requires the optional package 'mamba_ssm'."
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
        self.alpha_max = alpha_max
        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.time_to_ssm[-1].weight)
        nn.init.zeros_(self.time_to_ssm[-1].bias)
        nn.init.zeros_(self.time_to_ffn[-1].weight)
        nn.init.zeros_(self.time_to_ffn[-1].bias)

    def forward(self, hidden, time_emb, valid_mask=None):
        hidden = apply_mask(hidden, valid_mask)

        residual = hidden
        ssm_shift, ssm_scale, ssm_gate_raw = self.time_to_ssm(time_emb).chunk(3, dim=-1)
        ssm_gate = 1.0 + self.alpha_max * torch.tanh(ssm_gate_raw)
        hidden = modulate(self.norm1(hidden), ssm_shift, ssm_scale)
        hidden = self.mamba(hidden)
        hidden = residual + ssm_gate * self.dropout(hidden)

        residual = hidden
        ffn_shift, ffn_scale, ffn_gate_raw = self.time_to_ffn(time_emb).chunk(3, dim=-1)
        ffn_gate = 1.0 + self.alpha_max * torch.tanh(ffn_gate_raw)
        hidden = modulate(self.norm2(hidden), ffn_shift, ffn_scale)
        hidden = self.ffn(hidden)
        hidden = residual + ffn_gate * self.dropout(hidden)

        hidden = apply_mask(hidden, valid_mask)
        ssm_stats = masked_gate_stats(ssm_gate, valid_mask)
        ffn_stats = masked_gate_stats(ffn_gate, valid_mask)
        stats = {
            "tcond_ssm_gate_mean": ssm_stats["mean"],
            "tcond_ffn_gate_mean": ffn_stats["mean"],
            "tcond_ssm_gate_std": ssm_stats["std"],
            "tcond_ffn_gate_std": ffn_stats["std"],
            "tcond_ssm_gate_min": ssm_stats["min"],
            "tcond_ffn_gate_min": ffn_stats["min"],
            "tcond_ssm_gate_max": ssm_stats["max"],
            "tcond_ffn_gate_max": ffn_stats["max"],
        }
        return hidden, stats


class TimestepConditionedMambaDenoiser(nn.Module):
    def __init__(self, args, num_blocks):
        super().__init__()
        self.latest_stats = {}
        self.layers = nn.ModuleList(
            [
                TimestepConditionedMambaBlock(
                    hidden_size=args.hidden_size,
                    dropout=args.dropout,
                    alpha_max=args.tcond_gate_alpha_max,
                    d_state=args.mamba_d_state,
                    d_conv=args.mamba_d_conv,
                    expand=args.mamba_expand,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, hidden, valid_mask, time_emb):
        stats_buffer = {
            "tcond_ssm_gate_mean": [],
            "tcond_ffn_gate_mean": [],
            "tcond_ssm_gate_std": [],
            "tcond_ffn_gate_std": [],
            "tcond_ssm_gate_min": [],
            "tcond_ffn_gate_min": [],
            "tcond_ssm_gate_max": [],
            "tcond_ffn_gate_max": [],
        }
        for layer in self.layers:
            hidden, layer_stats = layer(hidden, time_emb, valid_mask)
            for key, value in layer_stats.items():
                stats_buffer[key].append(value)
        self.latest_stats = {
            key: float(torch.stack(values).mean().item())
            for key, values in stats_buffer.items()
        }
        return hidden

    def get_latest_stats(self):
        return dict(self.latest_stats)
