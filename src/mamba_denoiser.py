import torch
import torch.nn as nn


try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


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
            hidden = hidden * valid_mask.unsqueeze(-1)

        residual = hidden
        hidden = self.norm1(hidden)
        hidden = self.mamba(hidden)
        hidden = residual + self.dropout(hidden)

        residual = hidden
        hidden = self.norm2(hidden)
        hidden = self.ffn(hidden)
        hidden = residual + self.dropout(hidden)

        if valid_mask is not None:
            hidden = hidden * valid_mask.unsqueeze(-1)

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
