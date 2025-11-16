import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba  # 官方库
except:
    raise ImportError("Please install mamba-ssm: pip install mamba-ssm")


class MambaBlock(nn.Module):
    """
    Mamba Block for WetFramework
    -----------------------------------------------------------
    Matches the design in Fig. 2(b) of the paper:

        LN → Mamba → SiLU → Linear → DropPath → Residual

    Input:
        x : [B, N, C]
    Output:
        x : [B, N, C]
    """

    def __init__(self, dim, drop_path=0., dropout=0.):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        # Mamba SSM module (official)
        self.mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4
        )

        self.act = nn.SiLU()

        # Output projection (same dimension)
        self.proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        """
        x: [B, N, C]
        """
        shortcut = x

        # Pre-normalization
        x = self.norm(x)

        # Mamba SSM
        x = self.mamba(x)  # [B, N, C]

        # Nonlinearity
        x = self.act(x)

        # Projection + dropout
        x = self.proj(x)
        x = self.dropout(x)

        # Residual
        x = shortcut + self.drop_path(x)

        return x


class DropPath(nn.Module):
    """ Standard DropPath (Stochastic Depth) """

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()

        return x / keep_prob * random_tensor
