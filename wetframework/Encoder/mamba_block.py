import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba  # 官方库
except:
    raise ImportError("Please install mamba-ssm: pip install mamba-ssm")


class MambaBlock(nn.Module):
    """
    Mamba Block
    -----------------------------------------------------------
    Wrapper structure aligned with the simplified figure:

        LN → Mamba → DropPath → Residual

    Notes:
        - The official `Mamba` module already wraps the core Mamba block,
          including the internal selective SSM and local convolution-related operations.
        - This wrapper only keeps the outer pre-norm, stochastic depth, and residual connection.

    Input:
        x : [B, N, C]
    Output:
        x : [B, N, C]
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        """
        x: [B, N, C]
        """
        shortcut = x

        x = self.norm(x)
        x = self.mamba(x)
        x = shortcut + self.drop_path(x)

        return x


class DropPath(nn.Module):
    """Standard DropPath (Stochastic Depth)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob if drop_prob is not None else 0.0

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()

        return x / keep_prob * random_tensor
