import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """
    Standard Multi-Head Self Attention (MHSA)
    ------------------------------------------------
    Corresponds to the "Multi-Head Attention" in Fig. 2(a).

    Input:
        x : [B, N, C]

    Output:
        out : [B, N, C]
    """

    def __init__(self, dim, num_heads, attn_dropout=0., proj_dropout=0.):
        super().__init__()
        assert dim % num_heads == 0, "Embedding dim must be divisible by num_heads"

        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x):
        B, N, C = x.shape

        # Compute qkv
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum
        out = attn @ v  # [B, heads, N, C_head]
        out = out.transpose(1, 2).reshape(B, N, C)

        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TransformerBlock(nn.Module):
    """
    Transformer Block for WetFramework
    ------------------------------------------------
    Matches the structure of Fig. 2(a):

        LN → MHSA → DropPath → LN → MLP → DropPath → Residuals

    Inputs:
        x : [B, N, C]

    Outputs:
        x : [B, N, C]
    """

    def __init__(self, dim, num_heads,
                 mlp_ratio=4.0,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads,
                                           attn_dropout=attn_drop,
                                           proj_dropout=drop)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # MLP layer (Feed Forward Network)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop)
        )

        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        # MHSA branch
        x = x + self.drop_path1(self.attn(self.norm1(x)))

        # MLP branch
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x


class DropPath(nn.Module):
    """
    DropPath (Stochastic Depth)
    Standard implementation.
    """

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binary mask

        return x / keep_prob * random_tensor
