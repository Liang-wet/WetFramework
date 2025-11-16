import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic Depth per sample (token-wise)."""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor


class TDAM(nn.Module):
    """
    Token-Driven Attention Module (TDAM)
    --------------------------------------
     Matches paper Fig.2(c)

    Input:
        x : [B, N, C]  (already fused = tx + mx)

    Output:
        out : [B, N, C]

    Structure:
        token-attention  -> softmax over tokens
        channel-gating   -> sigmoid over channels
        fusion           -> weighted combination
        residual + LN
    """

    def __init__(self, dim, hidden_ratio=2, drop_path=0.0):
        super().__init__()

        hidden_dim = int(dim * hidden_ratio)

        # token importance MLP
        self.token_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        # channel importance MLP
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

        self.norm = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        """
        x : [B, N, C]
        (This x = tx + mx from encoder fusion)
        """
        shortcut = x

        # -----------------------------
        # 1. Token-wise attention
        # -----------------------------
        token_scores = self.token_mlp(x)          # [B, N, 1]
        token_attn = F.softmax(token_scores, dim=1)

        # -----------------------------
        # 2. Channel-wise gating
        # -----------------------------
        channel_scores = self.channel_mlp(x)      # [B, N, C]
        channel_gate = torch.sigmoid(channel_scores.mean(dim=1, keepdim=True))  # [B,1,C]

        # -----------------------------
        # 3. Combine
        # -----------------------------
        fused = x * token_attn + x * channel_gate

        # -----------------------------
        # 4. Residual + LN + DropPath
        # -----------------------------
        out = self.norm(shortcut + self.drop_path(fused))

        return out
