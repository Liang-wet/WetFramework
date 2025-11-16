import torch
import torch.nn as nn

from .patch_embed import OverlapPatchEmbedding
from .transformer_block import TransformerBlock
from .mamba_block import MambaBlock
from .tdam import TDAM


class WFEncoder(nn.Module):
    """
    WetFramework Encoder (TMHM + TDAM)
    ----------------------------------------------------------
    Fully aligned with the architecture shown in Fig.1

    Output list:
        features[0] → Stage 1 (1/4 resolution)
        features[1] → Stage 2 (1/8 resolution)
        features[2] → Stage 3 (1/16 resolution)
        features[3] → Stage 4 (1/32 resolution)
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=[64, 128, 320, 512],
                 depths=[3, 4, 6, 3],
                 num_heads=[1, 2, 5, 8],
                 drop_path_rate=0.1):

        super().__init__()

        self.num_stages = 4

        # -----------------------------------------------------
        # DropPath allocation: total blocks = transformer + mamba
        # -----------------------------------------------------
        total_blocks = sum(depths) * 2
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        dp_idx = 0

        # -----------------------------------------------------
        # Stage 1
        # -----------------------------------------------------
        self.patch_embed1 = OverlapPatchEmbedding(
            in_channels, embed_dims[0], patch_size=7, stride=4, padding=3
        )

        self.trans1 = nn.ModuleList([
            TransformerBlock(embed_dims[0], num_heads[0],
                             drop_path=dpr[dp_idx + i])
            for i in range(depths[0])
        ])
        dp_idx += depths[0]

        self.mamba1 = nn.ModuleList([
            MambaBlock(embed_dims[0], drop_path=dpr[dp_idx + i])
            for i in range(depths[0])
        ])
        dp_idx += depths[0]

        self.tdam1 = TDAM(embed_dims[0], drop_path=dpr[dp_idx - 1])

        # -----------------------------------------------------
        # Stage 2
        # -----------------------------------------------------
        self.patch_embed2 = OverlapPatchEmbedding(
            embed_dims[0], embed_dims[1], patch_size=3, stride=2, padding=1
        )

        self.trans2 = nn.ModuleList([
            TransformerBlock(embed_dims[1], num_heads[1],
                             drop_path=dpr[dp_idx + i])
            for i in range(depths[1])
        ])
        dp_idx += depths[1]

        self.mamba2 = nn.ModuleList([
            MambaBlock(embed_dims[1], drop_path=dpr[dp_idx + i])
            for i in range(depths[1])
        ])
        dp_idx += depths[1]

        self.tdam2 = TDAM(embed_dims[1], drop_path=dpr[dp_idx - 1])

        # -----------------------------------------------------
        # Stage 3
        # -----------------------------------------------------
        self.patch_embed3 = OverlapPatchEmbedding(
            embed_dims[1], embed_dims[2], patch_size=3, stride=2, padding=1
        )

        self.trans3 = nn.ModuleList([
            TransformerBlock(embed_dims[2], num_heads[2],
                             drop_path=dpr[dp_idx + i])
            for i in range(depths[2])
        ])
        dp_idx += depths[2]

        self.mamba3 = nn.ModuleList([
            MambaBlock(embed_dims[2], drop_path=dpr[dp_idx + i])
            for i in range(depths[2])
        ])
        dp_idx += depths[2]

        self.tdam3 = TDAM(embed_dims[2], drop_path=dpr[dp_idx - 1])

        # -----------------------------------------------------
        # Stage 4
        # -----------------------------------------------------
        self.patch_embed4 = OverlapPatchEmbedding(
            embed_dims[2], embed_dims[3], patch_size=3, stride=2, padding=1
        )

        self.trans4 = nn.ModuleList([
            TransformerBlock(embed_dims[3], num_heads[3],
                             drop_path=dpr[dp_idx + i])
            for i in range(depths[3])
        ])
        dp_idx += depths[3]

        self.mamba4 = nn.ModuleList([
            MambaBlock(embed_dims[3], drop_path=dpr[dp_idx + i])
            for i in range(depths[3])
        ])
        dp_idx += depths[3]

        self.tdam4 = TDAM(embed_dims[3], drop_path=dpr[dp_idx - 1])


    # --------------------------------------------------------
    # Single stage forward pass (Transformer + Mamba + TDAM)
    # --------------------------------------------------------
    def forward_stage(self, x, patch_embed, trans_blocks, mamba_blocks, tdam):
        # Patch embed
        x, H, W = patch_embed(x)    # [B, N, C]

        # Transformer path
        tx = x
        for blk in trans_blocks:
            tx = blk(tx)

        # Mamba path
        mx = x
        for blk in mamba_blocks:
            mx = blk(mx)

        # Feature fusion: tx + mx
        fused_tokens = tx + mx

        # TDAM enhancement
        fused_tokens = tdam(fused_tokens)

        # Restore to feature map
        B = fused_tokens.shape[0]
        fused_tokens = fused_tokens.transpose(1, 2).reshape(B, -1, H, W)

        return fused_tokens


    # --------------------------------------------------------
    # Full forward pass across all 4 stages
    # --------------------------------------------------------
    def forward(self, x):
        features = []

        # Stage 1
        x = self.forward_stage(
            x, self.patch_embed1, self.trans1, self.mamba1, self.tdam1
        )
        features.append(x)

        # Stage 2
        x = self.forward_stage(
            x, self.patch_embed2, self.trans2, self.mamba2, self.tdam2
        )
        features.append(x)

        # Stage 3
        x = self.forward_stage(
            x, self.patch_embed3, self.trans3, self.mamba3, self.tdam3
        )
        features.append(x)

        # Stage 4
        x = self.forward_stage(
            x, self.patch_embed4, self.trans4, self.mamba4, self.tdam4
        )
        features.append(x)

        return features
