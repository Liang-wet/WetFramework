import torch
import torch.nn as nn

class OverlapPatchEmbedding(nn.Module):
    """
    Overlap Patch Embedding Module for WetFramework
    ------------------------------------------------
    This module converts the input image into overlapped patch tokens.
    It corresponds to the 'Overlap Patch Embeddings' component in the
    encoder of WetFramework (see paper Fig. 1, Stage 1).

    Inputs:
        x : [B, C, H, W]

    Outputs:
        tokens : [B, N, C_embed]
        H_out, W_out : spatial resolution after patch embedding
    """

    def __init__(self, in_channels, embed_dim,
                 patch_size=7, stride=4, padding=3):
        super().__init__()

        # Conv2d with overlap (padding allows sliding patches)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )

        # LayerNorm on embedding dimension
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Forward pass of OverlapPatchEmbedding.

        Parameters:
            x: [B, C, H, W]

        Returns:
            x_tokens: [B, H_out*W_out, C_embed]
            H_out, W_out: spatial size after embedding
        """
        B, C, H, W = x.shape

        # Convolutional projection
        x = self.proj(x)       # [B, C_embed, H_out, W_out]
        H_out, W_out = x.shape[2], x.shape[3]

        # Flatten into token format
        x = x.flatten(2).transpose(1, 2)   # [B, N, C_embed]

        # LayerNorm
        x = self.norm(x)

        return x, H_out, W_out
