import torch
import torch.nn as nn
import torch.nn.functional as F

from .werm import WERM


class DropPath(nn.Module):

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()

        return x / keep_prob * random_tensor


class WFDecoder(nn.Module):
    """
    WetFramework Decoder (Fig.4(b) in the paper)
    ---------------------------------------------------------------
    Pipeline:

        Multi-scale encoder features:
          c1 (1/4), c2 (1/8), c3 (1/16), c4 (1/32)

            ↓ MLP-like 1×1 projections
            ↓ Upsample all to 1/4
            ↓ Concat
            ↓ Conv fusion
            ↓ WERM (Wavelet-Enhanced Reconstruction Module)
            ↓ DropPath + Conv  (residual refinement)
            ↓ Upsample ×4
            ↓ Output head

    Output:
        out : [B, out_channels, H, W]
    """

    def __init__(self,
                 embed_dims=(64, 128, 320, 512),
                 decoder_dim=256,
                 out_channels=1,
                 drop_path_rate=0.1):

        super().__init__()

        C1, C2, C3, C4 = embed_dims

        # ----------------------------------------------------
        # 1) MLP-like 1×1 projection for each encoder scale
        # ----------------------------------------------------
        self.proj1 = nn.Conv2d(C1, decoder_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(C2, decoder_dim, kernel_size=1)
        self.proj3 = nn.Conv2d(C3, decoder_dim, kernel_size=1)
        self.proj4 = nn.Conv2d(C4, decoder_dim, kernel_size=1)

        # ----------------------------------------------------
        # 2) Concat → Conv fusion (Fig.4(b))
        # ----------------------------------------------------
        self.fuse_conv = nn.Conv2d(
            decoder_dim * 4, decoder_dim, kernel_size=3, padding=1
        )

        # ----------------------------------------------------
        # 3) WERM (wavelet reconstruction block)
        # ----------------------------------------------------
        self.werm = WERM(channels=decoder_dim)

        # ----------------------------------------------------
        # 4) Residual refinement: DropPath + Conv
        # ----------------------------------------------------
        self.drop_path = DropPath(drop_path_rate)
        self.refine_conv = nn.Conv2d(
            decoder_dim, decoder_dim,
            kernel_size=3, padding=1
        )

        # ----------------------------------------------------
        # 5) Final 1×1 conv to output channels
        # ----------------------------------------------------
        self.out_head = nn.Conv2d(
            decoder_dim, out_channels, kernel_size=1
        )

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------
    def forward(self, features):
        """
        features:
            c1: [B, C1, H/4,  W/4]
            c2: [B, C2, H/8,  W/8]
            c3: [B, C3, H/16, W/16]
            c4: [B, C4, H/32, W/32]
        """
        c1, c2, c3, c4 = features

        B, _, H4, W4 = c1.shape   # target resolution = 1/4

        # -----------------------------------------------------
        # 1) 1×1 Conv projection
        # -----------------------------------------------------
        p1 = self.proj1(c1)
        p2 = self.proj2(c2)
        p3 = self.proj3(c3)
        p4 = self.proj4(c4)

        # -----------------------------------------------------
        # 2) Upsample all projected features to 1/4 scale
        # -----------------------------------------------------
        p2 = F.interpolate(p2, size=(H4, W4), mode="bilinear", align_corners=False)
        p3 = F.interpolate(p3, size=(H4, W4), mode="bilinear", align_corners=False)
        p4 = F.interpolate(p4, size=(H4, W4), mode="bilinear", align_corners=False)

        # -----------------------------------------------------
        # 3) Fuse multi-scale features
        # -----------------------------------------------------
        fused = torch.cat([p1, p2, p3, p4], dim=1)  # [B, 4D, H/4, W/4]
        fused = self.fuse_conv(fused)               # [B, D,  H/4, W/4]

        # -----------------------------------------------------
        # 4) WERM (Wavelet Enhancement)
        # -----------------------------------------------------
        x = self.werm(fused)  # [B, D, H/4, W/4]

        # -----------------------------------------------------
        # 5) Residual refinement with correct DropPath usage
        # -----------------------------------------------------
        res = self.refine_conv(x)
        x = x + self.drop_path(res)

        # -----------------------------------------------------
        # 6) Upsample to full resolution and predict
        # -----------------------------------------------------
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        out = self.out_head(x)

        return out
