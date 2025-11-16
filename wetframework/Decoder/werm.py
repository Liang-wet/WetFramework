import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse


class WERM(nn.Module):
    """
    Wavelet-Enhanced Reconstruction Module (WERM)
    ---------------------------------------------------------------
    This module matches Fig.4 in the paper and enhances decoder
    features by injecting multi-scale high-frequency structural cues.

    The module performs:
        1. Single-level DWT decomposition (yl, yh)
        2. Convolutional enhancement over wavelet high-frequency bands
        3. Channel-wise gating
        4. Wavelet inverse reconstruction to obtain refined features

    Inputs:
        x : [B, C, H, W]

    Outputs:
        x_rec : [B, C, H, W]
    """

    def __init__(self, channels, wave='haar'):
        super().__init__()

        # 1-level DWT (Fig.4 step 1)
        self.dwt = DWTForward(J=1, wave=wave)
        self.idwt = DWTInverse(wave=wave)

        # yh shape = [B, C, 3, H/2, W/2]  → reshape to [B, C*3, H/2, W/2]
        self.conv_high = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1)
        )

        # Channel-wise gating (Fig.4 step 3)
        self.gate = nn.Conv2d(channels * 3, channels * 3, kernel_size=1)

    def forward(self, x):
        """
        Input:
            x : [B, C, H, W]
        Output:
            x_rec : [B, C, H, W]
        """

        # ---------------------------------------------
        # 1) Wavelet decomposition
        # ---------------------------------------------
        yl, yh = self.dwt(x)  # yh = list containing [B, C, 3, H/2, W/2]
        high = yh[0]

        B, C, D, Hh, Wh = high.shape  # D = 3 orientations

        # ---------------------------------------------
        # 2) Fuse orientation dimension into channels
        #    high: [B, C, 3, H, W] → [B, C*3, H, W]
        # ---------------------------------------------
        high_reshaped = high.reshape(B, C * D, Hh, Wh)

        # ---------------------------------------------
        # 3) Convolutional enhancement on high-frequency bands
        # ---------------------------------------------
        high_processed = self.conv_high(high_reshaped)

        # ---------------------------------------------
        # 4) Channel-wise gating
        # ---------------------------------------------
        gate = torch.sigmoid(self.gate(high_processed))
        high_final = high_processed * gate

        # ---------------------------------------------
        # 5) Reshape back to wavelet format
        # ---------------------------------------------
        high_final = high_final.reshape(B, C, D, Hh, Wh)

        # ---------------------------------------------
        # 6) Wavelet inverse reconstruction
        # ---------------------------------------------
        x_rec = self.idwt((yl, [high_final]))

        return x_rec
