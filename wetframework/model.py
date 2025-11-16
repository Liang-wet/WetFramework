import torch
import torch.nn as nn

from wetframework.Encoder.encoder import WFEncoder
from wetframework.Decoder.decoder import WFDecoder


class WetFramework(nn.Module):
    """
    WetFramework: Unified Encoder-Decoder Architecture
    ---------------------------------------------------------------
    This module integrates:
        - WFEncoder (Transformer + Mamba + TDAM)
        - WFDecoder (Multi-scale fusion + WERM)

    It produces the final segmentation / wetland boundary output.

    Inputs:
        x : [B, C, H, W]   (e.g., Landsat/Sentinel image)

    Output:
        pred : [B, out_channels, H, W]
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=(64, 128, 320, 512),
                 depths=(3, 4, 6, 3),
                 num_heads=(1, 2, 5, 8),
                 out_channels=1,
                 drop_path_rate=0.1):
        super().__init__()

        # ----------------------------------------
        # 1) Encoder
        # ----------------------------------------
        self.encoder = WFEncoder(
            in_channels=in_channels,
            embed_dims=embed_dims,
            depths=depths,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate
        )

        # ----------------------------------------
        # 2) Decoder
        # ----------------------------------------
        self.decoder = WFDecoder(
            embed_dims=embed_dims,
            decoder_dim=256,
            out_channels=out_channels,
            drop_path_rate=drop_path_rate
        )

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: prediction map [B, out_channels, H, W]
        """
        features = self.encoder(x)
        pred = self.decoder(features)
        return pred

# ---------------------------------------------------------------
# WetFramework Quick Forward Test
# ---------------------------------------------------------------
# This test script verifies that the entire WetFramework pipeline
# (Encoder → Decoder → WERM → Output) runs successfully in a
# forward pass without errors.
#
# We simulate a remote-sensing input image tensor with:
#     Batch size = 4
#     Channels   = 3   (e.g., RGB / multispectral subset)
#     Resolution = 512 × 512
#
# The goal is to check:
#   1) Whether the encoder–decoder architecture is correctly wired.
#   2) Whether all custom modules (Transformer, Mamba, TDAM, WERM)
#      produce consistent shapes.
#   3) Whether the model produces the expected output shape.
#
# This test does NOT evaluate accuracy — it is only a sanity check
# to ensure the implementation is fully functional.
# ---------------------------------------------------------------

if __name__ == "__main__":
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(4, 3, 512, 512).to(device)

    model = WetFramework(
        in_channels=3,
        out_channels=1
    ).to(device)

    with torch.no_grad():
        y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
    print("\nModel forward test SUCCESS!")


