"""
U-Net surrogate for the 2D heat-equation workload.

The model learns:

    initial temperature field -> final temperature field

Input shape:
    [B, 1, 64, 64]

Output shape:
    [B, 1, 64, 64]

We use GroupNorm rather than BatchNorm so normalization is less
dependent on the local per-GPU batch size during later DDP scaling
experiments.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Convolution block
# ---------------------------------------------------------------------

class DoubleConv(nn.Module):
    """
    Two consecutive 3x3 convolution blocks.

    Each block performs:

        Conv2d
            ->
        GroupNorm
            ->
        ReLU

    Padding=1 keeps the spatial resolution unchanged:

        H x W -> H x W
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 8,
    ) -> None:

        super().__init__()


        # GroupNorm requires the channel count to be divisible
        # by the number of groups.
        if out_channels % num_groups != 0:
            raise ValueError(
                f"out_channels={out_channels} must be "
                f"divisible by num_groups={num_groups}."
            )


        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=out_channels,
            ),

            nn.ReLU(
                inplace=True
            ),


            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=out_channels,
            ),

            nn.ReLU(
                inplace=True
            ),
        )


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.block(x)


# ---------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------

class HeatUNet(nn.Module):
    """
    Three-level U-Net for 64x64 scalar temperature fields.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
    ) -> None:

        super().__init__()


        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8


        # =============================================================
        # Encoder
        # =============================================================

        self.encoder1 = DoubleConv(
            in_channels,
            c1,
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )


        self.encoder2 = DoubleConv(
            c1,
            c2,
        )

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )


        self.encoder3 = DoubleConv(
            c2,
            c3,
        )

        self.pool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )


        # =============================================================
        # Bottleneck
        # =============================================================

        self.bottleneck = DoubleConv(
            c3,
            c4,
        )


        # =============================================================
        # Decoder
        #
        # ConvTranspose2d doubles spatial resolution.
        #
        # Example:
        #
        #     [B, 512, 8, 8]
        #
        # becomes:
        #
        #     [B, 256, 16, 16]
        # =============================================================

        self.up3 = nn.ConvTranspose2d(
            in_channels=c4,
            out_channels=c3,
            kernel_size=2,
            stride=2,
        )

        # After concatenation:
        #
        # decoder: 256 channels
        # encoder: 256 channels
        #
        # total = 512 channels
        self.decoder3 = DoubleConv(
            c3 + c3,
            c3,
        )


        self.up2 = nn.ConvTranspose2d(
            in_channels=c3,
            out_channels=c2,
            kernel_size=2,
            stride=2,
        )

        self.decoder2 = DoubleConv(
            c2 + c2,
            c2,
        )


        self.up1 = nn.ConvTranspose2d(
            in_channels=c2,
            out_channels=c1,
            kernel_size=2,
            stride=2,
        )

        self.decoder1 = DoubleConv(
            c1 + c1,
            c1,
        )


        # =============================================================
        # Final projection
        #
        # 1x1 convolution changes only the number of channels.
        #
        # [B, 64, 64, 64]
        #
        #        ->
        #
        # [B, 1, 64, 64]
        # =============================================================

        self.output = nn.Conv2d(
            in_channels=c1,
            out_channels=out_channels,
            kernel_size=1,
        )


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # -------------------------------------------------------------
        # Encoder
        # -------------------------------------------------------------

        e1 = self.encoder1(x)

        e2 = self.encoder2(
            self.pool1(e1)
        )

        e3 = self.encoder3(
            self.pool2(e2)
        )


        # -------------------------------------------------------------
        # Bottleneck
        # -------------------------------------------------------------

        b = self.bottleneck(
            self.pool3(e3)
        )


        # -------------------------------------------------------------
        # Decoder stage 3
        # -------------------------------------------------------------

        d3 = self.up3(b)

        d3 = torch.cat(
            [
                d3,
                e3,
            ],
            dim=1,
        )

        d3 = self.decoder3(d3)


        # -------------------------------------------------------------
        # Decoder stage 2
        # -------------------------------------------------------------

        d2 = self.up2(d3)

        d2 = torch.cat(
            [
                d2,
                e2,
            ],
            dim=1,
        )

        d2 = self.decoder2(d2)


        # -------------------------------------------------------------
        # Decoder stage 1
        # -------------------------------------------------------------

        d1 = self.up1(d2)

        d1 = torch.cat(
            [
                d1,
                e1,
            ],
            dim=1,
        )

        d1 = self.decoder1(d1)


        prediction = self.output(d1)

        return prediction