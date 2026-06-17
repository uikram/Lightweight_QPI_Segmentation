"""
MobileNet-UNet: Lightweight encoder-decoder segmentation architecture.
Accepts single-channel quantitative phase images (QPI).
Encoder: MobileNetV2 (pretrained on ImageNet, adapted for 1-channel input).
Decoder: UNet-style skip connections with transposed convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from models.lora_utils import inject_lora_into_model, merge_lora_weights, get_lora_parameters


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """Upsampling block with skip connection."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBNReLU(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch from encoder pooling
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MobileNetUNet(nn.Module):
    """
    MobileNetV2-based UNet for single-channel QPI segmentation.

    Input:  (B, 1, H, W)  – single-channel phase map (float32, radians)
    Output: (B, num_classes, H, W) – segmentation logits
    """

    # MobileNetV2 feature channels at each skip level
    ENCODER_CHANNELS = [16, 24, 32, 96, 1280]

    def __init__(self, num_classes: int = 1, pretrained: bool = True,
                 decoder_channels: list = None):
        super().__init__()
        self.num_classes = num_classes

        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32]

        # ----- Encoder -----
        weights_to_use = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v2(weights=weights_to_use)
        
        # FINAL REFINEMENT: Dynamic class name extraction instead of hardcoding
        print(f"[MobileNetUNet] Using OFFICIAL implementation")
        print(f"[MobileNetUNet] Source: torchvision.models package")
        print(f"[MobileNetUNet] Model class: {backbone.__class__.__name__}")
        print(f"[MobileNetUNet] Checkpoint: {'IMAGENET1K_V1' if pretrained else 'None (Random Weights)'}")

        # Adapt first conv to accept 1-channel input
        # Average the 3-channel pretrained weights across channels
        orig_conv    = backbone.features[0][0]
        new_conv     = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        new_conv.weight.data = orig_conv.weight.data.mean(dim=1, keepdim=True)
        backbone.features[0][0] = new_conv

        features = backbone.features

        # Skip connection extraction points (MobileNetV2 inverted residual stages)
        self.enc0 = features[0]           # 32 ch, stride 2
        self.enc1 = features[1]           # 16 ch
        self.enc2 = features[2:4]         # 24 ch, stride 2
        self.enc3 = features[4:7]         # 32 ch, stride 2
        self.enc4 = features[7:14]        # 96 ch, stride 2
        self.enc5 = features[14:]         # 1280 ch, stride 2

        self.enc2 = nn.Sequential(*self.enc2)
        self.enc3 = nn.Sequential(*self.enc3)
        self.enc4 = nn.Sequential(*self.enc4)
        self.enc5 = nn.Sequential(*self.enc5)

        # ----- Decoder -----
        self.dec4 = DecoderBlock(1280, 96,  decoder_channels[0])
        self.dec3 = DecoderBlock(decoder_channels[0], 32, decoder_channels[1])
        self.dec2 = DecoderBlock(decoder_channels[1], 24, decoder_channels[2])
        self.dec1 = DecoderBlock(decoder_channels[2], 16, decoder_channels[3])

        # Final upsampling to input resolution
        self.final_up   = nn.ConvTranspose2d(decoder_channels[3], decoder_channels[3],
                                             kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)

        self._lora_injected = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e0 = self.enc0(x)   # /2
        e1 = self.enc1(e0)  # /2
        e2 = self.enc2(e1)  # /4
        e3 = self.enc3(e2)  # /8
        e4 = self.enc4(e3)  # /16
        e5 = self.enc5(e4)  # /32

        # Decoder with skip connections
        d4 = self.dec4(e5, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        out = self.final_up(d1)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.final_conv(out)

    def inject_lora(self, r: int = 4, lora_alpha: float = 1.0,
                    lora_dropout: float = 0.0, strategy: str = "encoder_only"):
        inject_lora_into_model(self, r=r, lora_alpha=lora_alpha,
                               lora_dropout=lora_dropout, strategy=strategy)
        # Forcefully unfreeze all decoder, upsampling, and final projection layers
        for name, param in self.named_parameters():
            if any(substring in name for substring in ["dec", "up", "final"]):
                param.requires_grad = True
        self._lora_injected = True
        return self

    def merge_lora(self):
        merge_lora_weights(self)
        return self

    def get_lora_params(self):
        return get_lora_parameters(self)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Feature extraction for evaluation compatibility."""
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        return e5.mean(dim=[2, 3])  # Global average pool → (B, 1280)

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}