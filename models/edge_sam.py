"""
EdgeSAM: Edge-optimized segmentation architecture for QPI.
Designed for low-latency deployment on edge devices (Jetson, etc.).
Uses a RepViT or ShuffleNet-based encoder for maximum efficiency.
LoRA applied to bottleneck and attention layers.

Requires: pip install edge-sam  (optional, falls back to efficient CNN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.lora_utils import inject_lora_into_model, merge_lora_weights, get_lora_parameters


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block for edge efficiency."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                            groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.act1 = nn.ReLU6(inplace=True)
        
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.act1(self.bn1(self.dw(x)))
        x = self.act2(self.bn2(self.pw(x)))
        return x


class InvertedResidual(nn.Module):
    """MobileNet-style inverted residual block."""

    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=6):
        super().__init__()
        hidden = int(in_ch * expand_ratio)
        self.use_res = stride == 1 and in_ch == out_ch

        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, hidden, 1, bias=False),
                       nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True)]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1,
                      groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class EdgeEncoder(nn.Module):
    """
    Efficient encoder for edge deployment.
    Single-channel input → hierarchical feature maps.
    """

    def __init__(self):
        super().__init__()
        # Stage 0: initial conv
        self.stage0 = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )
        # Stage 1
        self.stage1 = nn.Sequential(
            InvertedResidual(16, 16, stride=1, expand_ratio=1),
        )
        # Stage 2
        self.stage2 = nn.Sequential(
            InvertedResidual(16, 24, stride=2, expand_ratio=6),
            InvertedResidual(24, 24, stride=1, expand_ratio=6),
        )
        # Stage 3
        self.stage3 = nn.Sequential(
            InvertedResidual(24, 32, stride=2, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
        )
        # Stage 4
        self.stage4 = nn.Sequential(
            InvertedResidual(32, 64, stride=2, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
        )
        # Stage 5 (bottleneck)
        self.bottleneck = nn.Sequential(
            InvertedResidual(64, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 160, stride=2, expand_ratio=6),
            InvertedResidual(160, 160, stride=1, expand_ratio=6),
            InvertedResidual(160, 160, stride=1, expand_ratio=6),
            InvertedResidual(160, 320, stride=1, expand_ratio=6),
        )

        self.out_channels = [16, 24, 32, 64, 320]

    def forward(self, x):
        s0 = self.stage0(x)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.bottleneck(s4)
        return s1, s2, s3, s4, s5


class EdgeDecoder(nn.Module):
    """Feature Pyramid Network-style decoder for edge efficiency."""

    def __init__(self, encoder_channels, num_classes=1):
        super().__init__()
        c1, c2, c3, c4, c5 = encoder_channels

        self.lat5 = nn.Conv2d(c5, 128, 1)
        self.lat4 = nn.Conv2d(c4, 128, 1)
        self.lat3 = nn.Conv2d(c3, 128, 1)
        self.lat2 = nn.Conv2d(c2, 128, 1)

        self.smooth5 = DepthwiseSeparableConv(128, 128)
        self.smooth4 = DepthwiseSeparableConv(128, 128)
        self.smooth3 = DepthwiseSeparableConv(128, 128)
        self.smooth2 = DepthwiseSeparableConv(128, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU6(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU6(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )

    def forward(self, s1, s2, s3, s4, s5):
        p5 = self.smooth5(self.lat5(s5))
        p4 = self.smooth4(self.lat4(s4) + F.interpolate(p5, size=s4.shape[2:],
                                                          mode="nearest"))
        p3 = self.smooth3(self.lat3(s3) + F.interpolate(p4, size=s3.shape[2:],
                                                          mode="nearest"))
        p2 = self.smooth2(self.lat2(s2) + F.interpolate(p3, size=s2.shape[2:],
                                                          mode="nearest"))
        return self.final(p2)


class EdgeSAMSeg(nn.Module):
    """
    EdgeSAM-inspired segmentation model for real-time QPI analysis.
    Optimized for edge deployment: low latency, low memory.

    Input:  (B, 1, H, W)  – single-channel phase map
    Output: (B, num_classes, H, W) – segmentation logits
    """

    def __init__(self, num_classes: int = 1, pretrained: bool = False):
        super().__init__()
        self.num_classes = num_classes

        self.encoder = self._build_encoder(pretrained)

        # Track which encoder path was loaded so forward() can gate the resize correctly
        self.use_sam_encoder = not isinstance(self.encoder, EdgeEncoder)

        if isinstance(self.encoder, EdgeEncoder):
            self.decoder = EdgeDecoder(
                encoder_channels=self.encoder.out_channels,
                num_classes=num_classes,
            )
            self.use_simple_decoder = False
        else:
            bottleneck_dim = self.encoder.out_channels[-1] if hasattr(self.encoder, 'out_channels') else 256
            self.simple_decoder = nn.Sequential(
                nn.ConvTranspose2d(bottleneck_dim, 128, kernel_size=2, stride=2),
                nn.BatchNorm2d(128),
                nn.ReLU6(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU6(inplace=True),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU6(inplace=True),
                nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
                nn.BatchNorm2d(16),
                nn.ReLU6(inplace=True),
                nn.Conv2d(16, self.num_classes, kernel_size=1)
            )
            self.use_simple_decoder = True

    def _build_encoder(self, pretrained: bool):
        try:
            from edge_sam import sam_model_registry
            import os
            
            # FIX 1: Explicitly load the pretrained checkpoint
            ckpt_path = "weights/edge_sam_3x.pth"
            if pretrained and os.path.exists(ckpt_path):
                sam = sam_model_registry["edge_sam"](checkpoint=ckpt_path)
                print(f"[EdgeSAM] Loaded pretrained weights from {ckpt_path}")
            else:
                sam = sam_model_registry["edge_sam"](checkpoint=None)
                print("[WARNING] No checkpoint found. Initializing with RANDOM weights!")

            encoder = sam.image_encoder

            first_conv_name = None
            first_conv_layer = None
            parent_module = encoder
            
            for name, module in encoder.named_modules():
                if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                    first_conv_name = name.split('.')[-1]
                    first_conv_layer = module
                    for part in name.split('.')[:-1]:
                        parent_module = getattr(parent_module, part)
                    break
                    
            if first_conv_layer is None:
                raise AttributeError("Could not find initial 3-channel Conv2d to adapt.")
                
            new_conv = nn.Conv2d(
                1, first_conv_layer.out_channels, 
                first_conv_layer.kernel_size,
                first_conv_layer.stride, 
                first_conv_layer.padding, 
                bias=first_conv_layer.bias is not None
            )
            
            new_conv.weight.data = first_conv_layer.weight.data.mean(dim=1, keepdim=True)
            if first_conv_layer.bias is not None:
                new_conv.bias.data = first_conv_layer.bias.data.clone()
                
            setattr(parent_module, first_conv_name, new_conv)
            
            encoder.out_channels = [64, 128, 256, 512, 256]
            print("[EdgeSAM] Loaded official EdgeSAM encoder.")
            return encoder

        except Exception as e:
            print(f"\n[EdgeSAM] Failed to load official EdgeSAM because: {e}")
            print("[EdgeSAM] Falling back to built-in EdgeEncoder.")
            return EdgeEncoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[2:]

        # Only resize to 1024 if using the official pretrained SAM encoder
        if hasattr(self, 'use_sam_encoder') and self.use_sam_encoder and x.shape[-1] != 1024:
            x_enc = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
        elif not hasattr(self, 'use_sam_encoder') and x.shape[-1] != 1024:
             # Fallback if use_sam_encoder isn't defined
             x_enc = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
        else:
            x_enc = x

        # [DELETED DOUBLE NORMALIZATION HERE]

        skips = self.encoder(x_enc)
        
        if not self.use_simple_decoder:
            logits = self.decoder(*skips)
        else:
            features = skips if not isinstance(skips, tuple) else skips[-1]
            logits = self.simple_decoder(features)

        if logits.shape[2:] != orig_size:
            logits = F.interpolate(logits, size=orig_size, mode="bilinear", align_corners=False)
        return logits

    def inject_lora(self, r: int = 4, lora_alpha: float = 1.0,
                    lora_dropout: float = 0.0, strategy: str = "bottleneck"):
        inject_lora_into_model(self.encoder, r=r, lora_alpha=lora_alpha,
                               lora_dropout=lora_dropout, strategy=strategy)
        self._lora_injected = True
        return self

    def merge_lora(self):
        merge_lora_weights(self.encoder)
        return self

    def get_lora_params(self):
        return get_lora_parameters(self)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Feature extraction for evaluation."""
        skips = self.encoder(x)
        # FIX: Check if tuple vs single tensor
        if isinstance(skips, tuple):
            return skips[-1].mean(dim=[2, 3])
        return skips.mean(dim=[2, 3])

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}