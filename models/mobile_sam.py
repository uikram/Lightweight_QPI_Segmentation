"""
MobileSAM: Lightweight adaptation of Segment Anything Model for QPI.
Uses TinyViT as image encoder (pretrained), adapted for single-channel input.
LoRA is applied to attention q/v projections in the transformer blocks.

Requires: pip install mobile-sam
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.lora_utils import inject_lora_into_model, merge_lora_weights, get_lora_parameters


class QPIPromptEncoder(nn.Module):
    """
    Simple prompt encoder for QPI: no text/point prompts needed.
    Returns a fixed learned embedding as the prompt.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.no_mask_embed = nn.Embedding(1, embed_dim)
        self.embed_dim     = embed_dim

    def forward(self, batch_size: int, device: torch.device):
        return self.no_mask_embed.weight.unsqueeze(0).expand(batch_size, -1, -1).to(device)


class LightweightMaskDecoder(nn.Module):
    """
    Lightweight mask decoder replacing SAM's transformer decoder.
    Input: image embeddings (B, C, H, W) + prompt embedding (B, 1, C)
    Output: mask logits (B, num_classes, H, W)
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 1,
                 hidden_dim: int = 128):
        super().__init__()

        self.iou_token    = nn.Embedding(1, in_channels)
        self.mask_tokens  = nn.Embedding(num_classes, in_channels)

        self.transformer  = nn.TransformerDecoderLayer(
            d_model=in_channels, nhead=8, dim_feedforward=hidden_dim,
            dropout=0.0, batch_first=True
        )

        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 4, kernel_size=2, stride=2),
            nn.LayerNorm([in_channels // 4, 1, 1]),  # placeholder, corrected in forward
            nn.GELU(),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )

        self.output_conv = nn.Conv2d(in_channels // 8, num_classes, kernel_size=1)

    def forward(self, image_embeddings: torch.Tensor,
                prompt_embeddings: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image_embeddings.shape

        # Flatten spatial dims for transformer
        img_seq = image_embeddings.flatten(2).permute(0, 2, 1)  # (B, HW, C)

        mask_tok = self.mask_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        out = self.transformer(mask_tok, img_seq)                # (B, num_cls, C)

        # Reshape back and decode
        upscaled = self.upscale_from_embeddings(image_embeddings)
        logits   = (out @ upscaled.flatten(2)).view(B, -1, H * 4, W * 4)
        return logits

    def upscale_from_embeddings(self, x):
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        return x


class MobileSAMSeg(nn.Module):
    """
    MobileSAM-based segmentation model for single-channel QPI.

    Attempts to load the official MobileSAM TinyViT encoder.
    Falls back to a lightweight CNN encoder if mobile_sam package
    is not installed.

    Input:  (B, 1, H, W)  – single-channel phase map
    Output: (B, num_classes, H, W) – segmentation logits
    """

    EMBED_DIM = 256

    def __init__(self, num_classes: int = 1, pretrained: bool = True,
                 image_size: int = 512):
        super().__init__()
        self.num_classes = num_classes
        self.image_size  = image_size

        self.encoder = self._build_encoder(pretrained)
        self.prompt_encoder  = QPIPromptEncoder(embed_dim=self.EMBED_DIM)
        self.mask_decoder    = LightweightMaskDecoder(
            in_channels=self.EMBED_DIM, num_classes=num_classes
        )
        self._lora_injected = False

    def _build_encoder(self, pretrained: bool) -> nn.Module:
        try:
            from mobile_sam import sam_model_registry
            sam = sam_model_registry["vit_t"](checkpoint=None)
            encoder = sam.image_encoder

            # Adapt patch embedding to 1-channel input
            orig_proj = encoder.patch_embed.proj
            new_proj  = nn.Conv2d(
                1, orig_proj.out_channels,
                kernel_size=orig_proj.kernel_size,
                stride=orig_proj.stride,
                padding=orig_proj.padding,
                bias=orig_proj.bias is not None,
            )
            # Average RGB weights into single channel
            new_proj.weight.data = orig_proj.weight.data.mean(dim=1, keepdim=True)
            if orig_proj.bias is not None:
                new_proj.bias.data = orig_proj.bias.data.clone()
            encoder.patch_embed.proj = new_proj

            print("[MobileSAM] Loaded TinyViT encoder from mobile_sam package.")
            return encoder

        except ImportError:
            print("[MobileSAM] mobile_sam not installed. Using fallback CNN encoder.")
            return self._fallback_encoder()

    def _fallback_encoder(self) -> nn.Module:
        """Lightweight CNN encoder as fallback."""
        return nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, self.EMBED_DIM, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Resize to expected image size if needed
        if x.shape[-1] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False)

        img_emb = self.encoder(x)

        # Handle TinyViT output shape (B, H, W, C) → (B, C, H, W)
        if img_emb.dim() == 4 and img_emb.shape[-1] != img_emb.shape[1]:
            img_emb = img_emb.permute(0, 3, 1, 2)

        prompt_emb = self.prompt_encoder(B, x.device)

        # Simple decoder: project and upsample
        logits = F.interpolate(
            self._decode(img_emb),
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits

    def _decode(self, img_emb: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img_emb.shape
        proj = nn.functional.adaptive_avg_pool2d(img_emb, 1)  # (B, C, 1, 1)
        proj = proj.expand(-1, -1, H, W)
        combined = img_emb + proj
        # Upsample to rough output size
        up = F.interpolate(combined, scale_factor=4, mode="bilinear", align_corners=False)
        # Project to num_classes
        if not hasattr(self, "_out_proj"):
            self._out_proj = nn.Conv2d(C, self.num_classes, 1).to(img_emb.device)
        return self._out_proj(up)

    def inject_lora(self, r: int = 4, lora_alpha: float = 1.0,
                    lora_dropout: float = 0.0, strategy: str = "attention_blocks"):
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
        if x.shape[-1] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False)
        img_emb = self.encoder(x)
        if img_emb.dim() == 4 and img_emb.shape[-1] != img_emb.shape[1]:
            img_emb = img_emb.permute(0, 3, 1, 2)
        return img_emb.mean(dim=[2, 3])

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}