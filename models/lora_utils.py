"""
LoRA insertion utilities for segmentation models.
Supports encoder-only, attention-block, and bottleneck adaptation strategies.
Rank sweep: r = 2, 4, 8, 16.
"""

import torch
import torch.nn as nn
import math
from typing import List, Optional

_HEAD_PATTERNS = (
    "final_conv", 
    "mask_decoder", 
    "simple_decoder", 
    "prompt",
    "dec",          
    "final_up"    
)

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation for a Linear layer.
    W' = W + B * A
    W: frozen pretrained weight (d x k)
    B: trainable (d x r)
    A: trainable (r x k)
    """

    def __init__(self, in_features: int, out_features: int, r: int = 4,
                 lora_alpha: float = 1.0, lora_dropout: float = 0.0,
                 merge_weights: bool = False):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.r            = r
        self.lora_alpha   = lora_alpha
        self.scaling      = lora_alpha / r
        self.merged       = False
        self.merge_weights = merge_weights

        # Frozen base weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias   = nn.Parameter(torch.zeros(out_features), requires_grad=False)

        # Trainable low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return nn.functional.linear(x, self.weight, self.bias)

        base_out = nn.functional.linear(x, self.weight, self.bias)
        lora_out = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

    def merge(self):
        if not self.merged:
            # W' = W + (B @ A) * scaling
            A_weight = self.lora_A  # (r, in_features)
            B_weight = self.lora_B  # (out_features, r)
            
            # Compute the low-rank delta
            delta = (B_weight @ A_weight) * self.scaling
            
            # Reshape delta to match the base Conv2D/Linear weight shape
            delta = delta.view(self.weight.shape)
            
            self.weight.data += delta
            self.merged = True

    def unmerge(self):
        if self.merged:
            A_weight = self.lora_A
            B_weight = self.lora_B
            delta = (B_weight @ A_weight) * self.scaling
            delta = delta.view(self.weight.shape)
            self.weight.data -= delta
            self.merged = False

    @classmethod
    def from_linear(cls, linear: nn.Linear, r: int = 4, lora_alpha: float = 1.0,
                    lora_dropout: float = 0.0) -> "LoRALinear":
        """Replace an existing nn.Linear with a LoRALinear."""
        lora_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        lora_linear.weight.data = linear.weight.data.clone()
        if linear.bias is not None:
            lora_linear.bias.data = linear.bias.data.clone()
        return lora_linear


class LoRAConv2d(nn.Module):
    """
    Low-Rank Adaptation for Conv2d layers (bottleneck adaptation).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, groups: int = 1, r: int = 4,
                 lora_alpha: float = 1.0, lora_dropout: float = 0.0):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.groups       = groups
        self.r            = r
        self.scaling      = lora_alpha / r
        self.merged       = False

        # Properly scale the in_channels by groups for depthwise compatibility
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size),
            requires_grad=False
        )
        self.bias = None

        # Low-rank decomposition via 1x1 convolutions
        self.lora_A = nn.Conv2d(in_channels, r, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(r, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout2d(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass the captured groups and bias to the base convolution
        base_out = nn.functional.conv2d(
            x, self.weight, bias=self.bias, stride=self.stride, 
            padding=self.padding, groups=self.groups
        )
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return base_out + lora_out

    def merge(self):
        """Mathematically merge the low-rank A and B convolutions into the base weight."""
        if not self.merged:
            # weight_A: (r, in_channels, 1, 1)
            # weight_B: (out_channels, r, kernel_size, kernel_size)
            weight_A = self.lora_A.weight.data.squeeze(3).squeeze(2) # (r, in_channels)
            weight_B = self.lora_B.weight.data                       # (out_channels, r, k, k)
            
            # Einsum contraction: sum over the rank dimension
            delta = torch.einsum('o r h w, r i -> o i h w', weight_B, weight_A)
            
            self.weight.data += delta * self.scaling
            self.merged = True

    def unmerge(self):
        if self.merged:
            weight_A = self.lora_A.weight.data.squeeze(3).squeeze(2)
            weight_B = self.lora_B.weight.data
            
            delta = torch.einsum('o r h w, r i -> o i h w', weight_B, weight_A)
            
            self.weight.data -= delta * self.scaling
            self.merged = False

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d, r: int = 4,
                    lora_alpha: float = 1.0, lora_dropout: float = 0.0) -> "LoRAConv2d":
        lora_conv = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size[0],
            stride=conv.stride[0],
            padding=conv.padding[0],
            groups=conv.groups,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        lora_conv.weight.data = conv.weight.data.clone()
        
        # Preserve the original bias if it exists
        if conv.bias is not None:
            lora_conv.bias = nn.Parameter(conv.bias.data.clone(), requires_grad=False)
            
        return lora_conv


# ---------------------------------------------------------------------------
# Insertion helpers
# ---------------------------------------------------------------------------

INSERTION_STRATEGIES = ["encoder_only", "attention_blocks", "bottleneck"]


def inject_lora_into_model(
    model: nn.Module,
    r: int = 4,
    lora_alpha: float = 1.0,
    lora_dropout: float = 0.0,
    strategy: str = "attention_blocks",
    target_module_names: Optional[List[str]] = None,
) -> nn.Module:
    
    assert strategy in INSERTION_STRATEGIES, \
        f"strategy must be one of {INSERTION_STRATEGIES}"

    # Freeze ALL parameters first
    for param in model.parameters():
        param.requires_grad = False

    replacements = 0

    for name, module in model.named_modules():
        should_inject = _should_inject(name, module, strategy, target_module_names)

        if should_inject and isinstance(module, nn.Linear):
            lora_layer = LoRALinear.from_linear(module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            _replace_module(model, name, lora_layer)
            replacements += 1

        elif should_inject and isinstance(module, nn.Conv2d) and strategy in ("bottleneck", "encoder_only"):
            lora_layer = LoRAConv2d.from_conv2d(module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            _replace_module(model, name, lora_layer)
            replacements += 1

    # Unfreeze only the explicit segmentation head layers and prompts.
    for name, param in model.named_parameters():
        if any(pat in name for pat in _HEAD_PATTERNS):
            param.requires_grad = True
            
    # [CRITICAL FIX: UNFREEZE NORMALIZATION FOR DOMAIN ADAPTATION]
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.requires_grad = True
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.requires_grad = True
    
    print(f"[LoRA] Injected {replacements} LoRA layers (strategy={strategy}, r={r})")
    _print_trainable_parameters(model)
    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge all LoRA weights into base weights for inference (zero overhead)."""
    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            if hasattr(module, "merge"):
                try:
                    module.merge()
                except NotImplementedError:
                    pass  # Gracefully skip Conv2d merging
    print("[LoRA] All eligible LoRA weights merged into base weights.")
    return model


def _should_inject(name: str, module: nn.Module, strategy: str,
                   target_names: Optional[List[str]]) -> bool:
    # Guard against ALL grouped convolutions (groups > 1) to prevent merge() crashes
    if isinstance(module, nn.Conv2d) and module.groups > 1:
        return False

    if target_names is not None:
        return any(t in name for t in target_names)

    if strategy == "encoder_only":
        return "enc" in name and isinstance(module, (nn.Linear, nn.Conv2d))

    if strategy == "attention_blocks":
        # Removed "proj" to prevent accidental injection into backbone convolutions
        attention_keywords = [
            "q_proj", "v_proj", "query", "value",
            "attn.qkv", "self_attn", "qkv",
        ]
        return any(kw in name for kw in attention_keywords)

    if strategy == "bottleneck":
        # "bottleneck" targets the fallback EdgeEncoder's self.bottleneck
        # "neck" targets the official RepViT's self.neck
        return any(k in name for k in ["bottleneck", "neck"])

    return False

def _replace_module(model: nn.Module, name: str, new_module: nn.Module):
    """Safely replace a module, handling both named attributes and sequential indices."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
            
    attr = parts[-1]
    if attr.isdigit():
        parent[int(attr)] = new_module
    else:
        setattr(parent, attr, new_module)


def _print_trainable_parameters(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    pct       = 100 * trainable / total if total > 0 else 0
    print(f"[LoRA] Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Return only LoRA trainable parameters for optimizer."""
    return [p for p in model.parameters() if p.requires_grad]