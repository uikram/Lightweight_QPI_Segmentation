"""
Physics-preserving augmentations for Quantitative Phase Images (QPI).

IMPORTANT: Only geometric transforms are allowed.
The following are PROHIBITED because they destroy quantitative phase information:
    - Intensity jitter / color jitter
    - Gaussian noise on phase values
    - Random erasing
    - Contrast / brightness changes
    - Normalization beyond numerical stability

Allowed:
    - Random horizontal flip
    - Random vertical flip
    - Random rotation (multiples of 90° preserve cell morphology)
    - Random translation (crop + pad)
    - Random zoom (conservative, ±10%)
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, Optional


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, phase: torch.Tensor,
                 mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            phase = torch.flip(phase, dims=[-1])
            mask  = torch.flip(mask,  dims=[-1])
        return phase, mask


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, phase: torch.Tensor,
                 mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            phase = torch.flip(phase, dims=[-2])
            mask  = torch.flip(mask,  dims=[-2])
        return phase, mask


class RandomRotation90:
    """
    Rotate by multiples of 90 degrees.
    90° rotations preserve RBC morphology and phase statistics exactly.
    """

    def __init__(self, p: float = 0.75):
        self.p = p

    def __call__(self, phase: torch.Tensor,
                 mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            k = random.choice([1, 2, 3])  # 90, 180, 270
            phase = torch.rot90(phase, k=k, dims=[-2, -1])
            mask  = torch.rot90(mask,  k=k, dims=[-2, -1])
        return phase, mask


class RandomTranslation:
    """
    Random translation via padding and cropping.
    Preserves all phase values (no interpolation artifacts).
    """

    def __init__(self, max_shift: float = 0.1, p: float = 0.5):
        """
        max_shift: Maximum shift as fraction of image size.
        """
        self.max_shift = max_shift
        self.p         = p

    def __call__(self, phase: torch.Tensor,
                 mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > self.p:
            return phase, mask

        H, W = phase.shape[-2], phase.shape[-1]
        max_dy = int(H * self.max_shift)
        max_dx = int(W * self.max_shift)

        dy = random.randint(-max_dy, max_dy)
        dx = random.randint(-max_dx, max_dx)

        # Pad then crop to maintain size
        pad_top    = max(0, dy)
        pad_bottom = max(0, -dy)
        pad_left   = max(0, dx)
        pad_right  = max(0, -dx)

        phase = F.pad(phase, (pad_left, pad_right, pad_top, pad_bottom),
                      mode="constant", value=0.0)
        mask  = F.pad(mask,  (pad_left, pad_right, pad_top, pad_bottom),
                      mode="constant", value=0)

        # Crop back to original size
        start_y = pad_bottom
        start_x = pad_right
        phase = phase[..., start_y:start_y + H, start_x:start_x + W]
        mask  = mask[...,  start_y:start_y + H, start_x:start_x + W]

        return phase, mask

class PhaseNormalization:
    """
    Normalize phase map for numerical stability ONLY.
    Does NOT alter the physical meaning of phase values.
    Subtracts mean, divides by std — per-image.
    """
    def __init__(self):
        pass # No global stats needed

    def __call__(self, phase: torch.Tensor) -> torch.Tensor:
        # Per-image normalization for numerical stability
        return (phase - phase.mean()) / phase.std().clamp(min=1e-8)

class QPIAugmentation:
    # Removed `zoom` parameter
    def __init__(self, flip_h: bool = True, flip_v: bool = True,
                 rotate: bool = True, translate: bool = True,
                 normalize: bool = True):
        self.transforms = []

        if flip_h:
            self.transforms.append(RandomHorizontalFlip(p=0.5))
        if flip_v:
            self.transforms.append(RandomVerticalFlip(p=0.5))
        if rotate:
            self.transforms.append(RandomRotation90(p=0.75))
        if translate:
            self.transforms.append(RandomTranslation(max_shift=0.1, p=0.5))

        self.normalizer = PhaseNormalization() if normalize else None

    def __call__(self, phase: torch.Tensor,
                 mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            phase, mask = t(phase, mask)
        
        phase_raw = phase.clone() 
        
        if self.normalizer is not None:
            phase = self.normalizer(phase)
        return phase, mask, phase_raw

class QPIValTransform:
    def __init__(self):
        self.normalizer = PhaseNormalization()

    def __call__(self, phase: torch.Tensor,
                 mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase_raw = phase.clone()
        return self.normalizer(phase), mask, phase_raw