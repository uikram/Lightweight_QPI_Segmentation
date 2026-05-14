"""
Physics-Aware Phase Consistency Loss for QPI segmentation.

L_total = L_Dice + λ1 * L_PMC + λ2 * L_BGA + λ3 * L_PV

Components:
    L_Dice  – Standard Dice loss for segmentation accuracy
    L_PMC   – Phase-Mask Contrast: cell phase > background phase
    L_BGA   – Boundary-Gradient Alignment: seg boundary ↔ phase gradient
    L_PV    – Phase-Volume Preservation: optical volume consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice loss for binary and multi-class segmentation.
    L_Dice = 1 - (2 * |P ∩ G|) / (|P| + |G|)
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred:   (B, 1, H, W) logits or probabilities
        target: (B, H, W) binary long tensor
        """
        pred_prob = torch.sigmoid(pred).squeeze(1)  # (B, H, W)
        target_f  = target.float()

        intersection = (pred_prob * target_f).sum(dim=[1, 2])
        union        = pred_prob.sum(dim=[1, 2]) + target_f.sum(dim=[1, 2])

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class PhaseMaskContrast(nn.Module):
    """
    Phase-Mask Contrast Loss.
    Enforces that predicted cell regions have higher phase values
    than the surrounding background.

    L_PMC = max(0, μ_background - μ_cell + margin)
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(self, pred: torch.Tensor, phase_map: torch.Tensor) -> torch.Tensor:
        """
        pred:      (B, 1, H, W) logits
        phase_map: (B, 1, H, W) raw phase values (radians, unnormalized)
        """
        pred_mask = torch.sigmoid(pred).squeeze(1)  # (B, H, W)
        phase     = phase_map.squeeze(1)              # (B, H, W)

        eps = 1e-8
        mu_cell = (pred_mask * phase).sum(dim=[1, 2]) / (pred_mask.sum(dim=[1, 2]) + eps)
        mu_bg   = ((1 - pred_mask) * phase).sum(dim=[1, 2]) / \
                  ((1 - pred_mask).sum(dim=[1, 2]) + eps)

        loss = F.relu(mu_bg - mu_cell + self.margin)
        return loss.mean()


class BoundaryGradientAlignment(nn.Module):
    """
    Boundary-Gradient Alignment Loss.
    Aligns segmentation boundary with strong phase gradients.

    L_BGA = ||∇(Mask) - ∇(Phase)||_1
    """

    def __init__(self):
        super().__init__()

    def _spatial_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial gradient magnitude.
        x: (B, 1, H, W)
        Returns: (B, 1, H, W) gradient magnitude
        """
        # Sobel-like finite differences
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]   # (B, 1, H, W-1)
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]   # (B, 1, H-1, W)

        # Pad to original size
        dx = F.pad(dx, (0, 1), mode="replicate")  # (B, 1, H, W)
        dy = F.pad(dy, (0, 0, 0, 1), mode="replicate")  # (B, 1, H, W)

        return torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)

    def forward(self, pred: torch.Tensor, phase_map: torch.Tensor) -> torch.Tensor:
        """
        pred:      (B, 1, H, W) logits
        phase_map: (B, 1, H, W) raw phase values
        """
        pred_prob = torch.sigmoid(pred)  # (B, 1, H, W)

        grad_mask  = self._spatial_gradient(pred_prob)
        grad_phase = self._spatial_gradient(phase_map)

        # Normalize phase gradient to [0, 1] for scale invariance
        grad_phase_max = grad_phase.amax(dim=[2, 3], keepdim=True).clamp(min=1e-8)
        grad_phase_norm = grad_phase / grad_phase_max

        return F.l1_loss(grad_mask, grad_phase_norm)


class PhaseVolumePreservation(nn.Module):
    """
    Phase-Volume Preservation Loss.
    Ensures the phase-integrated optical quantity (dry mass proxy)
    is preserved between predicted and ground-truth masks.

    L_PV = |Σ_{pred}(Phase) - Σ_{GT}(Phase)|
    """

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                phase_map: torch.Tensor) -> torch.Tensor:
        """
        pred:      (B, 1, H, W) logits
        target:    (B, H, W) binary long tensor
        phase_map: (B, 1, H, W) raw phase values
        """
        pred_prob = torch.sigmoid(pred).squeeze(1)  # (B, H, W)
        target_f  = target.float()
        phase     = phase_map.squeeze(1)             # (B, H, W)

        pred_vol = (pred_prob * phase).sum(dim=[1, 2])
        gt_vol   = (target_f  * phase).sum(dim=[1, 2])

        if self.normalize:
            # Normalize by GT volume for scale invariance
            gt_vol_norm = gt_vol.abs().clamp(min=1e-8)
            loss = (torch.abs(pred_vol - gt_vol) / gt_vol_norm).mean()
        else:
            loss = torch.abs(pred_vol - gt_vol).mean()

        return loss


class PhysicsAwarePhaseLoss(nn.Module):
    """
    Combined Physics-Aware Phase Consistency Loss.

    L_total = L_Dice + λ1 * L_PMC + λ2 * L_BGA + λ3 * L_PV

    Also includes optional Binary Cross-Entropy for training stability.
    """

    def __init__(self,
                 lambda1: float = 0.1,   # PMC weight
                 lambda2: float = 0.05,  # BGA weight
                 lambda3: float = 0.1,   # PV weight
                 bce_weight: float = 0.5,
                 pmc_margin: float = 0.1,
                 pv_normalize: bool = True):
        super().__init__()
        self.lambda1    = lambda1
        self.lambda2    = lambda2
        self.lambda3    = lambda3
        self.bce_weight = bce_weight

        self.dice_loss  = DiceLoss()
        self.pmc_loss   = PhaseMaskContrast(margin=pmc_margin)
        self.bga_loss   = BoundaryGradientAlignment()
        self.pv_loss    = PhaseVolumePreservation(normalize=pv_normalize)
        self.bce_loss   = nn.BCEWithLogitsLoss()

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                phase_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:      (B, 1, H, W) raw logits from segmentation model
            target:    (B, H, W) binary ground-truth mask (long)
            phase_map: (B, 1, H, W) raw (unnormalized) phase values

        Returns:
            Scalar loss tensor.
        """
        target_f = target.float().unsqueeze(1)  # (B, 1, H, W)

        L_dice = self.dice_loss(pred, target)
        L_bce  = self.bce_loss(pred, target_f)
        L_pmc  = self.pmc_loss(pred, phase_map)
        L_bga  = self.bga_loss(pred, phase_map)
        L_pv   = self.pv_loss(pred, target, phase_map)

        L_seg   = (1 - self.bce_weight) * L_dice + self.bce_weight * L_bce
        L_total = L_seg + self.lambda1 * L_pmc + self.lambda2 * L_bga + self.lambda3 * L_pv

        return L_total

    def get_loss_components(self,
                            pred: torch.Tensor,
                            target: torch.Tensor,
                            phase_map: torch.Tensor) -> dict:
        """Return individual loss components for logging/ablation."""
        target_f = target.float().unsqueeze(1)

        with torch.no_grad():
            L_dice = self.dice_loss(pred, target).item()
            L_bce  = self.bce_loss(pred, target_f).item()
            L_pmc  = self.pmc_loss(pred, phase_map).item()
            L_bga  = self.bga_loss(pred, phase_map).item()
            L_pv   = self.pv_loss(pred, target, phase_map).item()

        return {
            "L_dice": L_dice,
            "L_bce":  L_bce,
            "L_pmc":  L_pmc,
            "L_bga":  L_bga,
            "L_pv":   L_pv,
            "L_total": (
                (1 - self.bce_weight) * L_dice +
                self.bce_weight * L_bce +
                self.lambda1 * L_pmc +
                self.lambda2 * L_bga +
                self.lambda3 * L_pv
            ),
        }


class DiceOnlyLoss(nn.Module):
    """
    Ablation baseline: Dice + BCE only, no physics terms.
    Used in ablation study (Section 4.5).
    """

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_loss  = DiceLoss()
        self.bce_loss   = nn.BCEWithLogitsLoss()

    def forward(self, pred, target, phase_map=None):
        target_f = target.float().unsqueeze(1)
        L_dice   = self.dice_loss(pred, target)
        L_bce    = self.bce_loss(pred, target_f)
        return (1 - self.bce_weight) * L_dice + self.bce_weight * L_bce


def get_loss(config) -> nn.Module:
    """Loss factory based on config."""
    loss_type = getattr(config, "loss_type", "physics_aware")

    if loss_type == "physics_aware":
        return PhysicsAwarePhaseLoss(
            lambda1=getattr(config, "lambda_pmc", 0.1),
            lambda2=getattr(config, "lambda_bga", 0.05),
            lambda3=getattr(config, "lambda_pv",  0.1),
            bce_weight=getattr(config, "bce_weight", 0.5),
            pmc_margin=getattr(config, "pmc_margin", 0.1),
        )
    elif loss_type == "dice_only":
        return DiceOnlyLoss(bce_weight=getattr(config, "bce_weight", 0.5))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. "
                         f"Choose from: physics_aware, dice_only")