"""
Physics-Aware Phase Consistency Loss for QPI segmentation.

Multi-class biology + binary physics bridge:

    L_total = L_CE + λ1 * L_PMC + λ2 * L_BGA + λ3 * L_PV

    L_CE  – CrossEntropyLoss over 5 classes (0=bg, 1-4=cell types)
    L_PMC – Phase-Mask Contrast: cell phase > background phase
    L_BGA – Boundary-Gradient Alignment: seg boundary ↔ phase gradient
    L_PV  – Phase-Volume Preservation: optical volume consistency

Physics losses receive foreground *probability* (already in [0,1]) derived
from the multi-class softmax output, so they skip the internal sigmoid step
via the apply_sigmoid=False path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# 5-class weights: [background, discocyte, echinocyte, spherocyte, stomatocyte]
# Higher weight for rarer / clinically important degradation classes.
# Indices match dataset pixel values: 0=bg, 1=disco, 2=echino, 3=sphero, 4=stomato
DEFAULT_CLASS_WEIGHTS = [0.5, 1.0, 1.5, 2.0, 2.0]


# ---------------------------------------------------------------------------
class PhaseMaskContrast(nn.Module):
    """
    Phase-Mask Contrast Loss.
    Enforces that predicted cell regions have higher phase values
    than the surrounding background.

    L_PMC = max(0, μ_background - μ_cell + margin)

    apply_sigmoid: set False when pred is already a probability map [0,1].
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(self, pred: torch.Tensor, phase_map: torch.Tensor,
                apply_sigmoid: bool = True) -> torch.Tensor:
        """
        pred:      (B, 1, H, W) logits  OR  foreground probabilities [0,1]
        phase_map: (B, 1, H, W) raw phase values (radians, unnormalized)
        """
        pred_mask = torch.sigmoid(pred).squeeze(1) if apply_sigmoid else pred.squeeze(1)
        phase     = phase_map.squeeze(1)

        eps    = 1e-8
        mu_cell = (pred_mask * phase).sum(dim=[1, 2]) / (pred_mask.sum(dim=[1, 2]) + eps)
        mu_bg   = ((1 - pred_mask) * phase).sum(dim=[1, 2]) / \
                  ((1 - pred_mask).sum(dim=[1, 2]) + eps)

        loss = F.relu(mu_bg - mu_cell + self.margin)
        return loss  # Shape is already (B,)

class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes=5, ignore_index=None, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # Register weight as a buffer so it automatically moves to the correct device
        self.register_buffer('weight', weight)

    def forward(self, pred, target):
        pred_soft = torch.softmax(pred, dim=1)
        loss = 0.0
        weight_sum = 0.0
        for c in range(self.num_classes):
            if self.ignore_index is not None and c == self.ignore_index:
                continue
            p = pred_soft[:, c]
            g = (target == c).float()
            intersection = (p * g).sum()
            c_loss = 1 - (2 * intersection + 1e-8) / (p.sum() + g.sum() + 1e-8)
            
            w = self.weight[c] if self.weight is not None else 1.0
            loss += w * c_loss
            weight_sum += w
        return loss / (weight_sum + 1e-8)

class BoundaryGradientAlignment(nn.Module):
    """
    Boundary-Gradient Alignment Loss.
    Aligns segmentation boundary with strong phase gradients.

    L_BGA = ||∇(Mask) - ∇(Phase)||_1

    apply_sigmoid: set False when pred is already a probability map [0,1].
    """

    def __init__(self):
        super().__init__()

    def _spatial_gradient(self, x: torch.Tensor) -> torch.Tensor:
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        dx = F.pad(dx, (0, 1), mode="constant", value=0.0)
        dy = F.pad(dy, (0, 0, 0, 1), mode="constant", value=0.0)
        return torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)

    def forward(self, pred: torch.Tensor, phase_map: torch.Tensor,
                apply_sigmoid: bool = True) -> torch.Tensor:
        """
        pred:      (B, 1, H, W) logits  OR  foreground probabilities [0,1]
        phase_map: (B, 1, H, W) raw phase values
        """
        pred_prob = torch.sigmoid(pred) if apply_sigmoid else pred

        grad_mask  = self._spatial_gradient(pred_prob)
        grad_phase = self._spatial_gradient(phase_map)

        grad_phase_max  = grad_phase.amax(dim=[2, 3], keepdim=True).clamp(min=1e-8)
        grad_phase_norm = grad_phase / grad_phase_max

        return F.l1_loss(grad_mask, grad_phase_norm, reduction='none').mean(dim=[1, 2, 3])


class PhaseVolumePreservation(nn.Module):
    def __init__(self, normalize: bool = True, epsilon: float = 1e-4):
        super().__init__()
        self.normalize = normalize
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                phase_map: torch.Tensor,
                apply_sigmoid: bool = True) -> torch.Tensor:
        
        pred_prob = torch.sigmoid(pred).squeeze(1) if apply_sigmoid else pred.squeeze(1)
        target_f  = target.float()
        phase     = phase_map.squeeze(1)

        pred_vol = (pred_prob * phase).sum(dim=[1, 2])
        gt_vol   = (target_f  * phase).sum(dim=[1, 2])

        if self.normalize:
            gt_vol_norm = gt_vol.abs().clamp(min=0.0) + self.epsilon
            loss = (torch.abs(pred_vol - gt_vol) / gt_vol_norm)
        else:
            loss = torch.abs(pred_vol - gt_vol)

        return loss # Returns tensor of shape (B,)


# ---------------------------------------------------------------------------
class PhysicsAwarePhaseLoss(nn.Module):
    def __init__(self,
                 lambda1: float = 0.1,
                 lambda2: float = 0.05,
                 lambda3: float = 0.1,
                 pmc_margin: float = 0.1,
                 pv_normalize: bool = True,
                 class_weights: Optional[list] = None):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        weights = class_weights if class_weights is not None else DEFAULT_CLASS_WEIGHTS
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        
        # FIX: Set ignore_index=None so the background class weight (0.5) is actually utilized
        self.ce_loss = MultiClassDiceLoss(num_classes=5, ignore_index=None, weight=weight_tensor)

        self.pmc_loss = PhaseMaskContrast(margin=pmc_margin)
        self.bga_loss = BoundaryGradientAlignment()
        self.pv_loss  = PhaseVolumePreservation(normalize=pv_normalize)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, phase_map: torch.Tensor) -> torch.Tensor:
        # 1. Biological classification loss
        L_ce = self.ce_loss(pred, target)

        # 2. Derive foreground probability
        pred_softmax   = torch.softmax(pred, dim=1)
        pred_fg_prob   = 1.0 - pred_softmax[:, 0:1, :, :]  # (B, 1, H, W)
        target_binary  = (target > 0).long()               # (B, H, W)

        # 3. Check which images in the batch actually contain cells
        has_cells = (target_binary.sum(dim=[1, 2]) > 0).float() # (B,)

        # 4. Compute unreduced physics losses
        L_pmc = self.pmc_loss(pred_fg_prob, phase_map, apply_sigmoid=False)
        L_bga = self.bga_loss(pred_fg_prob, phase_map, apply_sigmoid=False)
        L_pv  = self.pv_loss(pred_fg_prob, target_binary, phase_map, apply_sigmoid=False)
        
        # 5. Apply physics losses ONLY if the patch contains cells
        if has_cells.sum() > 0:
            L_pmc = (L_pmc * has_cells).sum() / has_cells.sum()
            L_bga = (L_bga * has_cells).sum() / has_cells.sum()
            L_pv  = (L_pv * has_cells).sum() / has_cells.sum()
        else:
            L_pmc = L_bga = L_pv = 0.0

        return L_ce + self.lambda1 * L_pmc + self.lambda2 * L_bga + self.lambda3 * L_pv

    def get_loss_components(self,
                            pred: torch.Tensor,
                            target: torch.Tensor,
                            phase_map: torch.Tensor) -> dict:
        """Return individual loss component values for logging/ablation."""

        with torch.no_grad():
            pred_softmax = torch.softmax(pred, dim=1)
            pred_fg_prob = 1.0 - pred_softmax[:, 0:1, :, :]
            target_binary = (target > 0).long()

            # L_ce is already a scalar, but the physics terms are (B,) tensors now
            L_ce  = self.ce_loss(pred, target).item()
            L_pmc = self.pmc_loss(pred_fg_prob, phase_map, apply_sigmoid=False).mean().item()
            L_bga = self.bga_loss(pred_fg_prob, phase_map, apply_sigmoid=False).mean().item()
            L_pv  = self.pv_loss(pred_fg_prob, target_binary, phase_map,
                                 apply_sigmoid=False).mean().item()

        return {
            "L_ce":   L_ce,
            "L_pmc":  L_pmc,
            "L_bga":  L_bga,
            "L_pv":   L_pv,
            "L_total": L_ce + self.lambda1 * L_pmc + self.lambda2 * L_bga + self.lambda3 * L_pv,
        }


# ---------------------------------------------------------------------------
class DiceOnlyLoss(nn.Module):
    """
    Ablation baseline: Dice Loss only, no physics terms.
    Ensures a valid apples-to-apples comparison with PhysicsAwarePhaseLoss.
    """
    def __init__(self, class_weights: Optional[list] = None):
        super().__init__()
        weights = class_weights if class_weights is not None else DEFAULT_CLASS_WEIGHTS
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        self.ce_loss = MultiClassDiceLoss(num_classes=5, ignore_index=None, weight=weight_tensor)

    def forward(self, pred, target, phase_map=None):
        return self.ce_loss(pred, target)


# ---------------------------------------------------------------------------
def get_loss(config) -> nn.Module:
    loss_type = getattr(config, "loss_type", "physics_aware")

    if loss_type == "physics_aware":
        return PhysicsAwarePhaseLoss(
            lambda1=getattr(config, "lambda1_pmc", 0.1),
            lambda2=getattr(config, "lambda2_bga", 0.05),
            lambda3=getattr(config, "lambda3_pv",  0.1),
            pmc_margin=getattr(config, "pmc_margin", 0.1),
            class_weights=getattr(config, "class_weights", None),
        )
    elif loss_type == "dice_only":
        return DiceOnlyLoss(class_weights=getattr(config, "class_weights", None))
    else:
        raise ValueError(
            f"Unknown loss type: '{loss_type}'. Choose from: physics_aware, dice_only"
        )
