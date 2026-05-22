"""
Physics-Aware Phase Consistency Loss for QPI segmentation.

Multi-class biology + binary physics bridge:

    L_total = L_Dice + λ1 * L_PMC + λ2 * L_BGA + λ3 * L_PV

    L_Dice – MultiClassDiceLoss over 5 classes (0=bg, 1-4=cell types)
    L_PMC  – Phase-Mask Contrast: cell phase > background phase
    L_BGA  – Boundary-Gradient Alignment: seg boundary ↔ phase gradient
    L_PV   – Phase-Volume Preservation: optical volume consistency
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
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin
        # FIX: Safely initialize tracking variables in __init__
        self._pmc_warn_count = 0

    def forward(self, pred: torch.Tensor, phase_map: torch.Tensor,
                apply_sigmoid: bool = True) -> torch.Tensor:
        
        pred_mask = torch.sigmoid(pred).squeeze(1) if apply_sigmoid else pred.squeeze(1)
        phase     = phase_map.squeeze(1)

        # Diagnostic: throttled warning for degenerate all-foreground collapse.
        if self.training and self._pmc_warn_count < 3:
            fg_ratio = pred_mask.mean().item()
            if fg_ratio > 0.85:
                self._pmc_warn_count += 1
                print(f"\n[Warning] PMC: fg_ratio={fg_ratio:.2%}. "
                      f"Possible degenerate collapse. "
                      f"({self._pmc_warn_count}/3 warnings)")

        cell_area = pred_mask.sum(dim=[1, 2]).clamp(min=1.0)
        bg_area   = (1 - pred_mask).sum(dim=[1, 2]).clamp(min=1.0)

        mu_cell = (pred_mask * phase).sum(dim=[1, 2]) / cell_area
        mu_bg   = ((1 - pred_mask) * phase).sum(dim=[1, 2]) / bg_area

        loss = F.relu(mu_bg - mu_cell + self.margin)
        return loss

class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes=5, ignore_index=None, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
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
            
            # SUPERIOR FIX: Use 1.0 Laplace smoothing. 
            # This is mathematically stable in FP16 and prevents 0/0 NaNs natively.
            smooth = 1.0
            intersection = (p * g).sum(dim=[1, 2])
            c_loss = 1 - (2 * intersection + smooth) / (p.sum(dim=[1, 2]) + g.sum(dim=[1, 2]) + smooth)
            c_loss = c_loss.mean()
            
            w = self.weight[c] if self.weight is not None else 1.0
            loss += w * c_loss
            weight_sum += w
            
        # Safe FP16 epsilon
        return loss / (weight_sum + 1e-4)

class BoundaryGradientAlignment(nn.Module):
    def __init__(self):
        super().__init__()

    def _spatial_gradient(self, x: torch.Tensor) -> torch.Tensor:
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        dx = F.pad(dx, (0, 1), mode="constant", value=0.0)
        dy = F.pad(dy, (0, 0, 0, 1), mode="constant", value=0.0)
        return torch.sqrt(dx ** 2 + dy ** 2 + 1e-4)

    def forward(self, pred: torch.Tensor, phase_map: torch.Tensor,
                apply_sigmoid: bool = True) -> torch.Tensor:
        
        pred_prob = torch.sigmoid(pred) if apply_sigmoid else pred

        grad_mask  = self._spatial_gradient(pred_prob)
        grad_phase = self._spatial_gradient(phase_map)

        # Per-sample max-normalisation to [0, 1] to fix gradient scaling mismatch
        grad_mask_norm  = grad_mask  / (grad_mask.amax(dim=[1, 2, 3],  keepdim=True) + 1e-8)
        grad_phase_norm = grad_phase / (grad_phase.amax(dim=[1, 2, 3], keepdim=True) + 1e-8)

        return F.l1_loss(grad_mask_norm, grad_phase_norm, reduction='none').mean(dim=[1, 2, 3])

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
    """
    Combined Physics-Aware Phase Consistency Loss for multi-class QPI.

    L_total = L_Dice + λ1 * L_PMC + λ2 * L_BGA + λ3 * L_PV

    The MultiClassDiceLoss handles the 5-class biological classification.
    Physics losses (PMC, BGA, PV) operate on foreground probability
    (cell vs background) derived from the softmax output...
    """
    def __init__(self,
                 lambda1: float = 0.1,
                 lambda2: float = 0.05,
                 lambda3: float = 0.1,
                 pmc_margin: float = 0.1,
                 # FIX: Changed to True. Unnormalized volume sums lead to extreme loss values 
                 # (~10k+) triggering FP16 exploding gradients and flat 0.0 Dice scores.
                 pv_normalize: bool = True,  
                 class_weights: Optional[list] = None):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        weights = class_weights if class_weights is not None else DEFAULT_CLASS_WEIGHTS
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        
        self.dice_loss = MultiClassDiceLoss(num_classes=5, ignore_index=None, weight=weight_tensor)

        self.pmc_loss = PhaseMaskContrast(margin=pmc_margin)
        self.bga_loss = BoundaryGradientAlignment()
        self.pv_loss  = PhaseVolumePreservation(normalize=pv_normalize)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, phase_map: torch.Tensor) -> torch.Tensor:
        # 1. Biological classification loss (Dice)
        L_dice = self.dice_loss(pred, target)

        # 2. Derive foreground probability
        pred_softmax   = torch.softmax(pred, dim=1)
        pred_fg_prob   = 1.0 - pred_softmax[:, 0:1, :, :]  
        target_binary  = (target > 0).long()               

        # 3. Check which images in the batch actually contain cells
        has_cells = (target_binary.sum(dim=[1, 2]) > 0).float() 

        # 4. Compute unreduced physics losses
        L_pmc = self.pmc_loss(pred_fg_prob, phase_map, apply_sigmoid=False)
        L_bga = self.bga_loss(pred_fg_prob, phase_map, apply_sigmoid=False)
        L_pv  = self.pv_loss(pred_fg_prob, target_binary, phase_map, apply_sigmoid=False)
        
        # 5. Apply physics losses ONLY if the patch contains cells
        if has_cells.sum() > 0:
            L_pmc = (L_pmc * has_cells).sum() / (has_cells.sum() + 1e-8)
            L_bga = (L_bga * has_cells).sum() / (has_cells.sum() + 1e-8)
            L_pv  = (L_pv * has_cells).sum() / (has_cells.sum() + 1e-8)
        else:
            # Fix: Keep fallback values as tensors on the correct device
            L_pmc = torch.tensor(0.0, device=pred.device)
            L_bga = torch.tensor(0.0, device=pred.device)
            L_pv  = torch.tensor(0.0, device=pred.device)

        return L_dice + self.lambda1 * L_pmc + self.lambda2 * L_bga + self.lambda3 * L_pv

    def get_loss_components(self,
                            pred: torch.Tensor,
                            target: torch.Tensor,
                            phase_map: torch.Tensor) -> dict:
        """Return individual loss component values for logging/ablation."""
        with torch.no_grad():
            pred_softmax = torch.softmax(pred, dim=1)
            pred_fg_prob = 1.0 - pred_softmax[:, 0:1, :, :]
            target_binary = (target > 0).long()
            has_cells = (target_binary.sum(dim=[1, 2]) > 0).float()

            L_dice = self.dice_loss(pred, target).item()
            
            L_pmc = self.pmc_loss(pred_fg_prob, phase_map, apply_sigmoid=False)
            L_bga = self.bga_loss(pred_fg_prob, phase_map, apply_sigmoid=False)
            L_pv  = self.pv_loss(pred_fg_prob, target_binary, phase_map, apply_sigmoid=False)

            if has_cells.sum() > 0:
                L_pmc = ((L_pmc * has_cells).sum() / (has_cells.sum() + 1e-8)).item()
                L_bga = ((L_bga * has_cells).sum() / (has_cells.sum() + 1e-8)).item()
                L_pv  = ((L_pv * has_cells).sum() / (has_cells.sum() + 1e-8)).item()
            else:
                L_pmc = L_bga = L_pv = 0.0

        return {
            "L_dice": L_dice,
            "L_pmc":  L_pmc,
            "L_bga":  L_bga,
            "L_pv":   L_pv,
            "L_total": L_dice + self.lambda1 * L_pmc + self.lambda2 * L_bga + self.lambda3 * L_pv,
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
    
    # Safely extract nested block if it exists
    loss_weights = getattr(config, "loss_weights", {})
    if not isinstance(loss_weights, dict):
        loss_weights = {}

    if loss_type == "physics_aware":
        return PhysicsAwarePhaseLoss(
            lambda1=loss_weights.get("lambda1_pmc", getattr(config, "lambda1_pmc", 0.1)),
            lambda2=loss_weights.get("lambda2_bga", getattr(config, "lambda2_bga", 0.05)),
            lambda3=loss_weights.get("lambda3_pv",  getattr(config, "lambda3_pv", 0.1)),
            pmc_margin=getattr(config, "pmc_margin", 0.1),
            class_weights=getattr(config, "class_weights", None),
        )
    elif loss_type == "dice_only":
        return DiceOnlyLoss(class_weights=getattr(config, "class_weights", None))
    else:
        raise ValueError(
            f"Unknown loss type: '{loss_type}'. Choose from: physics_aware, dice_only"
        )
