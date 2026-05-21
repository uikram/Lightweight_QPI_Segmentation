"""
Segmentation evaluation metrics for QPI holographic cell analysis.

Metrics:
    - Dice coefficient
    - Aggregated Jaccard Index (AJI)
    - Intersection-over-Union (IoU / Jaccard)
    - Boundary F1-score (BF1)
    - Phase-integrated optical quantity error (PhaseVolError)
    - Inference latency (ms/frame)
    - Frames per second (FPS)
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import time


class SegmentationMetrics:
    """
    Accumulates predictions over an epoch and computes all metrics.

    Usage:
        metrics = SegmentationMetrics(num_classes=1)
        for batch in loader:
            pred_binary = (sigmoid(logits) > 0.5).long()
            metrics.update(pred_binary, target, phase_raw)
        results = metrics.compute()
    """

    def __init__(self, num_classes: int = 5, boundary_tolerance: int = 2):
        self.num_classes         = num_classes
        self.boundary_tolerance  = boundary_tolerance
        self.reset()

    def reset(self):
        # Arrays to track per-class metrics
        self._tp = np.zeros(self.num_classes)
        self._fp = np.zeros(self.num_classes)
        self._fn = np.zeros(self.num_classes)

        # Binary/Physics tracking
        self._aji_intersection = 0.0
        self._aji_union        = 0.0
        self._boundary_tp = 0.0
        self._boundary_fp = 0.0
        self._boundary_fn = 0.0
        self._phase_vol_errors: List[float] = []
        self._latencies: List[float]        = []
        self._n_samples = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor, phase_raw: Optional[torch.Tensor] = None):
        pred_np   = pred.cpu().numpy().astype(np.uint8)
        target_np = target.cpu().numpy().astype(np.uint8)
        B = pred_np.shape[0]
        self._n_samples += B
        
        from scipy.ndimage import label as cc_label

        for i in range(B):
            p = pred_np[i]
            t = target_np[i]
            
            # 1. PER-CLASS BIOLOGICAL METRICS (Dice, IoU)
            for c in range(1, self.num_classes): # Skip background (0)
                self._tp[c] += np.logical_and(p == c, t == c).sum()
                self._fp[c] += np.logical_and(p == c, t != c).sum()
                self._fn[c] += np.logical_and(p != c, t == c).sum()

            # 2. BINARY PHYSICS METRICS (Collapse to foreground for AJI, BF1, Phase Vol)
            p_binary = (p > 0).astype(np.uint8)
            t_binary = (t > 0).astype(np.uint8)

            # Connected Components for AJI (Your Fix 1)
            p_labeled, _ = cc_label(p_binary)
            t_labeled, _ = cc_label(t_binary)
            inter, union = _aji_components(p_labeled, t_labeled)
            self._aji_intersection += inter
            self._aji_union        += union

            # Boundary F1
            btp, bfp, bfn = _boundary_stats(p_binary, t_binary, self.boundary_tolerance)
            self._boundary_tp += btp
            self._boundary_fp += bfp
            self._boundary_fn += bfn

            # Phase volume error (Must use binary masks, not class IDs!)
            if phase_raw is not None:
                ph = phase_raw[i].cpu().numpy() if torch.is_tensor(phase_raw) else phase_raw[i]
                pv_error = _phase_volume_error(p_binary, t_binary, ph)
                self._phase_vol_errors.append(pv_error)

    def compute(self) -> Dict[str, float]:
        eps = 1e-8
        
        # Macro-averaged per-class metrics
        dice_per_class = (2 * self._tp + eps) / (2 * self._tp + self._fp + self._fn + eps)
        iou_per_class  = (self._tp + eps) / (self._tp + self._fp + self._fn + eps)
        
        # Mean scores ignoring background (index 0)
        mean_dice = float(np.mean(dice_per_class[1:]))
        mean_iou  = float(np.mean(iou_per_class[1:]))

        aji = (self._aji_intersection + eps) / (self._aji_union + eps)
        b_prec = (self._boundary_tp + eps) / (self._boundary_tp + self._boundary_fp + eps)
        b_rec  = (self._boundary_tp + eps) / (self._boundary_tp + self._boundary_fn + eps)
        bf1    = 2 * b_prec * b_rec / (b_prec + b_rec + eps)

        results = {
            "mean_dice": mean_dice,
            "mean_iou":  mean_iou,
            "aji":       float(aji),
            "bf1":       float(bf1),
        }
        
        # Add per-class logging
        classes = ["discocyte", "echinocyte", "spherocyte", "stomatocyte"]
        for c_idx, c_name in enumerate(classes, start=1):
            results[f"dice_{c_name}"] = float(dice_per_class[c_idx])

        if self._phase_vol_errors:
            results["phase_vol_error"] = float(np.mean(self._phase_vol_errors))
            results["phase_vol_error_std"] = float(np.std(self._phase_vol_errors))

        if self._latencies:
            results["latency_mean_ms"] = float(np.mean(self._latencies))
            results["latency_p99_ms"]  = float(np.percentile(self._latencies, 99))
            results["fps"]             = float(1000.0 / np.mean(self._latencies))

        return results

    def print_results(self, results: Optional[Dict] = None, prefix: str = ""):
        if results is None:
            results = self.compute()
        print(f"\n{prefix}Segmentation Metrics:")
        print(f"  Dice:            {results.get('mean_dice', 0):.4f}")
        print(f"  IoU:             {results.get('mean_iou', 0):.4f}")
        print(f"  AJI:             {results.get('aji', 0):.4f}")
        print(f"  Boundary F1:     {results.get('bf1', 0):.4f}")
        if "phase_vol_error" in results:
            print(f"  Phase Vol Error: {results['phase_vol_error']:.4f} "
                  f"± {results.get('phase_vol_error_std', 0):.4f}")
        if "latency_mean_ms" in results:
            print(f"  Latency:         {results['latency_mean_ms']:.2f} ms/frame "
                  f"(p99: {results['latency_p99_ms']:.2f} ms)")
            print(f"  FPS:             {results['fps']:.1f}")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _aji_components(pred: np.ndarray, target: np.ndarray):
    """
    Aggregated Jaccard Index components for a single image pair.
    For binary masks (single object class), equivalent to standard IoU.
    For multi-instance, each predicted instance is matched to best GT instance.
    """
    pred_ids   = np.unique(pred[pred > 0])
    target_ids = np.unique(target[target > 0])

    if len(target_ids) == 0 and len(pred_ids) == 0:
        return 1.0, 1.0  # Both empty = perfect

    if len(target_ids) == 0:
        return 0.0, pred.sum().item()

    if len(pred_ids) == 0:
        return 0.0, target.sum().item()

    total_inter = 0.0
    total_union = 0.0
    used_pred   = set()

    for t_id in target_ids:
        t_mask = target == t_id
        best_iou  = 0.0
        best_p_id = -1

        for p_id in pred_ids:
            p_mask = pred == p_id
            inter  = np.logical_and(t_mask, p_mask).sum()
            union  = np.logical_or(t_mask, p_mask).sum()
            iou    = inter / (union + 1e-8)
            if iou > best_iou:
                best_iou  = iou
                best_p_id = p_id

        if best_p_id >= 0:
            p_mask = pred == best_p_id
            inter  = np.logical_and(t_mask, p_mask).sum()
            union  = np.logical_or(t_mask, p_mask).sum()
            total_inter += inter
            total_union += union
            used_pred.add(best_p_id)
        else:
            total_union += t_mask.sum()

    # Unmatched predictions add to union
    for p_id in pred_ids:
        if p_id not in used_pred:
            total_union += (pred == p_id).sum()

    return float(total_inter), float(total_union)


def _boundary_stats(pred: np.ndarray, target: np.ndarray,
                    tolerance: int = 2):
    """
    Boundary F1 score statistics.
    Extracts boundary pixels (erosion difference) and checks overlap
    within a tolerance distance.
    """
    from scipy.ndimage import binary_erosion, binary_dilation

    def get_boundary(mask):
        eroded = binary_erosion(mask)
        return mask.astype(bool) & ~eroded

    pred_b   = get_boundary(pred)
    target_b = get_boundary(target)

    if not pred_b.any() and not target_b.any():
        return 1.0, 0.0, 0.0

    # Dilate boundaries by tolerance to allow proximity matching
    pred_dilated   = binary_dilation(pred_b,   iterations=tolerance)
    target_dilated = binary_dilation(target_b, iterations=tolerance)

    tp = np.logical_and(pred_b,   target_dilated).sum()
    fp = np.logical_and(pred_b,   ~target_dilated).sum()
    fn = np.logical_and(target_b, ~pred_dilated).sum()

    return float(tp), float(fp), float(fn)


def _phase_volume_error(pred: np.ndarray, target: np.ndarray,
                        phase: np.ndarray) -> float:
    """
    Normalized phase-integrated optical quantity error.
    Measures how well the predicted mask preserves the dry-mass proxy.
    """
    pred_vol = (pred.astype(np.float32) * phase).sum()
    gt_vol   = (target.astype(np.float32) * phase).sum()

    if abs(gt_vol) < 1e-8:
        return 0.0 if abs(pred_vol) < 1e-8 else 1.0

    return abs(pred_vol - gt_vol) / abs(gt_vol)


# ---------------------------------------------------------------------------
# Latency benchmarking utility
# ---------------------------------------------------------------------------

def benchmark_latency(model: torch.nn.Module, config,
                       n_warmup: int = 50, n_runs: int = 200) -> Dict[str, float]:
    """
    Measure inference latency for a segmentation model.
    Returns mean, std, p50, p95, p99 latency in ms, and FPS.
    """
    device = config.device
    H = W  = getattr(config, "image_size", 256)

    model.eval().to(device)
    dummy = torch.randn(1, 1, H, W, device=device)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            model(dummy)
    if "cuda" in str(device):
        torch.cuda.synchronize()

    latencies = []
    for _ in range(n_runs):
        if "cuda" in str(device):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(dummy)
        if "cuda" in str(device):
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    return {
        "mean_ms":  float(np.mean(latencies)),
        "std_ms":   float(np.std(latencies)),
        "p50_ms":   float(np.percentile(latencies, 50)),
        "p95_ms":   float(np.percentile(latencies, 95)),
        "p99_ms":   float(np.percentile(latencies, 99)),
        "fps":      float(1000.0 / np.mean(latencies)),
        "n_runs":   n_runs,
    }