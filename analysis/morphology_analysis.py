import numpy as np
import cv2
import torch

class RBCMorphologyAnalyzer:
    """
    Quantitative morphology analyzer from segmented phase images.
    Tracks: projected area, circularity, and phase-integrated optical volume.
    NOTE: Cell classification is handled directly by the AI's 5-class prediction.
    """
    def __init__(self):
        # We no longer need the circularity_threshold because we trust the AI to classify.
        pass

    def compute_morphology(self, mask, phase_map):
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        if isinstance(phase_map, torch.Tensor):
            phase_map = phase_map.detach().cpu().numpy()
            
        # Binarize to find the physical contours for math metrics
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return {"area": 0.0, "circularity": 0.0, "optical_volume": 0.0}
            
        # Target the primary cell in the patch (largest contour) to prevent math errors
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 1. Projected Area (Instance-specific)
        area = cv2.contourArea(largest_contour)
        
        # 2. Circularity (Instance-specific)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-8)
        
        # Cap circularity at 1.0 to prevent pixel-grid anomalies
        circularity = min(float(circularity), 1.0)
        
        # 3. Phase-Integrated Optical Volume
        opt_volume = np.sum(binary_mask * phase_map)
        
        return {
            "area": float(area), 
            "circularity": float(circularity), 
            "optical_volume": float(opt_volume)
        }