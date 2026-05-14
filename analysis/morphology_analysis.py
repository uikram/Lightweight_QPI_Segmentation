import numpy as np
import cv2
import torch

class RBCMorphologyAnalyzer:
    """
    Quantitative morphology analyzer from segmented phase images.
    Tracks: projected area, circularity, and phase-integrated optical volume.
    """
    def __init__(self):
        # Thresholds can be empirically tuned based on your dataset
        self.circularity_threshold = 0.85

    def compute_morphology(self, mask, phase_map):
        # Ensure mask is a binary numpy array
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        if isinstance(phase_map, torch.Tensor):
            phase_map = phase_map.detach().cpu().numpy()
            
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # 1. Projected Area
        area = np.sum(binary_mask)
        
        # 2. Circularity
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return {"area": 0.0, "circularity": 0.0, "optical_volume": 0.0}
            
        perimeter = cv2.arcLength(contours[0], True)
        # Circularity formula: 4 * pi * Area / Perimeter^2
        circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-8)
        
        # 3. Phase-Integrated Optical Volume
        opt_volume = np.sum(binary_mask * phase_map)
        
        return {
            "area": float(area), 
            "circularity": float(circularity), 
            "optical_volume": float(opt_volume)
        }

    def classify_morphology(self, circularity, area):
        """Rule-based classification for storage-induced degradation."""
        if circularity >= self.circularity_threshold:
            return "discocyte" # Healthy/Baseline
        elif circularity < self.circularity_threshold and area > np.mean(area): # Example heuristic
            return "stomatocyte"
        else:
            return "echinocyte" # Degrading