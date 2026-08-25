import numpy as np
import cv2

class RBCMorphologyAnalyzer:
    def __init__(self):
        pass

    def compute_morphology(self, mask, phase_map):
        if hasattr(mask, 'detach'):
            mask = mask.detach().cpu().numpy()
        else:
            mask = np.asarray(mask)
            
        if hasattr(phase_map, 'detach'):
            phase_map = phase_map.detach().cpu().numpy()
        else:
            phase_map = np.asarray(phase_map)

        mask = np.squeeze(mask)
        phase_map = np.squeeze(phase_map)

        binary_mask = (mask > 0.5).astype(np.uint8)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return {"area": 0.0, "circularity": 0.0, "optical_volume": 0.0,
                    "mean_phase": 0.0, "max_phase": 0.0, "dry_mass": 0.0}

        largest_contour = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-8)
        circularity = min(float(circularity), 1.0)

        opt_volume = np.sum(binary_mask * phase_map)

        cell_phase_pixels = phase_map[binary_mask > 0]
        mean_phase = float(np.mean(cell_phase_pixels)) if len(cell_phase_pixels) > 0 else 0.0
        max_phase  = float(np.max(cell_phase_pixels))  if len(cell_phase_pixels) > 0 else 0.0

        calibration_constant = 1.0
        dry_mass = float(opt_volume * calibration_constant)

        return {
            "area": float(area),
            "circularity": float(circularity),
            "optical_volume": float(opt_volume),
            "mean_phase": mean_phase,
            "max_phase": max_phase,
            "dry_mass": dry_mass,
        }
