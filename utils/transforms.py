import torch
from torchvision import transforms

class TransformFactory:
    @staticmethod
    def get_transform(config):
        """
        Physics-preserving transformations for Quantitative Phase Images (QPI).
        Strictly avoids RGB conversion and semantic ImageNet normalization.
        """
        image_size = getattr(config, "image_size", 256)
        
        print(f"---- Using QPI Physics-Preserving Transform ({image_size}x{image_size}) ----")
        
        # Note: Phase normalization (subtracting mean, dividing by std) 
        # should happen inside the Dataset loader per-image for numerical stability, 
        # NOT here across a global dataset mean.
        return transforms.Compose([
            transforms.ToPILImage(), # Assuming input is a 1-channel numpy array
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor() # Converts back to (1, H, W) float32 tensor
        ])