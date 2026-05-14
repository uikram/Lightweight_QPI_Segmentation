import torch
from evaluation.seg_metrics import SegmentationMetrics

class SegmentationEvaluator:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device
        # Initialize the metrics tracker from seg_metrics.py
        self.seg_metrics = SegmentationMetrics(num_classes=config.get('num_classes', 1))

    def evaluate(self, test_loader):
        self.model.eval()
        self.seg_metrics.reset()

        print(f"\n[Evaluation] Running Segmentation Inference...")
        with torch.no_grad():
            for batch in test_loader:
                # Unpack the QPI Dataset batch
                # Assuming batch yields: (phase_images, target_masks, phase_raw)
                images = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                
                # If phase_raw is passed for the optical volume calculation
                phase_raw = batch[2].to(self.device) if len(batch) > 2 else images

                # Forward pass
                logits = self.model(images)
                
                # Binarize predictions
                pred_binary = (torch.sigmoid(logits) > 0.5).long()

                # Update metrics
                self.seg_metrics.update(pred_binary, targets, phase_raw)

        results = self.seg_metrics.compute()
        return results