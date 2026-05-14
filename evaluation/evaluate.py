import torch
from evaluation.seg_metrics import SegmentationMetrics

class SegmentationEvaluator:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = getattr(config, 'device', 'cuda')
        # Initialize the metrics tracker
        self.seg_metrics = SegmentationMetrics(num_classes=getattr(config, 'num_classes', 1))

    def evaluate(self, test_loader):
        self.model.eval()
        self.seg_metrics.reset()

        print(f"\n[Evaluation] Running Segmentation Inference...")
        with torch.no_grad():
            for batch in test_loader:
                # FIXED: QPIDataset returns a dictionary, not a tuple
                images = batch["phase"].to(self.device)
                targets = batch["mask"].to(self.device)
                
                # Fetch unnormalized phase for optical volume calculation
                phase_raw = batch.get("phase_raw", images).to(self.device)

                # Forward pass
                logits = self.model(images)
                
                # Binarize predictions
                pred_binary = (torch.sigmoid(logits) > 0.5).long()

                # Update metrics
                self.seg_metrics.update(pred_binary, targets, phase_raw)

        results = self.seg_metrics.compute()
        return results