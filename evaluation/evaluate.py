import torch
from evaluation.seg_metrics import SegmentationMetrics


class SegmentationEvaluator:
    def __init__(self, model, config):
        self.model  = model
        self.config = config
        self.device = getattr(config, 'device', 'cuda')
        self.seg_metrics = SegmentationMetrics(
            num_classes=getattr(config, 'num_classes', 5)
        )

    def evaluate(self, test_loader):
        self.model.eval()
        self.seg_metrics.reset()

        print(f"\n[Evaluation] Running segmentation inference...")
        with torch.no_grad():
            for batch in test_loader:
                images    = batch["phase"].to(self.device)
                targets   = batch["mask"].to(self.device)
                phase_raw = batch.get("phase_raw", images).to(self.device)

                logits = self.model(images)            # (B, num_classes, H, W)

                # Multi-class → binary for physics-aware segmentation metrics.
                # argmax selects the predicted class per pixel;
                # any non-background class (> 0) is treated as cell.
                pred_classes  = torch.argmax(logits, dim=1)   # (B, H, W)
                pred_binary   = (pred_classes > 0).long()
                target_binary = (targets > 0).long()

                self.seg_metrics.update(pred_binary, target_binary, phase_raw)

        results = self.seg_metrics.compute()
        self.seg_metrics.print_results(results, prefix="Test ")
        return results
