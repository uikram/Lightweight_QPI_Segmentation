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
        class_names = ['background', 'discocyte', 'echinocyte', 'spherocyte', 'stomatocyte']
        num_classes = getattr(self.config, 'num_classes', 5)
        per_class_tp = [0.0] * num_classes
        per_class_fp = [0.0] * num_classes
        per_class_fn = [0.0] * num_classes
        print(f"\n[Evaluation] Running segmentation inference...")
        with torch.no_grad():
            for batch in test_loader:
                
                images    = batch["phase"].to(self.device)
                targets   = batch["mask"].to(self.device)
                phase_raw = batch.get("phase_raw", images).to(self.device)

                logits = self.model(images)            # (B, num_classes, H, W)

                pred_classes  = torch.argmax(logits, dim=1)   # (B, H, W)

                # Pass the raw multi-class tensors (0-4)
                self.seg_metrics.update(pred_classes, targets, phase_raw)

                for c in range(1, num_classes):
                    pred_c   = (pred_classes == c).long()
                    target_c = (targets == c).long()
                    per_class_tp[c] += (pred_c * target_c).sum().item()
                    per_class_fp[c] += (pred_c * (1 - target_c)).sum().item()
                    per_class_fn[c] += ((1 - pred_c) * target_c).sum().item()

        results = self.seg_metrics.compute()
        self.seg_metrics.print_results(results, prefix="Test ")

        print("\nPer-Class Dice:")
        
        for c in range(1, num_classes):
            dice_c = (2 * per_class_tp[c]) / (2 * per_class_tp[c] + per_class_fp[c] + per_class_fn[c] + 1e-8)
            print(f"  [{class_names[c]}]: {dice_c:.4f}")
            results[f"dice_{class_names[c]}"] = dice_c # Optional: save to results dict

        return results
