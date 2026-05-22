import torch
from evaluation.seg_metrics import SegmentationMetrics
from analysis.morphology_analysis import RBCMorphologyAnalyzer

class SegmentationEvaluator:
    def __init__(self, model, config):
        self.model  = model
        self.config = config
        self.device = getattr(config, 'device', 'cuda')
        self.seg_metrics = SegmentationMetrics(
            num_classes=getattr(config, 'num_classes', 5)
        )
        # --- FIX 4 ADDITION: Initialize the analyzer ---
        self.morphology_analyzer = RBCMorphologyAnalyzer()

    def evaluate(self, test_loader):
        self.model.eval()
        self.seg_metrics.reset()
        
        print(f"\n[Evaluation] Running segmentation inference and morphology analysis...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                images    = batch["phase"].to(self.device)
                targets   = batch["mask"].to(self.device)
                phase_raw = batch.get("phase_raw", images).to(self.device)

                logits = self.model(images)            # (B, num_classes, H, W)
                pred_classes = torch.argmax(logits, dim=1)   # (B, H, W)

                # Pass the raw multi-class tensors (0-4); seg_metrics handles the rest natively
                self.seg_metrics.update(pred_classes, targets, phase_raw)

                # --- FIX 4 ADDITION: Biological Morphology Tracking ---
                # 1. Get the probability of foreground (1.0 minus background probability)
                pred_softmax = torch.softmax(logits, dim=1)
                pred_fg_prob = 1.0 - pred_softmax[:, 0, :, :] 

                # 2. Loop through the images in this batch to extract metrics
                for i in range(len(images)):
                    # Find the dominant cell class predicted by the AI in this patch (ignoring background 0)
                    patch_classes = pred_classes[i][pred_classes[i] > 0]
                    if len(patch_classes) > 0:
                        dominant_class = torch.mode(patch_classes).values.item()
                    else:
                        dominant_class = 0 # Background
                        
                    # Calculate the physical metrics (Area, Circularity, Volume)
                    metrics = self.morphology_analyzer.compute_morphology(pred_fg_prob[i], phase_raw[i])
                    
                    # Print or log the metrics for your biological analysis!
                    global_img_idx = (batch_idx * len(images)) + i
                    print(f"Image {global_img_idx} | AI Class: {dominant_class} | Area: {metrics['area']:.2f} | Circularity: {metrics['circularity']:.4f} | Opt Volume: {metrics['optical_volume']:.2f}")

        # FIX: Removed the redundant manual per-class Dice calculation loop.
        results = self.seg_metrics.compute()
        self.seg_metrics.print_results(results, prefix="Test ")

        return results