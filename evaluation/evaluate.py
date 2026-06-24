import torch
import pandas as pd
from pathlib import Path
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
        self.morphology_analyzer = RBCMorphologyAnalyzer()

    def evaluate(self, test_loader):
        self.model.eval()
        self.seg_metrics.reset()
        
        sample_logs = []
        
        print(f"\n[Evaluation] Running segmentation inference and morphology analysis...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                images    = batch["phase"].to(self.device)
                targets   = batch["mask"].to(self.device)
                phase_raw = batch.get("phase_raw", images).to(self.device)

                logits = self.model(images)            
                pred_classes = torch.argmax(logits, dim=1)   

                self.seg_metrics.update(pred_classes, targets, phase_raw)

                pred_softmax = torch.softmax(logits, dim=1)
                pred_fg_prob = 1.0 - pred_softmax[:, 0, :, :] 

                for i in range(len(images)):
                    patch_classes = pred_classes[i][pred_classes[i] > 0]
                    if len(patch_classes) > 0:
                        dominant_class = torch.mode(patch_classes).values.item()
                    else:
                        dominant_class = 0 
                        
                    metrics = self.morphology_analyzer.compute_morphology(pred_fg_prob[i], phase_raw[i])
                    
                    sample_logs.append({
                        "stem": batch.get("stem", [f"img_{batch_idx}_{i}"])[i], 
                        "storage_day": batch["storage_day"][i].item() if "storage_day" in batch else 0,
                        "gt_class": batch["morphology_class"][i].item() if "morphology_class" in batch else 0,
                        "pred_class": dominant_class,
                        "area": metrics["area"],
                        "circularity": metrics["circularity"],
                        "opt_volume": metrics["optical_volume"],
                    })
                    
                    global_img_idx = (batch_idx * len(images)) + i
                    print(f"Image {global_img_idx} | AI Class: {dominant_class} | Area: {metrics['area']:.2f} | Circularity: {metrics['circularity']:.4f} | Opt Volume: {metrics['optical_volume']:.2f}")

        # Dynamic subfolder resolution
        base_results_dir = Path(getattr(self.config, 'results_dir', 'results'))
        
        # Extract the specific model or run name from the config
        run_name = getattr(self.config, 'run_name', getattr(self.config, 'model_type', 'default_run'))
        
        # Prevent duplicate nesting if the base directory already includes the run name
        if base_results_dir.name == run_name:
            run_dir = base_results_dir
        else:
            run_dir = base_results_dir / run_name
            
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Safely extract the current LoRA rank from the config (defaults to 'base' if not found)
        current_rank = getattr(self.config, 'lora_rank', getattr(self.config, 'r', 'base'))
        
        # Append the rank to the filename so they never overwrite each other
        csv_filename = f"morphology_trends_rank_{current_rank}.csv"
        csv_path = run_dir / csv_filename
        
        pd.DataFrame(sample_logs).to_csv(csv_path, index=False)
        print(f"\n[Evaluation] Saved temporal morphology trends to {csv_path}")

        results = self.seg_metrics.compute()
        self.seg_metrics.print_results(results, prefix="Test ")

        return results