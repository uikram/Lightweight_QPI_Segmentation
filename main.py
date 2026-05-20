"""
Main Entry Point for Holographic AI Framework.
Strictly separates Training, Evaluation, and Rank Sweep modes.
"""
import argparse
import os
os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(__file__), "cache")
import gc
import torch
import warnings
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from datasets.qpi_dataset import get_qpi_loaders
from models import get_model
from training.trainer_seg import SegmentationTrainer
from evaluation.evaluate import SegmentationEvaluator
from evaluation.metrics import MetricsTracker
from utils.helpers import seed_everything
from utils.config import load_config_from_yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Physics-Aware QPI Segmentation Models")
    # Added 'sweep' to the allowed modes
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'sweep'])
    parser.add_argument('--config', type=str, required=True, help="Path to config yaml")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    # Added ranks argument specifically for the sweep mode
    parser.add_argument('--ranks', type=int, nargs='+', default=[2, 4, 8, 16], help="List of LoRA ranks to sweep")
    return parser.parse_args()

def init_environment(args, override_rank=None):
    """Setup config, device, data, and model."""
    config = load_config_from_yaml(args.config)
    config.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
    
    # If sweeping, dynamically override the rank and directories in memory
    if override_rank is not None:
        config.lora_r = override_rank
        base_dir = getattr(config, 'results_dir', Path('results'))
        # Create separate output folders for each rank (e.g., results_dir_r8)
        config.results_dir = Path(f"{str(base_dir)}_r{override_rank}")
        config.checkpoint_dir = config.results_dir / "checkpoints"
        
    print(f"Initializing {config.architecture.upper()} (LoRA r={getattr(config, 'lora_r', 'None')})")
    model = get_model(config.architecture, config)
    model.to(config.device)
    
    return model, config

def run_train(args):
    model, config = init_environment(args)
    
    train_loader, val_loader, _ = get_qpi_loaders(config, num_workers=config.num_workers)
    metrics_tracker = MetricsTracker(config.architecture, config.results_dir)
    
    print(f"\n{'='*40}\nSTARTING TRAINING\n{'='*40}")
    trainer = SegmentationTrainer(
        model=model, 
        config=config,
        metrics_tracker=metrics_tracker
    )
    trainer.train()
    print("Training Complete.")

def run_evaluate(args):
    model, config = init_environment(args)
    
    best_model_path = config.checkpoint_dir / "best_model.pt"
    if best_model_path.exists():
        ckpt = torch.load(best_model_path, map_location=config.device)
        model.load_state_dict(ckpt.get("model_state", ckpt))
        print(f"Loaded checkpoint from {best_model_path}")
    else:
        print("Warning: No checkpoint found. Evaluating with initialized weights.")
        
    model.eval()
    
    # FIX: Extract the test_loader (3rd element) instead of val_loader
    _, _, test_loader = get_qpi_loaders(config, num_workers=config.num_workers)
    
    if test_loader is None:
        print("No test dataset found. Aborting evaluation.")
        return
        
    evaluator = SegmentationEvaluator(model, config)
    
    print(f"\n{'='*40}\nSTARTING EVALUATION\n{'='*40}")
    # FIX: Evaluate strictly on the test set
    metrics = evaluator.evaluate(test_loader)
    print("\nEvaluation Complete. Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

def run_sweep(args):
    """Executes the systematic evaluation of different LoRA ranks sequentially."""
    print(f"\n{'='*50}\nSTARTING LORA RANK SWEEP: {args.ranks}\n{'='*50}")
    
    for r in args.ranks:
        print(f"\n\n>>>>> TRAINING WITH LORA RANK: {r} <<<<<")
        
        # Initialize everything with the overridden rank
        model, config = init_environment(args, override_rank=r)
        
        # Override lora_alpha dynamically to keep the LoRA scaling ratio (alpha/r) equal to 1.0
        config.lora_alpha = float(r)
        metrics_tracker = MetricsTracker(f"{config.architecture}_r{r}", config.results_dir)
        trainer = SegmentationTrainer(model, config, metrics_tracker)
        trainer.train()
        print(f"Finished training for rank {r}.")
        
        # Critical: Free up GPU memory before starting the next rank
        del model
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print(f"\n{'='*50}\nRANK SWEEP COMPLETE\n{'='*50}")

def main():
    args = parse_args()
    if args.mode == 'train': run_train(args)
    elif args.mode == 'evaluate': run_evaluate(args)
    elif args.mode == 'sweep': run_sweep(args)

if __name__ == "__main__":
    main()