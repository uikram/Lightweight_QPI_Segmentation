"""
Main Entry Point for Holographic AI Framework.
Strictly separates Training and Evaluation modes.
"""
import argparse
import os
import torch
import warnings
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from datasets.qpi_dataset import get_qpi_loaders
from models import get_model
from training.trainer_seg import SegmentationTrainer
from evaluation.evaluate import SegmentationEvaluator  # FIXED: Corrected import
from evaluation.metrics import MetricsTracker
from utils.helpers import seed_everything
from utils.config import load_config_from_yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Physics-Aware QPI Segmentation Models")
    # FIXED: Removed 'benchmark' from choices
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate'])
    parser.add_argument('--config', type=str, required=True, help="Path to config yaml")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def init_environment(args):
    """Setup config, device, data, and model."""
    config = load_config_from_yaml(args.config)
    config.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
    
    print(f"Initializing {config.architecture.upper()}")
    model = get_model(config.architecture, config)
    model.to(config.device)
    
    return model, config

def run_train(args):
    model, config = init_environment(args)
    
    # Use built-in factory to get data loaders
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
    
    # Load best weights
    best_model_path = config.checkpoint_dir / "best_model.pt"
    if best_model_path.exists():
        ckpt = torch.load(best_model_path, map_location=config.device)
        model.load_state_dict(ckpt.get("model_state", ckpt))
        print(f"Loaded checkpoint from {best_model_path}")
    else:
        print("Warning: No checkpoint found. Evaluating with initialized weights.")
        
    model.eval()
    
    _, _, test_loader = get_qpi_loaders(config, num_workers=config.num_workers)
    if test_loader is None:
        print("No test dataset found in data_root. Aborting evaluation.")
        return
        
    evaluator = SegmentationEvaluator(model, config)
    
    print(f"\n{'='*40}\nSTARTING EVALUATION\n{'='*40}")
    metrics = evaluator.evaluate(test_loader)
    print("\nEvaluation Complete. Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

def main():
    args = parse_args()
    if args.mode == 'train': run_train(args)
    elif args.mode == 'evaluate': run_evaluate(args)

if __name__ == "__main__":
    main()