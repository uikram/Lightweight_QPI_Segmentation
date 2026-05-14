"""
Main Entry Point for Holographic AI Framework.
Strictly separates Training, Evaluation, and Benchmarking modes.
"""
import argparse
import os
import yaml
import torch
import warnings
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader

from models import get_model
from datasets.qpi_dataset import QPIDataset
from training.trainer_seg import SegmentationTrainer
from training.losses import PhysicsAwarePhaseLoss
from evaluation.seg_metrics import SegmentationEvaluator
from evaluation.profiling import ModelProfiler
from utils.helpers import seed_everything

def parse_args():
    parser = argparse.ArgumentParser(description="Physics-Aware QPI Segmentation Models")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'benchmark'])
    parser.add_argument('--config', type=str, required=True, help="Path to config yaml")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def init_environment(args):
    """Setup config, device, data, and model."""
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    config['device'] = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
    
    print(f"Initializing {config['model']['architecture'].upper()}")
    model = get_model(config['model']['architecture'], config.get('lora', None))
    model.to(config['device'])
    
    return model, config

def get_dataloader(config, split='train'):
    """Initializes the QPI Dataset."""
    dir_key = f"{split}_dir" # assumes config has phase_dir and mask_dir under dataset
    dataset = QPIDataset(
        phase_dir=config['dataset']['phase_dir'].replace('train', split), 
        mask_dir=config['dataset']['mask_dir'].replace('train', split), 
        config=config
    )
    return DataLoader(
        dataset, 
        batch_size=config['dataset']['batch_size'], 
        shuffle=(split == 'train'),
        num_workers=config['dataset']['num_workers']
    )

# --- MODES ---

def run_train(args):
    model, config = init_environment(args)
    dataloader = get_dataloader(config, split='train')
    
    print("Initializing Physics-Aware Loss...")
    criterion = PhysicsAwarePhaseLoss(
        lambda1=config['loss_weights']['lambda1_pmc'],
        lambda2=config['loss_weights']['lambda2_bga'],
        lambda3=config['loss_weights']['lambda3_pv']
    )
    
    print(f"\n{'='*40}\nSTARTING TRAINING\n{'='*40}")
    trainer = SegmentationTrainer(
        model=model, 
        dataloader=dataloader, 
        criterion=criterion, 
        config=config,
        device=config['device']
    )
    trainer.train()
    print("Training Complete.")

def run_evaluate(args):
    model, config = init_environment(args)
    # TODO: Implement load_checkpoint for the new segmentation models
    # model = load_checkpoint(model, config) 
    model.eval()
    
    test_loader = get_dataloader(config, split='test')
    evaluator = SegmentationEvaluator(model, config)
    
    print(f"\n{'='*40}\nSTARTING EVALUATION\n{'='*40}")
    metrics = evaluator.evaluate(test_loader)
    print("Evaluation Complete. Results:", metrics)

def run_benchmark(args):
    """
    Run benchmarking to measure edge deployment feasibility (Latency, FPS, Memory).
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUITE: Edge Deployability ({args.config})")
    print(f"{'='*60}")

    model, config = init_environment(args)
    model.eval()
    
    test_loader = get_dataloader(config, split='test')
    
    # Use the ModelProfiler from your old repo
    run_name = f"{config['model']['architecture']}_benchmark"
    profiler = ModelProfiler(model, run_name, config)
    
    print(f"\n>>> Running Profiler on {config['device']}...")
    try:
        # Assuming ModelProfiler measures latency, FPS, and peak memory
        results = profiler.profile(test_loader, num_samples=100) 
        profiler.save_results(results)
        profiler.print_summary(results)
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback; traceback.print_exc()

def main():
    args = parse_args()
    if args.mode == 'train': run_train(args)
    elif args.mode == 'evaluate': run_evaluate(args)
    elif args.mode == 'benchmark': run_benchmark(args)

if __name__ == "__main__":
    main()