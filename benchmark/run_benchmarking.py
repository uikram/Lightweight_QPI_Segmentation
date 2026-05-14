import torch
import json
import time
import argparse
import sys
import gc
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models import get_model
from utils.config import load_config_from_yaml

# Configuration Defaults
LATENCY_WARMUP_RUNS = 50
LATENCY_RUNS = 200

def force_cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def measure_latency_and_memory(model, dummy_input, device):
    """Measures inference latency and peak GPU memory."""
    # Warmup
    model.eval()
    for _ in range(LATENCY_WARMUP_RUNS):
        with torch.no_grad():
            model(dummy_input)
    if device != 'cpu':
        torch.cuda.synchronize()

    # Reset memory stats
    if device != 'cpu':
        torch.cuda.reset_peak_memory_stats()
        
    timings = []
    
    # Benchmark
    for _ in range(LATENCY_RUNS):
        start = time.perf_counter()
        with torch.no_grad():
            model(dummy_input)
        if device != 'cpu':
            torch.cuda.synchronize()
        end = time.perf_counter()
        timings.append((end - start) * 1000)

    timings_arr = np.array(timings)
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device != 'cpu' else 0.0
    
    return {
        "mean_ms": float(np.mean(timings_arr)),
        "std_ms":  float(np.std(timings_arr)),
        "p50_ms":  float(np.percentile(timings_arr, 50)),
        "p99_ms":  float(np.percentile(timings_arr, 99)),
        "fps":     float(1000.0 / np.mean(timings_arr)),
        "peak_memory_mb": peak_mem_mb
    }

def run_benchmark(config_path, device='cuda'):
    config = load_config_from_yaml(config_path)
    config.device = device
    
    print(f"\nBenchmarking Architecture: {config.architecture.upper()}")
    print("-" * 60)
    
    model = get_model(config.architecture, config)
    model.to(device)
    
    # QPI Dummy Input: (Batch, 1 Channel, H, W)
    batch_size = 1
    image_size = getattr(config, 'image_size', 256)
    dummy_input = torch.randn(batch_size, 1, image_size, image_size, device=device)
    
    stats = measure_latency_and_memory(model, dummy_input, device)
    
    print(f"  -> Mean Latency: {stats['mean_ms']:.2f} ms")
    print(f"  -> P99 Latency:  {stats['p99_ms']:.2f} ms")
    print(f"  -> FPS:          {stats['fps']:.1f}")
    print(f"  -> Peak Memory:  {stats['peak_memory_mb']:.2f} MB")
    
    del model
    force_cleanup()
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', default='results/benchmarks')
    args = parser.parse_args()

    # Define the config files for the new QPI segmentation models
    configs_to_test = [
        "configs/edge_sam_lora.yaml",
        "configs/mobilenet_unet_lora.yaml", 
        "configs/mobile_sam_lora.yaml"
    ]
    
    results = {}
    
    for cfg_path in configs_to_test:
        if Path(cfg_path).exists():
            arch_name = Path(cfg_path).stem
            results[arch_name] = run_benchmark(cfg_path, args.device)
        else:
            print(f"Warning: {cfg_path} not found. Skipping.")

    # Save Results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "qpi_segmentation_benchmark.json"
    
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n[DONE] Saved benchmark results to {out_file}")