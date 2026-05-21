"""
Standalone benchmarking script for QPI segmentation models.
Measures inference latency, FPS, and peak GPU memory using dummy tensors.
LoRA is injected using the rank specified in each model's config so that
benchmark numbers reflect the actual LoRA-adapted models used in the paper.

Usage:
    python benchmark/run_benchmarking.py --device cuda
"""

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

LATENCY_WARMUP_RUNS = 50
LATENCY_RUNS        = 200


def force_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def measure_latency_and_memory(model, dummy_input, device):
    """Measures inference latency (ms) and peak GPU memory (MB)."""
    model.eval()

    # Warmup
    for _ in range(LATENCY_WARMUP_RUNS):
        with torch.no_grad():
            model(dummy_input)
    if device != 'cpu':
        torch.cuda.synchronize()

    # Reset memory stats before benchmark window
    if device != 'cpu':
        torch.cuda.reset_peak_memory_stats()

    timings = []
    for _ in range(LATENCY_RUNS):
        if device != 'cpu':
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model(dummy_input)
        if device != 'cpu':
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)

    timings_arr  = np.array(timings)
    peak_mem_mb  = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                    if device != 'cpu' else 0.0)

    return {
        "mean_ms":        float(np.mean(timings_arr)),
        "std_ms":         float(np.std(timings_arr)),
        "p50_ms":         float(np.percentile(timings_arr, 50)),
        "p99_ms":         float(np.percentile(timings_arr, 99)),
        "fps":            float(1000.0 / np.mean(timings_arr)),
        "peak_memory_mb": float(peak_mem_mb),
    }


def run_benchmark(config_path, device='cuda'):
    config        = load_config_from_yaml(config_path)
    config.device = device

    arch = config.architecture.upper()
    print(f"\nBenchmarking: {arch}  (LoRA r={getattr(config, 'lora_r', 'None')})")
    print("-" * 60)

    model = get_model(config.architecture, config)
    model.to(device)
    if hasattr(model, 'merge_lora'):
        try:
            model.merge_lora()
            print("  -> LoRA weights merged for zero-latency benchmarking.")
        except NotImplementedError:
            print("  -> LoRA merge skipped (Conv2d layers; slight overhead retained).")
    # Image size must match the model's expected input.
    # MobileSAM internally resizes to 512; using the correct size here
    # ensures dummy latency reflects real inference cost.
    image_size = getattr(config, 'image_size', 256)

    # Single-channel QPI dummy input
    dummy_input = torch.randn(1, 1, image_size, image_size, device=device)

    # Count trainable vs total parameters
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    stats = measure_latency_and_memory(model, dummy_input, device)

    print(f"  Architecture : {arch}")
    print(f"  Image size   : {image_size}x{image_size}")
    print(f"  Total params : {total_params:,}")
    print(f"  Trainable    : {trainable_params:,}  "
          f"({100*trainable_params/total_params:.2f}%)")
    print(f"  Mean latency : {stats['mean_ms']:.2f} ms")
    print(f"  P99 latency  : {stats['p99_ms']:.2f} ms")
    print(f"  FPS          : {stats['fps']:.1f}")
    print(f"  Peak memory  : {stats['peak_memory_mb']:.1f} MB")

    stats["total_params"]     = total_params
    stats["trainable_params"] = trainable_params
    stats["image_size"]       = image_size
    stats["architecture"]     = arch
    stats["lora_r"]           = getattr(config, 'lora_r', None)

    del model
    force_cleanup()
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', default='results/benchmarks')
    args = parser.parse_args()

    configs_to_test = [
        "configs/mobilenet_unet_lora.yaml",
        "configs/mobile_sam_lora.yaml",
        "configs/edge_sam_lora.yaml",
    ]

    results = {}
    for cfg_path in configs_to_test:
        if Path(cfg_path).exists():
            arch_name = Path(cfg_path).stem
            results[arch_name] = run_benchmark(cfg_path, args.device)
        else:
            print(f"Warning: {cfg_path} not found. Skipping.")

    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "qpi_segmentation_benchmark.json"

    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n[DONE] Results saved to {out_file}")
