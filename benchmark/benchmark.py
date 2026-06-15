"""
Comprehensive Benchmarking Script for QPI Segmentation Models.
Implements Section 4.4 of the Research Plan:
- LoRA Rank Sweep Benchmarking
- FP16 vs FP32 Inference Comparison
- TensorRT Acceleration Analysis
- Peak Memory and Parameter Counting
"""

import torch
import json
import time
import argparse
import sys
import gc
import numpy as np
import pandas as pd
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


def compile_tensorrt(model, dummy_input, precision):
    """Compiles the PyTorch model to TensorRT for edge acceleration analysis."""
    try:
        import torch_tensorrt
        print("  -> Compiling model with Torch-TensorRT...")
        trt_dtype = torch.half if precision == 'fp16' else torch.float
        
        # TensorRT compilation
        trt_model = torch_tensorrt.compile(
            model,
            inputs=[torch_tensorrt.Input(dummy_input.shape, dtype=trt_dtype)],
            enabled_precisions={trt_dtype},
            workspace_size=1 << 30  # 1 GB workspace
        )
        return trt_model
    except ImportError:
        print("  -> [ERROR] torch_tensorrt not installed. Skipping TensorRT compilation.")
        print("     Install via: pip install torch-tensorrt")
        return None
    except Exception as e:
        print(f"  -> [ERROR] TensorRT compilation failed: {e}")
        return None


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


def run_benchmark_scenario(config_path, rank, precision, use_trt, device='cuda', checkpoint_dir=None):
    """Runs a specific benchmarking scenario, loading trained weights if available."""
    config = load_config_from_yaml(config_path)
    config.device = device
    
    # Inject specific LoRA rank for this sweep
    config.lora_r = rank
    config.lora_alpha = float(rank)

    arch_lower = config.architecture.lower()
    arch_upper = config.architecture.upper()
    scenario_name = f"{arch_upper} | r={rank} | {precision.upper()} | TRT={use_trt}"
    print(f"\nBenchmarking: {scenario_name}")
    print("-" * 65)

    # 1. Initialize the base structural model configuration
    model = get_model(config.architecture, config)
    
    # 2. Automatically locate and load your trained checkpoint if requested
    if checkpoint_dir:
        # Resolves to e.g., results/edge_sam_lora_r2
        cp_dir = Path(checkpoint_dir) / f"{arch_lower}_lora_r{rank}"
        
        # Look specifically for best_model.pt inside the directory tree (handles the checkpoints/ subfolder)
        checkpoint_path = None
        for path in cp_dir.rglob("best_model.pt"):
            checkpoint_path = path
            break
                
        if checkpoint_path and checkpoint_path.exists():
            print(f"  -> Loading trained weights from: {checkpoint_path}")
            try:
                state_dict = torch.load(checkpoint_path, map_location=device)
                
                # Unpack the dictionary saved by trainer_seg.py
                if "model_state" in state_dict:
                    state_dict = state_dict["model_state"]
                    
                # strict=False safely loads the weights into the structural architecture
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if len(unexpected) == 0:
                    print("     Weights matched structural architecture successfully.")
                else:
                    print(f"     Loaded with partial match (Unexpected keys found).")
            except Exception as e:
                print(f"  -> [Warning] Failed to load checkpoint weights: {e}")
        else:
            print(f"  -> [Notice] No best_model.pt found in {cp_dir}. Benchmarking structural baseline.")

    model.to(device)
    
    # 3. Merge LoRA weights into base weights for zero-latency overhead inference
    if hasattr(model, 'merge_lora'):
        try:
            model.merge_lora()
            print("  -> LoRA weights merged for zero-latency benchmarking.")
        except NotImplementedError:
            print("  -> LoRA merge skipped (Conv2d layers; slight structural overhead retained).")

    # 4. Handle precision casting
    if precision == 'fp16':
        model = model.half()
        dummy_input = torch.randn(1, 1, getattr(config, 'image_size', 256), getattr(config, 'image_size', 256), device=device).half()
        print("  -> Converted model and inputs to FP16.")
    else:
        dummy_input = torch.randn(1, 1, getattr(config, 'image_size', 256), getattr(config, 'image_size', 256), device=device)

    # 5. Compile TensorRT if requested
    if use_trt:
        trt_model = compile_tensorrt(model, dummy_input, precision)
        if trt_model is not None:
            model = trt_model
        else:
            print("  -> Falling back to standard PyTorch execution.")
            use_trt = False # Update flag if compilation failed

    # Count parameters (Original PyTorch model parameters)
    total_params     = sum(p.numel() for p in model.parameters()) if not use_trt else "N/A (TRT)"
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) if not use_trt else "N/A"

    # Run Benchmark
    stats = measure_latency_and_memory(model, dummy_input, device)

    print(f"  Total params : {total_params}")
    print(f"  Mean latency : {stats['mean_ms']:.2f} ms")
    print(f"  P99 latency  : {stats['p99_ms']:.2f} ms")
    print(f"  FPS          : {stats['fps']:.1f}")
    print(f"  Peak memory  : {stats['peak_memory_mb']:.1f} MB")

    stats.update({
        "architecture": arch_upper,
        "lora_r": rank,
        "precision": precision.upper(),
        "tensorrt": use_trt,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "image_size": getattr(config, 'image_size', 256)
    })

    del model
    del dummy_input
    force_cleanup()
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', default='results/benchmarks')
    parser.add_argument('--train_results_dir', default=None, help="Path to your trained model directories (e.g., 'results')")
    parser.add_argument('--ranks', type=int, nargs='+', default=[2, 4, 8, 16, 32], help="LoRA ranks to benchmark")
    parser.add_argument('--precisions', type=str, nargs='+', default=['fp32', 'fp16'], help="Precisions to benchmark")
    parser.add_argument('--trt', action='store_true', help="Run TensorRT compilation and benchmarking")
    args = parser.parse_args()

    configs_to_test = [
        "configs/edge_sam_lora.yaml",
        "configs/mobilenet_unet_lora.yaml",
        "configs/mobile_sam_lora.yaml"
    ]

    all_results = []
    
    # Nested loops executing Section 4.4 requirements
    for cfg_path in configs_to_test:
        if not Path(cfg_path).exists():
            print(f"Warning: {cfg_path} not found. Skipping.")
            continue
            
        for rank in args.ranks:
            for precision in args.precisions:
                # 1. Benchmark Standard PyTorch (FP32 or FP16)
                stats = run_benchmark_scenario(
                    config_path=cfg_path, 
                    rank=rank, 
                    precision=precision, 
                    use_trt=False, 
                    device=args.device,
                    checkpoint_dir=args.train_results_dir
                )
                all_results.append(stats)
                
                # 2. Benchmark TensorRT (if requested)
                if args.trt and args.device == 'cuda':
                    trt_stats = run_benchmark_scenario(
                        config_path=cfg_path, 
                        rank=rank, 
                        precision=precision, 
                        use_trt=True, 
                        device=args.device,
                        checkpoint_dir=args.train_results_dir
                    )
                    all_results.append(trt_stats)

    # Save to JSON
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    json_file = out_dir / "hardware_benchmark_sweep.json"
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    # Save to CSV for easy copy-pasting into LaTeX/Excel for the paper
    csv_file = out_dir / "hardware_benchmark_sweep.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(csv_file, index=False)

    print(f"\n[DONE] Results saved to {out_dir}")
    print("CSV generated for easy LaTeX table formatting.")