"""
Comprehensive Benchmarking Script for QPI Segmentation Models.
Implements Section 4.4, 4.5, and 5 of the Research Plan.
"""

import torch
import json
import time
import argparse
import sys
import gc
import random
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models import get_model
from utils.config import load_config_from_yaml
from datasets.qpi_dataset import get_qpi_loaders

LATENCY_WARMUP_RUNS = 20
LATENCY_RUNS        = 100

def force_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_hardware_info():
    """Logs system hardware and software versions for paper reproducibility."""
    return {
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "pytorch_ver": torch.__version__,
        "cuda_ver": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "cudnn_ver": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
    }

def get_parameter_breakdown(model, model_name):
    """Robustly splits parameters into Encoder and Decoder using explicit object references."""
    model_name = model_name.lower()
    enc_params_set = set()
    
    # Safely aggregate explicit encoder parameters
    if hasattr(model, 'encoder'):
        enc_params_set.update(model.encoder.parameters())
    elif model_name == "mobilenet_unet":
        for i in range(6):
            enc_module = getattr(model, f'enc{i}', None)
            if enc_module is not None:
                enc_params_set.update(enc_module.parameters())

    enc_total, enc_trainable = 0, 0
    dec_total, dec_trainable = 0, 0
    
    for param in model.parameters():
        num_params = param.numel()
        is_trainable = param.requires_grad
        
        if param in enc_params_set:
            enc_total += num_params
            if is_trainable: enc_trainable += num_params
        else:
            dec_total += num_params
            if is_trainable: dec_trainable += num_params
            
    overall_total = enc_total + dec_total
    overall_trainable = enc_trainable + dec_trainable
    
    return {
        "enc_total": enc_total,
        "enc_trainable": enc_trainable,
        "enc_trainable_ratio_%": (enc_trainable / enc_total * 100) if enc_total > 0 else 0,
        "dec_total": dec_total,
        "dec_trainable": dec_trainable,
        "dec_trainable_ratio_%": (dec_trainable / dec_total * 100) if dec_total > 0 else 0,
        "overall_total": overall_total,
        "overall_trainable": overall_trainable,
        "overall_trainable_ratio_%": (overall_trainable / overall_total * 100) if overall_total > 0 else 0
    }

def compile_tensorrt(model, dummy_input, precision, workspace_gb):
    try:
        import torch_tensorrt
        print("  -> Compiling model with Torch-TensorRT...")
        trt_dtype = torch.half if precision == 'fp16' else torch.float
        
        trt_model = torch_tensorrt.compile(
            model,
            inputs=[torch_tensorrt.Input(dummy_input.shape, dtype=trt_dtype)],
            enabled_precisions={trt_dtype},
            workspace_size=int(workspace_gb * (1 << 30))
        )
        return trt_model
    except Exception as e:
        print(f"  -> [ERROR] TensorRT compilation failed: {e}")
        return None

def measure_latency_and_memory(model, test_images, device):
    model.eval()
    
    for i in range(LATENCY_WARMUP_RUNS):
        img = test_images[i % len(test_images)]
        with torch.no_grad():
            model(img)
            
    if device != 'cpu':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    timings = []
    for i in range(LATENCY_RUNS):
        img = test_images[i % len(test_images)]
        
        if device != 'cpu':
            torch.cuda.synchronize()
            
        start = time.perf_counter()
        with torch.no_grad():
            model(img)
            
        if device != 'cpu':
            torch.cuda.synchronize()
            
        timings.append((time.perf_counter() - start) * 1000)

    timings_arr  = np.array(timings)
    peak_alloc_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if device != 'cpu' else 0.0
    peak_reserved_mb = (torch.cuda.max_memory_reserved() / (1024 ** 2)) if device != 'cpu' else 0.0

    return {
        "latency_mean_ms": float(np.mean(timings_arr)),
        "latency_p50_ms":  float(np.percentile(timings_arr, 50)),
        "latency_p99_ms":  float(np.percentile(timings_arr, 99)),
        "fps":             float(1000.0 / np.mean(timings_arr)),
        "peak_allocated_mb": peak_alloc_mb,
        "peak_reserved_mb":  peak_reserved_mb
    }

def run_benchmark_scenario(config_path, rank, precision, use_trt, device, checkpoint_dir, trt_workspace):
    config = load_config_from_yaml(config_path)
    config.device = device
    config.batch_size = 1 
    
    if rank == 0:
        config.lora_r = None 
    else:
        config.lora_r = rank
        config.lora_alpha = float(rank)

    arch_lower = config.architecture.lower()
    arch_upper = config.architecture.upper()
    scenario_name = f"{arch_upper} | r={'FULL' if rank==0 else rank} | {precision.upper()} | TRT={use_trt}"
    print(f"\nBenchmarking: {scenario_name}")
    print("-" * 65)

    # 1. Fetch metrics from the exact folder structure established in trainer_seg.py
    val_metrics = {}
    if checkpoint_dir:
        rank_str = f"r{rank}" if rank > 0 else "full"
        cp_dir = Path(checkpoint_dir) / f"{arch_lower}_lora_{rank_str}"
        
        metrics_file = None
        for p in cp_dir.rglob("metrics.json"):
            metrics_file = p
            break
            
        if metrics_file and metrics_file.exists():
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                val_best = data.get("seg_metrics", {}).get("val_best", {})
                val_metrics = {
                    "mean_dice": val_best.get("mean_dice"),
                    "mean_iou": val_best.get("mean_iou"),
                    "aji": val_best.get("aji"),
                    "bf1": val_best.get("bf1"),
                    "phase_vol_error": val_best.get("phase_vol_error")
                }
            print(f"  -> Accuracy metrics loaded from {metrics_file.parent.name}")

    # 2. Build Model
    model = get_model(config.architecture, config)
    model.to(device)
    
    # --- EXPLICIT ENCODER LOGGING FOR VERIFICATION ---
    if hasattr(model, 'encoder'):
        print(f"  -> [VERIFICATION] Encoder Type: {type(model.encoder)}")
    else:
        print(f"  -> [VERIFICATION] No unified 'encoder' module (Expected for MobileNetUNet)")
    # -------------------------------------------------
    
    # 3. Param counts BEFORE merge
    param_stats = get_parameter_breakdown(model, arch_lower)

    # 4. Merge LoRA for zero-latency overhead inference
    if hasattr(model, 'merge_lora') and rank > 0:
        try:
            model.merge_lora()
            print("  -> LoRA weights merged for zero-latency benchmarking.")
        except NotImplementedError:
            pass

    # 5. Fetch and Randomly Sample Validation Images
    _, val_loader, _ = get_qpi_loaders(config, num_workers=0)
    all_val_images = []
    for batch in val_loader:
        all_val_images.append(batch["phase"])
        
    random.seed(42) # Reproducibility
    total_needed = LATENCY_WARMUP_RUNS + LATENCY_RUNS
    sampled_images = random.sample(all_val_images, min(total_needed, len(all_val_images)))
    
    test_images = []
    for img in sampled_images:
        img = img.to(device)
        if precision == 'fp16':
            img = img.half()
        test_images.append(img)
        
    if precision == 'fp16':
        model = model.half()

    # 6. TensorRT Compilation
    if use_trt:
        trt_model = compile_tensorrt(model, test_images[0], precision, trt_workspace)
        if trt_model is not None:
            model = trt_model
        else:
            use_trt = False 

    # 7. Execute Benchmark
    stats = measure_latency_and_memory(model, test_images, device)

    print(f"  Encoder Trainable: {param_stats['enc_trainable_ratio_%']:.2f}% | Overall: {param_stats['overall_trainable_ratio_%']:.2f}%")
    print(f"  Mean latency     : {stats['latency_mean_ms']:.2f} ms")
    print(f"  FPS              : {stats['fps']:.1f}")
    if device != 'cpu':
        print(f"  Peak Reserved    : {stats['peak_reserved_mb']:.1f} MB")

    combined_stats = {
        "architecture": arch_upper,
        "device": device.upper(),
        "lora_r": "FULL" if rank == 0 else rank,
        "precision": precision.upper(),
        "tensorrt": use_trt,
    }
    
    combined_stats.update(get_hardware_info()) # Append Hardware Meta
    combined_stats.update(param_stats) 
    combined_stats.update(val_metrics) 
    combined_stats.update(stats)       

    del model
    del test_images
    force_cleanup()
    return combined_stats

if __name__ == "__main__":
    # Optimize cuDNN for hardware
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', default='results/benchmarks')
    parser.add_argument('--train_results_dir', default='results')
    parser.add_argument('--ranks', type=int, nargs='+', default=[0, 2, 4, 8, 16, 32])
    parser.add_argument('--precisions', type=str, nargs='+', default=['fp32', 'fp16'])
    parser.add_argument('--trt', action='store_true')
    parser.add_argument('--trt_workspace', type=float, default=1.0)
    args = parser.parse_args()

    configs_to_test = [
        "configs/edge_sam_lora.yaml",
        "configs/mobilenet_unet_lora.yaml",
        "configs/mobile_sam_lora.yaml"
    ]

    all_results = []
    
    for cfg_path in configs_to_test:
        if not Path(cfg_path).exists():
            continue
            
        for rank in args.ranks:
            for precision in args.precisions:
                if args.device == 'cpu' and precision == 'fp16':
                    continue 
                    
                stats = run_benchmark_scenario(cfg_path, rank, precision, False, args.device, args.train_results_dir, args.trt_workspace)
                all_results.append(stats)
                
                if args.trt and args.device == 'cuda':
                    trt_stats = run_benchmark_scenario(cfg_path, rank, precision, True, args.device, args.train_results_dir, args.trt_workspace)
                    all_results.append(trt_stats)

    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = out_dir / f"hardware_benchmark_{args.device}.csv"
    df = pd.DataFrame(all_results)
    
    # Priority sorting for the final output CSV
    cols = df.columns.tolist()
    headline_cols = [
        "architecture", "device", "gpu_name", "lora_r", "precision", "tensorrt", 
        "enc_trainable_ratio_%", "dec_trainable_ratio_%", "overall_trainable_ratio_%",
        "mean_dice", "bf1", "phase_vol_error", 
        "latency_mean_ms", "fps", "peak_allocated_mb", "peak_reserved_mb"
    ]
    ordered_cols = [c for c in headline_cols if c in cols] + [c for c in cols if c not in headline_cols]
    df = df[ordered_cols]
    
    df.to_csv(csv_file, index=False)
    print(f"\n[DONE] Trade-off CSV generated at {csv_file}")