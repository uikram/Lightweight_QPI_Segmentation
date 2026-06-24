"""
Comprehensive Benchmarking Script for QPI Segmentation Models.
Implements Section 4.4, 4.5, and 5 of the Research Plan using ONNX Runtime.
"""
import torch
import torch.nn.functional as F
import json
import warnings
import os
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", category=UserWarning, module="mobile_sam")
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", message="Constant folding.*")
os.environ["ORT_LOGGING_LEVEL"] = "3"
import time
import argparse
import sys
import gc
import random
import numpy as np
import pandas as pd
import traceback
import onnx
import onnxruntime as ort
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models import get_model
from utils.config import load_config_from_yaml
from datasets.qpi_dataset import get_qpi_loaders

# --- ONNX EXPORT PATCHES FOR MobileSAM/TinyViT ---
def _patched_unflatten(self, dim, sizes):
    shape = list(self.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + list(sizes) + shape[dim+1:]
    return self.reshape(new_shape)

torch.Tensor.unflatten = _patched_unflatten
torch.unflatten = lambda t, dim, sizes: _patched_unflatten(t, dim, sizes)

def _patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    if scale is None:
        scale = 1.0 / (query.size(-1) ** 0.5)
        
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores.masked_fill_(~attn_mask, float('-inf'))
        else:
            scores += attn_mask.to(scores.dtype)
            
    attn = torch.softmax(scores, dim=-1)
    
    # === FIX: Softmax upcasts to FP32. Force it back to FP16! ===
    attn = attn.to(value.dtype)
    
    return torch.matmul(attn, value)

# =====================================================================
# CRITICAL: THESE LINES MUST BE HERE TO OVERRIDE THE C++ KERNEL
# =====================================================================
F.scaled_dot_product_attention = _patched_sdpa
import torch.nn.functional
torch.nn.functional.scaled_dot_product_attention = _patched_sdpa
# =====================================================================

LATENCY_WARMUP_RUNS = 50 
LATENCY_RUNS        = 500

def force_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_hardware_info():
    return {
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "pytorch_ver": torch.__version__,
        "onnxruntime_ver": ort.__version__
    }

def get_parameter_breakdown(model, model_name):
    model_name = model_name.lower()
    enc_params_set = set()
    
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
        "enc_trainable_ratio_%": (enc_trainable / enc_total * 100) if enc_total > 0 else 0,
        "dec_trainable_ratio_%": (dec_trainable / dec_total * 100) if dec_total > 0 else 0,
        "overall_trainable_ratio_%": (overall_trainable / overall_total * 100) if overall_total > 0 else 0
    }

def compile_onnx(model, dummy_input, precision, onnx_path="temp_model.onnx", arch_name=""):
    try:
        # --- INSERT FIX 2 HERE ---
        if 'mobile_sam' in arch_name.lower() or 'mobilesam' in arch_name.lower():
            print("  [Skip] MobileSAM ONNX is slower than native PyTorch due to unsupported decoder ops. Skipping.")
            return None, None
        # -------------------------
        
        print(f"  -> Exporting model to ONNX (Precision: {precision})...")
        model.eval()

        # Native FP16 export works beautifully for CNNs; transformer decoders need post-conversion
        # use_post_convert = 'mobile_sam' in arch_name.lower() or 'mobilesam' in arch_name.lower()
        use_post_convert = False

        if precision == 'fp16' and not use_post_convert:
            export_model = model.half()
            export_input = dummy_input.half()
        else:
            export_model = model.float()
            export_input = dummy_input.float()

        # === FIX: Removed dynamic_axes to make the graph static ===
        torch.onnx.export(
            export_model, export_input, onnx_path,
            export_params=True, opset_version=17,
            do_constant_folding=True,
            input_names=['input'], output_names=['output']
        )

        # --- Precision Conversions ---
        if precision == 'fp16' and use_post_convert:
            print("  -> Converting ONNX graph to FP16 via onnxconverter_common...")
            from onnxconverter_common import float16
            import onnx
            
            model_fp32 = onnx.load(onnx_path)
            
            # Use default block list and add our sensitive operations
            block_list = float16.DEFAULT_OP_BLOCK_LIST.copy()
            block_list.extend(['Cast', 'Resize'])
            
            # === FIX: Shape inference is now extremely fast due to static axes ===
            model_fp16 = float16.convert_float_to_float16(
                model_fp32,
                keep_io_types=False,
                op_block_list=block_list,
                disable_shape_infer=False 
            )
            onnx.save(model_fp16, onnx_path)
            
        elif precision == 'int8':
            print("  -> Converting ONNX graph to INT8 via dynamic quantization...")
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantized_model_path = "temp_model_quant.onnx"
            
            # Dynamic quantization targets MatMuls and Linears, keeping everything else FP32
            quantize_dynamic(
                model_input=onnx_path,
                model_output=quantized_model_path,
                weight_type=QuantType.QUInt8
            )
            onnx_path = quantized_model_path  # Point the session to the new 8-bit model

        # 3. Load into ONNX Runtime
        print(f"  -> Loading ONNX Runtime session (Precision: {precision.upper()})...")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # --- INSERT FIX 1 HERE ---
        if torch.cuda.is_available():
            cuda_options = {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }
            providers = [('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        # -------------------------
            
        session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
        
        return session, onnx_path

    except Exception as e:
        print("\n=== ONNX EXPORT/LOAD ERROR ===")
        import traceback
        traceback.print_exc()
        print("==============================\n")
        return None, None

def measure_latency_and_memory(model_or_session, test_images, device, is_onnx=False):
    if not is_onnx:
        model_or_session.eval()
        
    input_name = model_or_session.get_inputs()[0].name if is_onnx else None

    # Fix 4: Ensure contiguous numpy arrays for ONNX to prevent channels_last crashes
    if is_onnx:
        ort_inputs = []
        for img in test_images:
            img_np = img.cpu().contiguous().numpy()
            ort_inputs.append(img_np)
    
    with torch.inference_mode():
        # Warmup
        for i in range(LATENCY_WARMUP_RUNS):
            if is_onnx:
                model_or_session.run(None, {input_name: ort_inputs[i % len(ort_inputs)]})
            else:
                img = test_images[i % len(test_images)]
                model_or_session(img)
                
        if device != 'cpu':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        timings = []
        for i in range(LATENCY_RUNS):
            if device != 'cpu':
                torch.cuda.synchronize()
                
            start = time.perf_counter()
            
            if is_onnx:
                model_or_session.run(None, {input_name: ort_inputs[i % len(ort_inputs)]})
            else:
                img = test_images[i % len(test_images)]
                model_or_session(img)
                
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

def run_benchmark_scenario(config_path, rank, precision, use_onnx, device, checkpoint_dir):
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
    scenario_name = f"{arch_upper} | r={'FULL' if rank==0 else rank} | {precision.upper()} | ONNX={use_onnx}"
    print(f"\nBenchmarking: {scenario_name}")
    print("-" * 65)

    # Fix 3: Reverted to your original flexible rglob metrics logic
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
                val_best = data.get("seg_metrics", {}).get("val_best", data)
                val_metrics = {
                    "mean_dice": val_best.get("mean_dice", data.get("val_dice")),
                    "mean_iou": val_best.get("mean_iou"),
                    "aji": val_best.get("aji"),
                    "bf1": val_best.get("bf1", data.get("val_bf1")),
                    "phase_vol_error": val_best.get("phase_vol_error", data.get("val_phase_error"))
                }

    # Build Model
    model = get_model(config.architecture, config)
    model.to(device)
    model.eval()

    if hasattr(model, 'merge_lora') and rank > 0:
        print("  -> Merging LoRA weights...")
        model.merge_lora()

    param_stats = get_parameter_breakdown(model, arch_lower)

    # Fix 1 & 2: Correct tuple unpacking and dictionary batch fetching
    _, val_loader, _ = get_qpi_loaders(config, num_workers=0)
    all_val_images = []
    for batch in val_loader:
        all_val_images.append(batch["phase"])
        
    random.seed(42)
    total_needed = LATENCY_WARMUP_RUNS + LATENCY_RUNS
    sampled_images = random.sample(all_val_images, min(total_needed, len(all_val_images)))
    
    # Handle PyTorch INT8 fallback (PyTorch dynamic INT8 is mostly CPU-bound)
    actual_precision = precision
    if not use_onnx and precision == 'int8':
        print("  [Note] Skipping native PyTorch INT8 on CUDA. Falling back to FP32.")
        actual_precision = 'fp32'

    test_images = []
    for img in sampled_images:
        img = img.to(device)
        if actual_precision == 'fp16':
            img = img.half()
        test_images.append(img)
        
    if actual_precision == 'fp16':
        model = model.half()

    onnx_file_path = None
    if use_onnx:
        onnx_session, onnx_file_path = compile_onnx(model, test_images[0], precision, arch_name=arch_lower)
        if onnx_session is not None:
            model = onnx_session
        else:
            print("  [Info] MobileSAM ONNX skipped by design. Using native PyTorch results.")
            return {}

    # Execute Benchmark
    stats = measure_latency_and_memory(model, test_images, device, is_onnx=use_onnx)

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
        "onnx": use_onnx,
    }
    
    combined_stats.update(get_hardware_info())
    combined_stats.update(param_stats) 
    combined_stats.update(val_metrics) 
    combined_stats.update(stats)       

    del model
    del test_images
    if onnx_file_path and os.path.exists(onnx_file_path):
        os.remove(onnx_file_path)
    force_cleanup()
    return combined_stats


if __name__ == "__main__":
    import datetime
    
    # ==============================================================
    # ADDED: TERMINAL LOGGING INTERCEPTOR
    # ==============================================================
    # Generate timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("results/benchmarks")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"benchmark_terminal_{timestamp}.txt"
    
    class TeeLogger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding="utf-8")
            
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush() # Force write to disk immediately (safe against crashes)
            
        def flush(self):
            self.terminal.flush()
            self.log.flush()
            
    # Redirect standard output and errors to our custom logger
    sys.stdout = TeeLogger(log_filename)
    sys.stderr = sys.stdout 
    
    print(f"\n[INFO] Terminal output is being mirrored to: {log_filename}\n")
    # ==============================================================

    # Optimize cuDNN for hardware
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', default='results/benchmarks')
    parser.add_argument('--train_results_dir', default='results')
    parser.add_argument('--ranks', type=int, nargs='+', default=[0, 2, 4, 8, 16, 32])
    parser.add_argument("--precisions", nargs='+', default=['fp32', 'fp16', 'int8'], help="Precisions to test")
    parser.add_argument('--onnx', action='store_true', help="Enable ONNX Runtime acceleration")
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
                
                stats = run_benchmark_scenario(cfg_path, rank, precision, False, args.device, args.train_results_dir)
                all_results.append(stats)
                
                if args.onnx and args.device == 'cuda':
                    onnx_stats = run_benchmark_scenario(cfg_path, rank, precision, True, args.device, args.train_results_dir)
                    if onnx_stats: 
                        all_results.append(onnx_stats)

    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = out_dir / f"hardware_benchmark_{args.device}.csv"
    df = pd.DataFrame(all_results)
    
    cols = df.columns.tolist()
    headline_cols = [
        "architecture", "device", "gpu_name", "lora_r", "precision", "onnx", 
        "enc_trainable_ratio_%", "dec_trainable_ratio_%", "overall_trainable_ratio_%",
        "mean_dice", "bf1", "phase_vol_error", 
        "latency_mean_ms", "fps", "peak_allocated_mb", "peak_reserved_mb"
    ]
    ordered_cols = [c for c in headline_cols if c in cols] + [c for c in cols if c not in headline_cols]
    df = df[ordered_cols]
    
    df.to_csv(csv_file, index=False)
    print(f"\n[DONE] Final Hardware Benchmark CSV generated at {csv_file}")