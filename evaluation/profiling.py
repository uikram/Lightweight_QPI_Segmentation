"""
Advanced profiling utilities for model performance analysis.
Compatible with main.py --mode benchmark.
"""

import time
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

class ModelProfiler:
    """
    Profiler for measuring Latency, Throughput, and Peak Memory.
    """
    
    def __init__(self, model, model_name, config):
        self.model = model
        self.model_name = model_name
        self.config = config
        self.device = config.device
        self.output_dir = Path(config.results_dir) / 'benchmarks'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def profile(self, dataloader, num_samples=100):
        """
        Run profiling on a dataloader.
        """
        print(f"Profiling {self.model_name} on {self.device.upper()}...")
        self.model.eval()
        self.model.to(self.device)
        
        latencies = []
        peak_memory = 0.0
        
        # Create an iterator that loops if dataset is too small
        def infinite_loader():
            while True:
                for batch in dataloader:
                    yield batch
        
        batch_iter = infinite_loader()
        
        # 1. Warmup
        print("Warming up...")
        with torch.no_grad():
            for _ in range(5):
                batch = next(batch_iter)
                self._run_forward(batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        
        # 2. Measure Memory (Reset before run)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
        # 3. Profile Loop
        print(f"  Measuring metrics over {num_samples} samples...")
        pbar = tqdm(total=num_samples, desc="Profiling")
        
        current_samples = 0
        with torch.no_grad():
            while current_samples < num_samples:
                batch = next(batch_iter)
                batch_size = self._get_batch_size(batch)
                
                # Timer start
                if torch.cuda.is_available(): torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                self._run_forward(batch)
                
                # Timer end
                if torch.cuda.is_available(): torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                # Record Latency (ms)
                latencies.append((end_time - start_time) * 1000)
                
                current_samples += batch_size
                pbar.update(batch_size)
                
                if current_samples >= num_samples:
                    break
        
        pbar.close()
        
        # 4. Capture Peak Memory
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB
        
        # Calculate Stats
        avg_latency = np.mean(latencies)
        throughput = (1000.0 / avg_latency) * self.config.batch_size # images/sec
        
        results = {
            "model": self.model_name,
            "device": self.device,
            "samples": current_samples,
            "latency_avg_ms": round(avg_latency, 2),
            "latency_p95_ms": round(np.percentile(latencies, 95), 2),
            "throughput_imgs_sec": round(throughput, 2),
            "peak_memory_mb": round(peak_memory, 2)
        }
        
        return results

    def _get_batch_size(self, batch):
        """Helper to get batch size from dict or list."""
        if isinstance(batch, dict):
            # Try common keys
            for k in ['images', 'pixel_values', 'input_ids']:
                if k in batch: return batch[k].size(0)
            return list(batch.values())[0].size(0)
        elif isinstance(batch, (list, tuple)):
            return batch[0].size(0)
        return 1

    def _run_forward(self, batch):
        """Handle moving to device and forward pass."""
        # Move to device
        if isinstance(batch, dict):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            batch = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch]
        else:
            batch = batch.to(self.device)

        # Forward
        if isinstance(batch, dict):
            # FrozenCLIP might expect specific args
            if hasattr(self.model, 'vision_encoder') and 'images' in batch:
                self.model(
                    images=batch['images'], 
                    input_ids=batch.get('input_ids'), 
                    attention_mask=batch.get('attention_mask'),
                    labels=batch.get('labels')
                )
            else:
                self.model(**batch)
        elif hasattr(self.model, 'encode_image'):
             # CLIP models
            if isinstance(batch, (list, tuple)):
                 self.model.encode_image(batch[0])
            else:
                 self.model.encode_image(batch)
        else:
            # Fallback
            if isinstance(batch, (list, tuple)):
                self.model(*batch)
            else:
                self.model(batch)

    def save_results(self, results):
        """Save results to JSON."""
        filename = f"{self.model_name}_benchmark.json"
        save_path = self.output_dir / filename
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {save_path}")

    def print_summary(self, results):
        """Print pretty summary."""
        print("\n" + "="*40)
        print(f"BENCHMARK RESULTS: {results['model']}")
        print("="*40)
        print(f"Latency (Avg):    {results['latency_avg_ms']} ms")
        print(f"Throughput:       {results['throughput_imgs_sec']} img/sec")
        print(f"Peak Memory:      {results['peak_memory_mb']} MB")
        print("="*40 + "\n")