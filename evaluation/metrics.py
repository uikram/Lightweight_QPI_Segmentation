"""
Metrics tracker for QPI segmentation experiments.
Tracks segmentation accuracy, quantitative phase preservation,
edge deployment metrics, and training history.
"""

import json
import time
import torch
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class MetricsTracker:
    """Unified metrics tracking for QPI segmentation models."""

    def __init__(self, model_name: str, results_dir: Path):
        self.model_name  = model_name.upper()
        self.results_dir = Path(results_dir) / self.model_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {
            "model_name":         self.model_name,
            "timestamp":          datetime.now().isoformat(),
            "parameters":         {},
            "memory":             {},
            "latency":            {},
            "training_time":      {},
            "evaluation_results": {},
            "training_history":   {},
            "seg_metrics":        {},
            "phase_metrics":      {},
            "edge_deployment":    {},
        }

        self.training_history = {
            "epochs":     [],
            "train_loss": [],
            "val_loss":   [],
            "val_dice":   [],
            "val_aji":    [],
            "val_bf1":    [],
            "val_phase_vol_error": [],
        }

        self.train_start_time = None

    # ------------------------------------------------------------------
    # Parameter tracking
    # ------------------------------------------------------------------

    def track_parameters(self, model):
        if hasattr(model, "count_parameters"):
            counts    = model.count_parameters()
            total     = counts["total"]
            trainable = counts["trainable"]
        else:
            total     = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        frozen = total - trainable
        self.metrics["parameters"] = {
            "total_parameters":      int(total),
            "trainable_parameters":  int(trainable),
            "frozen_parameters":     int(frozen),
            "trainable_percentage":  round(100 * trainable / total, 4) if total > 0 else 0.0,
        }
        print(f"\n[Params] Total: {total:,} | Trainable: {trainable:,} "
              f"({self.metrics['parameters']['trainable_percentage']:.2f}%)")

    # ------------------------------------------------------------------
    # Epoch metrics
    # ------------------------------------------------------------------

    def track_epoch_metrics(self, epoch: int, train_loss: float = None,
                            val_loss: float = None, val_metrics: dict = None,
                            train_accuracy: float = None, val_accuracy: float = None):
        self.training_history["epochs"].append(int(epoch))

        if train_loss is not None:
            self.training_history["train_loss"].append(float(train_loss))
        if val_loss is not None:
            self.training_history["val_loss"].append(float(val_loss))

        if val_metrics is not None:
            self.training_history["val_dice"].append(
                float(val_metrics.get("dice", 0.0))
            )
            self.training_history["val_aji"].append(
                float(val_metrics.get("aji", 0.0))
            )
            self.training_history["val_bf1"].append(
                float(val_metrics.get("bf1", 0.0))
            )
            self.training_history["val_phase_vol_error"].append(
                float(val_metrics.get("phase_vol_error", 0.0))
            )

        self.metrics["training_history"] = self.training_history

    # ------------------------------------------------------------------
    # Segmentation metrics
    # ------------------------------------------------------------------

    def track_seg_metrics(self, split: str, metrics: dict):
        """Track final segmentation metrics for a split."""
        self.metrics["seg_metrics"][split] = metrics
        print(f"\n[{split.upper()} Metrics]")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    def track_phase_metrics(self, split: str, phase_vol_error: float,
                            phase_vol_error_std: float = 0.0):
        """Track quantitative phase preservation metrics."""
        self.metrics["phase_metrics"][split] = {
            "phase_vol_error":     phase_vol_error,
            "phase_vol_error_std": phase_vol_error_std,
        }

    def track_ablation_results(self, ablation_name: str, results: dict):
        """Track ablation study results (loss components, rank sweep, etc.)."""
        if "ablation" not in self.metrics:
            self.metrics["ablation"] = {}
        self.metrics["ablation"][ablation_name] = results

    # ------------------------------------------------------------------
    # Edge deployment metrics
    # ------------------------------------------------------------------

    def track_edge_deployment(self, latency_results: dict,
                               battery_capacity_J: float = 39960.0,
                               inference_rate_hz: float = 10.0):
        """
        Track edge deployment feasibility metrics.
        Compatible with the existing benchmark/run_benchmarking.py format.
        """
        mean_ms  = latency_results.get("mean_ms", 0.0)
        p99_ms   = latency_results.get("p99_ms",  0.0)
        energy_J = latency_results.get("energy_J", 0.0)

        wcl = p99_ms + 1.0 + 2.0  # p99 + OS jitter + buffer
        tau_max_ms = 37.77          # from SiL simulation

        inferences_per_charge = battery_capacity_J / energy_J if energy_J > 0 else 0
        runtime_hr = inferences_per_charge / inference_rate_hz / 3600

        self.metrics["edge_deployment"] = {
            "latency":           latency_results,
            "worst_case_ms":     wcl,
            "satisfies_rt":      bool(wcl <= 1000.0 / inference_rate_hz),
            "satisfies_stability": bool(wcl < tau_max_ms),
            "stability_margin_ms": round(tau_max_ms - wcl, 2),
            "inferences_per_charge": inferences_per_charge,
            "runtime_hr":        runtime_hr,
            "runtime_min":       runtime_hr * 60,
        }

    # ------------------------------------------------------------------
    # Timers
    # ------------------------------------------------------------------

    def start_training_timer(self):
        self.train_start_time = time.time()

    def end_training_timer(self):
        if self.train_start_time is None:
            return
        total_s = time.time() - self.train_start_time
        self.metrics["training_time"] = {
            "total_seconds": round(total_s, 2),
            "total_minutes": round(total_s / 60, 2),
            "total_hours":   round(total_s / 3600, 4),
        }

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def track_gpu_memory(self, stage: str = "training"):
        if torch.cuda.is_available():
            current  = torch.cuda.memory_allocated() / (1024 ** 2)
            peak     = torch.cuda.max_memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            self.metrics["memory"][f"{stage}_current_gpu_mb"] = round(current, 2)
            self.metrics["memory"][f"{stage}_peak_gpu_mb"]    = round(peak, 2)
            self.metrics["memory"][f"{stage}_reserved_gpu_mb"] = round(reserved, 2)
            print(f"[Memory/{stage}] Current: {current:.0f} MB | Peak: {peak:.0f} MB")
        else:
            self.metrics["memory"][f"{stage}_gpu_mb"] = "N/A"

    # ------------------------------------------------------------------
    # Evaluation results (kept for compatibility)
    # ------------------------------------------------------------------

    def track_evaluation_results(self, dataset_name: str, task: str,
                                  results: Dict[str, Any]):
        if "evaluation_results" not in self.metrics:
            self.metrics["evaluation_results"] = {}
        if dataset_name not in self.metrics["evaluation_results"]:
            self.metrics["evaluation_results"][dataset_name] = {}
        self.metrics["evaluation_results"][dataset_name][task] = results

    # ------------------------------------------------------------------
    # Save / print
    # ------------------------------------------------------------------

    def save_metrics(self, filename: str = "metrics.json"):
        self.metrics["training_history"] = self.training_history
        filepath = self.results_dir / filename
        with open(filepath, "w") as f:
            json.dump(self.metrics, f, indent=4, default=str)
        print(f"\n[Metrics] Saved to {filepath}")

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"METRICS SUMMARY: {self.model_name}")
        print(f"{'='*60}")

        p = self.metrics.get("parameters", {})
        if p:
            print(f"\nParameters:")
            print(f"  Total:     {p.get('total_parameters', 0):,}")
            print(f"  Trainable: {p.get('trainable_parameters', 0):,} "
                  f"({p.get('trainable_percentage', 0):.2f}%)")

        seg = self.metrics.get("seg_metrics", {})
        for split, vals in seg.items():
            print(f"\n{split.upper()} Segmentation:")
            for k, v in vals.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")

        edge = self.metrics.get("edge_deployment", {})
        if edge:
            print(f"\nEdge Deployment:")
            print(f"  WCL:             {edge.get('worst_case_ms', 0):.2f} ms")
            print(f"  RT satisfied:    {edge.get('satisfies_rt', False)}")
            print(f"  Runtime:         {edge.get('runtime_min', 0):.1f} min/charge")

        print(f"\n{'='*60}")