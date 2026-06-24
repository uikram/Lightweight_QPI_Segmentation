"""
Metrics tracker for QPI segmentation experiments.
Tracks segmentation accuracy, phase preservation, and training history.
"""

import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import numpy as np


class MetricsTracker:
    def __init__(self, model_name: str, results_dir: Path):
        self.model_name  = model_name.upper()
        self.results_dir = Path(results_dir) / self.model_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {
            "model_name":         self.model_name,
            "timestamp":          datetime.now().isoformat(),
            "parameters":         {},
            "training_history":   {},
            "seg_metrics":        {},
            "phase_metrics":      {},
        }

        self.training_history = {
            "epochs":     [],
            "train_loss": [],
            "val_loss":   [],
            "train_L_dice": [],
            "train_L_pmc": [],
            "train_L_bga": [],
            "train_L_pv": [],
            "val_dice":   [],
            "val_aji":    [],
            "val_bf1":    [],
            "val_phase_vol_error": [],
        }

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

    def track_epoch_metrics(self, epoch: int, train_loss: float = None,
                            val_loss: float = None, val_metrics: dict = None,
                            train_loss_components: dict = None):
        self.training_history["epochs"].append(int(epoch))

        if train_loss is not None:
            self.training_history["train_loss"].append(float(train_loss))
        if val_loss is not None:
            self.training_history["val_loss"].append(float(val_loss))

        # Log individual physics loss components
        if train_loss_components is not None:
            self.training_history["train_L_dice"].append(float(train_loss_components.get("L_dice", 0.0)))
            self.training_history["train_L_pmc"].append(float(train_loss_components.get("L_pmc", 0.0)))
            self.training_history["train_L_bga"].append(float(train_loss_components.get("L_bga", 0.0)))
            self.training_history["train_L_pv"].append(float(train_loss_components.get("L_pv", 0.0)))

        if val_metrics is not None:
            self.training_history["val_dice"].append(float(val_metrics.get("mean_dice", 0.0)))
            self.training_history["val_aji"].append(float(val_metrics.get("aji", 0.0)))
            self.training_history["val_bf1"].append(float(val_metrics.get("bf1", 0.0)))
            self.training_history["val_phase_vol_error"].append(float(val_metrics.get("phase_vol_error", 0.0)))

        self.metrics["training_history"] = self.training_history

    def track_seg_metrics(self, split: str, metrics: dict):
        self.metrics["seg_metrics"][split] = metrics
        print(f"\n[{split.upper()} Metrics]")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    def track_phase_metrics(self, split: str, phase_vol_error: float,
                            phase_vol_error_std: float = 0.0):
        self.metrics["phase_metrics"][split] = {
            "phase_vol_error":     phase_vol_error,
            "phase_vol_error_std": phase_vol_error_std,
        }

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

        print(f"\n{'='*60}")