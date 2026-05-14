"""
Segmentation trainer for QPI holographic cell analysis.
Handles all three architectures: MobileNet-UNet, MobileSAM, EdgeSAM.
Supports LoRA and full fine-tuning modes.
Implements rank sweep experiments (r = 2, 4, 8, 16).
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
import json
import time

from training.losses import PhysicsAwarePhaseLoss, get_loss
from datasets.qpi_dataset import get_qpi_loaders
from evaluation.seg_metrics import SegmentationMetrics


class SegmentationTrainer:
    """
    Trainer for QPI segmentation models with physics-aware loss.

    Supports:
        - LoRA adaptation (any rank)
        - Full fine-tuning
        - Mixed precision (FP16)
        - Gradient accumulation
        - Physics-aware loss with component logging
        - Automatic checkpoint saving
    """

    def __init__(self, model: nn.Module, config, metrics_tracker):
        self.model   = model
        self.config  = config
        self.metrics = metrics_tracker
        self.device  = config.device

        self.model.to(self.device)

        # --- Optimizer ---
        # Only optimize trainable parameters (LoRA or full)
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            # No LoRA injected — full fine-tuning
            print("[Trainer] No LoRA detected. Training all parameters.")
            for p in model.parameters():
                p.requires_grad = True
            trainable = list(model.parameters())

        no_decay = ["bias", "LayerNorm.weight", "BatchNorm2d.weight",
                    "bn.weight", "bn.bias"]

        param_groups = [
            {
                "params": [p for n, p in model.named_parameters()
                           if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": getattr(config, "weight_decay", 1e-4),
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(param_groups, lr=getattr(config, "learning_rate", 1e-4))

        # --- Scheduler ---
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=getattr(config, "num_epochs", 50),
            eta_min=getattr(config, "learning_rate", 1e-4) * 0.01,
        )

        # --- Loss ---
        self.criterion = get_loss(config)

        # --- Mixed precision ---
        self.use_fp16 = getattr(config, "fp16", True)
        self.scaler   = GradScaler() if self.use_fp16 else None

        # --- Gradient accumulation ---
        self.grad_accum = getattr(config, "gradient_accumulation_steps", 1)

        # --- State ---
        self.global_step   = 0
        self.best_dice     = 0.0
        self.best_val_loss = float("inf")

        # --- Output dirs ---
        self.checkpoint_dir = Path(getattr(config, "checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Trainer] Model: {type(model).__name__} | "
              f"Device: {self.device} | FP16: {self.use_fp16}")
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_count     = sum(p.numel() for p in model.parameters())
        print(f"[Trainer] Params: {trainable_count:,} trainable / {total_count:,} total "
              f"({100 * trainable_count / total_count:.2f}%)")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        print(f"\n{'='*60}")
        print("QPI SEGMENTATION TRAINING")
        print(f"{'='*60}")

        train_loader, val_loader, _ = get_qpi_loaders(
            self.config, num_workers=getattr(self.config, "num_workers", 4)
        )

        print(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

        self.metrics.track_parameters(self.model)
        self.metrics.track_gpu_memory("pre_training")
        self.metrics.start_training_timer()

        num_epochs = getattr(self.config, "num_epochs", 50)

        for epoch in range(num_epochs):
            train_loss, loss_components = self.train_epoch(epoch, train_loader)
            val_loss, val_metrics       = self.validate(val_loader)

            self.scheduler.step()

            # Log
            self.metrics.track_epoch_metrics(
                epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
            )

            dice = val_metrics.get("dice", 0.0)
            is_best = dice > self.best_dice
            if is_best:
                self.best_dice = dice

            self.save_checkpoint(epoch, val_loss, val_metrics, is_best=is_best)

            print(f"\n[Epoch {epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Dice: {dice:.4f} | "
                  f"Best Dice: {self.best_dice:.4f}")
            print(f"  Loss components: "
                  + " | ".join(f"{k}: {v:.4f}" for k, v in loss_components.items()))

        self.metrics.end_training_timer()
        self.metrics.track_gpu_memory("post_training")
        print("\nTraining complete.")

    # ------------------------------------------------------------------
    # Epoch
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int, loader):
        self.model.train()
        total_loss = 0.0
        avg_components = {}
        n_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", dynamic_ncols=True)

        for step, batch in enumerate(pbar):
            loss, components = self._train_step(batch, step)

            total_loss += loss
            for k, v in components.items():
                avg_components[k] = avg_components.get(k, 0.0) + v
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "dice": f"{components.get('L_dice', 0):.4f}",
                "pmc":  f"{components.get('L_pmc', 0):.4f}",
            })

        avg_loss = total_loss / max(n_batches, 1)
        avg_comp = {k: v / max(n_batches, 1) for k, v in avg_components.items()}
        return avg_loss, avg_comp

    def _train_step(self, batch, step):
        phase     = batch["phase"].to(self.device)
        mask      = batch["mask"].to(self.device)
        phase_raw = batch.get("phase_raw", phase).to(self.device)

        if self.use_fp16:
            with autocast(dtype=torch.float16):
                pred = self.model(phase)
                loss = self.criterion(pred, mask, phase_raw)
                loss = loss / self.grad_accum
            self.scaler.scale(loss).backward()
        else:
            pred = self.model(phase)
            loss = self.criterion(pred, mask, phase_raw)
            loss = loss / self.grad_accum
            loss.backward()

        if (step + 1) % self.grad_accum == 0:
            if self.use_fp16:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1

        components = self.criterion.get_loss_components(pred.detach(), mask, phase_raw) \
            if hasattr(self.criterion, "get_loss_components") else {}

        return loss.item() * self.grad_accum, components

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, loader):
        self.model.eval()
        seg_metrics = SegmentationMetrics(num_classes=1)
        total_loss  = 0.0
        n_batches   = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validating", leave=False):
                phase     = batch["phase"].to(self.device)
                mask      = batch["mask"].to(self.device)
                phase_raw = batch.get("phase_raw", phase).to(self.device)

                if self.use_fp16:
                    with autocast(dtype=torch.float16):
                        pred = self.model(phase)
                        loss = self.criterion(pred, mask, phase_raw)
                else:
                    pred = self.model(phase)
                    loss = self.criterion(pred, mask, phase_raw)

                total_loss += loss.item()
                n_batches  += 1

                pred_binary = (torch.sigmoid(pred.squeeze(1)) > 0.5).long()
                seg_metrics.update(pred_binary, mask, phase_raw.squeeze(1))

        val_loss    = total_loss / max(n_batches, 1)
        val_metrics = seg_metrics.compute()
        return val_loss, val_metrics

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, val_loss: float,
                        val_metrics: dict, is_best: bool = False):
        checkpoint = {
            "epoch":        epoch,
            "global_step":  self.global_step,
            "model_state":  self.model.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "scheduler":    self.scheduler.state_dict(),
            "val_loss":     val_loss,
            "val_metrics":  val_metrics,
            "best_dice":    self.best_dice,
            "config":       {k: str(v) for k, v in vars(self.config).items()},
        }

        epoch_path = self.checkpoint_dir / f"epoch_{epoch+1}.pt"
        torch.save(checkpoint, epoch_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  ★ New best: Dice={self.best_dice:.4f} saved to {best_path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.best_dice = ckpt.get("best_dice", 0.0)
        print(f"[Trainer] Loaded checkpoint from {path} "
              f"(epoch={ckpt.get('epoch', '?')}, "
              f"best_dice={self.best_dice:.4f})")


# ------------------------------------------------------------------
# LoRA rank sweep experiment
# ------------------------------------------------------------------

def run_rank_sweep(model_class, config, metrics_tracker,
                   ranks=(2, 4, 8, 16),
                   strategy: str = "attention_blocks"):
    """
    Ablation: train the same model with different LoRA ranks.
    Reports Dice, AJI, latency, and param count per rank.

    Usage:
        from models import MobileNetUNet
        results = run_rank_sweep(MobileNetUNet, config, metrics)
    """
    import copy
    results = {}

    for r in ranks:
        print(f"\n{'='*60}")
        print(f"RANK SWEEP: r = {r}")
        print(f"{'='*60}")

        cfg      = copy.deepcopy(config)
        cfg.lora_r = r

        model = model_class(
            num_classes=getattr(config, "num_classes", 1),
            pretrained=getattr(config, "pretrained", True),
        )
        model.inject_lora(r=r, lora_alpha=float(r),
                          lora_dropout=getattr(config, "lora_dropout", 0.0),
                          strategy=strategy)

        trainer = SegmentationTrainer(model, cfg, metrics_tracker)
        trainer.train()

        # Measure latency after merging
        model.merge_lora()
        latency_ms = _measure_latency(model, config)

        results[f"r={r}"] = {
            "trainable_params": sum(p.numel() for p in model.parameters()
                                    if p.requires_grad),
            "best_dice":        trainer.best_dice,
            "latency_ms":       latency_ms,
        }

    print("\n[Rank Sweep Results]")
    for k, v in results.items():
        print(f"  {k}: Dice={v['best_dice']:.4f} | "
              f"Latency={v['latency_ms']:.2f}ms | "
              f"Params={v['trainable_params']:,}")

    return results


def _measure_latency(model, config, n_runs: int = 100) -> float:
    """Quick latency measurement for rank sweep reporting."""
    import time
    device = config.device
    H = W  = getattr(config, "image_size", 256)

    model.eval().to(device)
    dummy = torch.randn(1, 1, H, W, device=device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(dummy)
    if device != "cpu":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            model(dummy)
    if device != "cpu":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_runs * 1000  # ms

    return elapsed