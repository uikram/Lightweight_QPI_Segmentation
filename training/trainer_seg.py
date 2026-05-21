"""
Segmentation Trainer for Physics-Aware QPI Holographic Cell Analysis.
Handles: mixed-precision training, LoRA-aware optimizer, physics-aware loss,
         best-model checkpointing, and per-epoch metric logging.
"""

import os
import torch
from tqdm import tqdm
from pathlib import Path

from training.losses import get_loss
from datasets.qpi_dataset import get_qpi_loaders
from evaluation.seg_metrics import SegmentationMetrics


class SegmentationTrainer:
    def __init__(self, model, config, metrics_tracker):
        self.model   = model
        self.config  = config
        self.metrics = metrics_tracker
        self.device  = config.device

        # ── Data ──────────────────────────────────────────────────────────────
        self.train_loader, self.val_loader, _ = get_qpi_loaders(
            config, num_workers=getattr(config, 'num_workers', 4)
        )

        # ── Optimizer ─────────────────────────────────────────────────────────
        # Only optimise parameters that require gradients.
        # After LoRA injection this means only the low-rank A/B matrices;
        # before injection it means all parameters (full fine-tune).
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError(
                "No trainable parameters found. "
                "If LoRA was injected, check that lora_r is set in the config."
            )
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=getattr(config, 'learning_rate', 0.001),
        )

        # ── LR Scheduler (cosine annealing as per research plan) ──────────────
        epochs = getattr(config, 'epochs', 50)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

        # ── Loss ──────────────────────────────────────────────────────────────
        self.criterion = get_loss(config).to(self.device)

        # ── Mixed precision ───────────────────────────────────────────────────
        use_fp16 = getattr(config, 'mixed_precision', 'fp16') == 'fp16'
        
        # Dynamically handle local architecture (MPS) vs Server GPU (CUDA)
        device_type = 'cuda' if 'cuda' in str(self.device) else ('mps' if 'mps' in str(self.device) else 'cpu')
        
        try:
            self.scaler = torch.amp.GradScaler(device_type, enabled=use_fp16)
            self._autocast = lambda: torch.amp.autocast(device_type, enabled=use_fp16)
        except (TypeError, AttributeError):
            # PyTorch < 2.0 fallback (assumes CUDA)
            self.scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
            self._autocast = lambda: torch.cuda.amp.autocast(enabled=use_fp16)

        # ── Checkpointing ─────────────────────────────────────────────────────
        # Accept either an explicit checkpoint_dir or fall back to results_dir.
        # Both default to './results' if absent from config so training never
        # crashes on a missing YAML key.
        ckpt_dir = getattr(
            config,
            'checkpoint_dir',
            getattr(config, 'results_dir', Path('results'))
        )
        self.checkpoint_dir = Path(ckpt_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.epochs    = epochs
        self.best_dice = 0.0

        # ── Metrics tracker ───────────────────────────────────────────────────
        num_classes = getattr(config, 'num_classes', 5)
        self.seg_metrics = SegmentationMetrics(num_classes=num_classes)

    # ──────────────────────────────────────────────────────────────────────────
    def train(self):
        self.metrics.track_parameters(self.model)

        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}  "
                  f"lr={self.optimizer.param_groups[0]['lr']:.2e}")

            train_loss           = self._train_epoch()
            val_loss, val_metrics = self._val_epoch()

            self.scheduler.step()

            self.metrics.track_epoch_metrics(
                epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                val_metrics=val_metrics,
            )

            dice   = val_metrics.get("mean_dice", 0.0)
            is_best = dice > self.best_dice
            if is_best:
                self.best_dice = dice
                ckpt_path = self.checkpoint_dir / "best_model.pt"
                torch.save(
                    {"epoch": epoch + 1,
                     "model_state": self.model.state_dict(),
                     "dice": self.best_dice},
                    ckpt_path,
                )
                print(f"  -> Best model saved (Dice: {self.best_dice:.4f})")


    # ──────────────────────────────────────────────────────────────────────────
    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc="  Train", leave=False)

        for batch in pbar:
            images    = batch["phase"].to(self.device)       # (B, 1, H, W)
            targets   = batch["mask"].to(self.device)        # (B, H, W) long 0-4
            phase_raw = batch.get("phase_raw", images).to(self.device)

            self.optimizer.zero_grad()

            with self._autocast():
                logits = self.model(images)                  # (B, C, H, W)
                loss   = self.criterion(logits, targets, phase_raw)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    # ──────────────────────────────────────────────────────────────────────────
    def _val_epoch(self):
        self.model.eval()
        self.seg_metrics.reset()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="  Val  ", leave=False):
                images    = batch["phase"].to(self.device)
                targets   = batch["mask"].to(self.device)
                phase_raw = batch.get("phase_raw", images).to(self.device)

                with self._autocast():
                    logits = self.model(images)
                    loss   = self.criterion(logits, targets, phase_raw)

                total_loss += loss.item()

                # Physics metrics and binarization are now handled internally
                pred_classes = torch.argmax(logits, dim=1)     # (B, H, W)
                
                # Pass the RAW multi-class tensors (0-4) so the tracker can score morphology!
                self.seg_metrics.update(pred_classes, targets, phase_raw)

        val_loss    = total_loss / len(self.val_loader)
        val_metrics = self.seg_metrics.compute()
        self.seg_metrics.print_results(val_metrics, prefix="Val ")
        return val_loss, val_metrics
