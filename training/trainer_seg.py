# --- Inside trainer_seg.py / train() method ---
            
            # Log
            self.metrics.track_epoch_metrics(
                epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                val_metrics=val_metrics # FIXED: Was missing, causing empty history
            )

            dice = val_metrics.get("dice", 0.0)
            is_best = dice > self.best_dice
            if is_best:
                self.best_dice = dice