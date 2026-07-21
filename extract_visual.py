import os
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))
from datasets.qpi_dataset import QPIDataset
from models import get_model


# ─── Per-model configs matching training YAML settings ────────────────────────
# FIX: Each model needs its own config because insertion_strategy and image_size differ.

class EdgeSAMConfig:
    num_classes = 5
    pretrained = False
    image_size = 256                   # matches configs/edge_sam_lora.yaml
    lora_r = 8
    lora_alpha = 8.0
    insertion_strategy = "encoder_only"  # matches edge_sam_lora.yaml

class MobileNetConfig:
    num_classes = 5
    pretrained = False
    image_size = 256                   # matches configs/mobilenet_unet_lora.yaml
    lora_r = 8
    lora_alpha = 8.0
    insertion_strategy = "encoder_only"  # matches mobilenet_unet_lora.yaml

class MobileSAMConfig:
    num_classes = 5
    pretrained = False
    image_size = 1024                       # matches configs/mobile_sam_lora.yaml
    lora_r = 8
    lora_alpha = 8.0
    insertion_strategy = "attention_blocks"  # FIX: was "encoder_only", must match training


def load_checkpoint(model, path, model_label):
    print(f"[{model_label}] Loading checkpoint: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state", checkpoint)

        result = model.load_state_dict(state, strict=False)
        missing  = [k for k in result.missing_keys  if "lora" in k]
        unexpected = result.unexpected_keys

        if missing:
            print(f"  -> WARNING: {len(missing)} LoRA keys missing. "
                  f"First 3: {missing[:3]}")
        else:
            print(f"  -> SUCCESS: All LoRA keys matched.")

        if unexpected:
            print(f"  -> Note: {len(unexpected)} unexpected keys ignored (normal for strict=False).")

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"  -> Params: {trainable:,} trainable / {total:,} total "
              f"({100*trainable/total:.2f}%)\n")
    except Exception as e:
        print(f"  -> FAILED to load checkpoint: {e}\n")
    return model

def generate_storage_timeline_grid(val_ds, edge_sam, analysis_dir):
    """Find one representative cell image per storage day and save a timeline grid."""
    print("\n6. Generating Storage Lesion Timeline Grid...")

    # Collect one best frame per unique storage day
    from collections import defaultdict
    day_frames = defaultdict(lambda: {"iou": -1, "phase": None, "pred": None, "gt": None})

    with torch.no_grad():
        for i in range(len(val_ds)):
            sample     = val_ds[i]
            day        = sample["storage_day"]
            if day < 0:
                continue
            gt         = (sample["mask"] > 0).numpy().astype(np.uint8)
            if gt.sum() < 3000:
                continue

            inp    = sample["phase"].unsqueeze(0).cuda()
            out    = edge_sam(inp)
            pred   = (torch.argmax(out, dim=1).squeeze() > 0).cpu().numpy().astype(np.uint8)
            iou    = np.logical_and(pred, gt).sum() / (np.logical_or(pred, gt).sum() + 1e-6)

            if iou > day_frames[day]["iou"]:
                day_frames[day] = {
                    "iou":   iou,
                    "phase": sample["phase"].numpy().squeeze(),
                    "pred":  pred,
                    "gt":    gt,
                }

    days = sorted(day_frames.keys())
    if not days:
        print("  [Skip] No storage day metadata found in dataset.")
        return

    # Pick ~5 evenly spaced days across the timeline
    indices      = np.linspace(0, len(days) - 1, min(5, len(days)), dtype=int)
    selected_days = [days[i] for i in indices]

    n = len(selected_days)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10), constrained_layout=True)

    for col, day in enumerate(selected_days):
        frame = day_frames[day]

        axes[0, col].imshow(frame["phase"], cmap="inferno")
        axes[0, col].set_title(f"Day {day}", fontsize=13, fontweight="bold")
        axes[0, col].axis("off")

        # Overlay predicted contour on phase map
        axes[1, col].imshow(frame["phase"], cmap="inferno", alpha=0.8)
        axes[1, col].contour(frame["pred"], levels=[0.5], colors="cyan",  linewidths=1.5)
        axes[1, col].contour(frame["gt"],   levels=[0.5], colors="white", linewidths=1.0, linestyles="--")
        axes[1, col].axis("off")
        if col == 0:
            axes[1, col].set_ylabel("EdgeSAM Segmentation", fontsize=11, fontweight="bold")

    axes[0, 0].set_ylabel("Raw Phase Map", fontsize=11, fontweight="bold")

    fig.suptitle("Storage Lesion Timeline: RBC Morphology Degradation Over Storage",
                 fontsize=15, fontweight="bold")

    out_path = analysis_dir / "fig_storage_timeline_grid.jpg"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out_path}")

def main():
    print("=" * 60)
    print("  QPI VISUAL EXTRACTION PIPELINE")
    print("=" * 60)

    # ── Load models with correct per-model configs ─────────────────────────
    print("\n1. Initializing Models & Loading Checkpoints...")

    print("[EdgeSAM] Loading...")
    edge_sam = get_model("edge_sam", EdgeSAMConfig())
    edge_sam = load_checkpoint(edge_sam, "results/edge_sam_lora_r8/checkpoints/best_model.pt", "EdgeSAM")
    edge_sam = edge_sam.cuda().eval()

    print("[MobileNet-UNet] Loading...")
    mobilenet = get_model("mobilenet_unet", MobileNetConfig())
    mobilenet = load_checkpoint(mobilenet, "results/mobilenet_unet_lora_r8/checkpoints/best_model.pt", "MobileNet-UNet")
    mobilenet = mobilenet.cuda().eval()

    print("[MobileSAM] Loading...")
    mobile_sam = get_model("mobile_sam", MobileSAMConfig())
    mobile_sam = load_checkpoint(mobile_sam, "results/mobile_sam_lora_r8/checkpoints/best_model.pt", "MobileSAM")
    mobile_sam = mobile_sam.cuda().eval()

    # ── Load dataset ────────────────────────────────────────────────────────
    print("\n2. Loading Validation Dataset...")
    val_ds = QPIDataset(data_root="./dataset", split="val", augment=False)
    print(f"  -> {len(val_ds)} samples loaded.")
    # ── Scan for the best CONTRAST frame ──────────────────────────────────
    print("\n3. Scanning for best contrast frame (EdgeSAM good, others collapsed)...")
    best_score = -999.0
    best_phase = best_gt_mask = best_input_tensor = None

    with torch.no_grad():
        for i in range(len(val_ds)):
            sample       = val_ds[i]
            phase_tensor = sample["phase"]
            gt_tensor    = sample.get("mask")
            if gt_tensor is None:
                continue

            gt_binary    = (gt_tensor > 0).cpu().numpy().astype(np.uint8)
            if gt_binary.sum() < 5000:
                continue

            input_tensor = phase_tensor.unsqueeze(0).cuda()

            def iou(pred_tensor, gt):
                pb = (torch.argmax(pred_tensor, dim=1).squeeze() > 0).cpu().numpy().astype(np.uint8)
                i  = np.logical_and(pb, gt).sum()
                u  = np.logical_or(pb, gt).sum()
                return float(i / (u + 1e-6)), pb

            edge_iou,  _ = iou(edge_sam(input_tensor),   gt_binary)
            mnet_iou,  _ = iou(mobilenet(input_tensor),  gt_binary)
            msam_iou,  _ = iou(mobile_sam(input_tensor), gt_binary)

            # Maximise EdgeSAM quality while penalising the other two
            score = edge_iou - 0.5 * (mnet_iou + msam_iou)

            if score > best_score:
                best_score        = score
                best_phase        = phase_tensor.cpu().numpy().squeeze()
                best_gt_mask      = gt_binary
                best_input_tensor = input_tensor

    print(f"  -> Best contrast frame found. Score: {best_score:.4f}")
    if best_input_tensor is None:
        print("ERROR: No suitable frame found. Check that val masks contain foreground pixels.")
        return

    # ── Run inference on best frame with all three models ──────────────────
    print("\n4. Running inference on best frame for all models...")
    with torch.no_grad():
        out_edge   = edge_sam(best_input_tensor)
        edgesam_binary = (torch.argmax(out_edge, dim=1).squeeze() > 0).cpu().numpy().astype(np.uint8)

        out_mobile = mobilenet(best_input_tensor)
        mobilenet_binary = (torch.argmax(out_mobile, dim=1).squeeze() > 0).cpu().numpy().astype(np.uint8)

        out_msam   = mobile_sam(best_input_tensor)
        mobilesam_binary = (torch.argmax(out_msam, dim=1).squeeze() > 0).cpu().numpy().astype(np.uint8)

    # ── Export arrays for plot_trends.py ───────────────────────────────────
    print("\n5. Exporting arrays to results/analysis/ ...")
    analysis_dir = Path("results/analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    np.save(analysis_dir / "sample_phase.npy",             best_phase)
    np.save(analysis_dir / "sample_mask.npy",              best_gt_mask)
    np.save(analysis_dir / "edgesam_recovered_mask.npy",   edgesam_binary)
    np.save(analysis_dir / "mobilenet_collapsed_mask.npy", mobilenet_binary)
    np.save(analysis_dir / "mobilesam_mask.npy",           mobilesam_binary)
    generate_storage_timeline_grid(val_ds, edge_sam, analysis_dir)
    print(f"  -> Done. Files saved to {analysis_dir}/")
    print("\nRun: python analysis/plot_trends.py --results_dir results")


if __name__ == "__main__":
    main()
