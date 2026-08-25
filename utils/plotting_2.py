"""

Visualization utilities for generating comparison plots and training curves.

"""



import csv

import json

import matplotlib.pyplot as plt

import numpy as np

from pathlib import Path

from matplotlib.lines import Line2D

from matplotlib.patches import Patch





def load_metrics(model_name: str, results_dir: Path, metric_type: str):

    """Load metrics from JSON file."""

    model_dir = results_dir / model_name

    

    # Load fixed filename (no timestamp)

    metric_file = model_dir / f"{metric_type}.json" 

    

    if not metric_file.exists():

        return None

    

    with open(metric_file, 'r') as f:

        return json.load(f)







def plot_training_curves(model_name: str, results_dir: Path, plots_dir: Path):

    """

    Generate training/validation loss and accuracy curves for a single model.

    

    Args:

        model_name: Name of the model (e.g., 'CLIP_LORA', 'FROZEN')

        results_dir: Directory containing results

        plots_dir: Directory to save plots

    """

    plots_dir.mkdir(parents=True, exist_ok=True)

    

    # Load training history

    history = load_metrics(model_name, results_dir, 'training_history')

    

    if not history:

        print(f"⚠️ No training history found for {model_name}")

        return

    

    epochs = history.get('epochs', [])

    train_loss = history.get('train_loss', [])

    val_loss = history.get('val_loss', [])

    train_acc = history.get('train_accuracy', [])

    val_acc = history.get('val_accuracy', [])

    

    if not epochs:

        print(f"⚠️ No epoch data found for {model_name}")

        return

    

    # Determine what we have

    has_loss = len(train_loss) > 0

    has_val_loss = len(val_loss) > 0

    has_accuracy = len(train_acc) > 0

    has_val_acc = len(val_acc) > 0

    

    # === PLOT 1: Loss Curves ===

    if has_loss or has_val_loss:

        fig, ax = plt.subplots(figsize=(10, 6))

        

        if has_loss:

            ax.plot(epochs[:len(train_loss)], train_loss, 

                   marker='o', linewidth=2, label='Training Loss', 

                   color='#2E86AB', markersize=6)

        

        if has_val_loss:

            ax.plot(epochs[:len(val_loss)], val_loss, 

                   marker='s', linewidth=2, label='Validation Loss', 

                   color='#A23B72', markersize=6)

        

        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')

        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')

        ax.set_title(f'{model_name} - Training Loss Curves', 

                    fontsize=14, fontweight='bold', pad=20)

        ax.legend(fontsize=11, framealpha=0.9)

        ax.grid(True, alpha=0.3, linestyle='--')

        ax.set_xlim(left=min(epochs) if epochs else 0)

        

        # Add min loss annotation

        if has_val_loss and val_loss:

            min_val_loss = min(val_loss)

            min_epoch = epochs[val_loss.index(min_val_loss)]

            ax.annotate(f'Min: {min_val_loss:.4f}',

                       xy=(min_epoch, min_val_loss),

                       xytext=(10, 10), textcoords='offset points',

                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),

                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        

        plt.tight_layout()

        plt.savefig(plots_dir / f'{model_name.lower()}_loss_curves.png', 

                   dpi=300, bbox_inches='tight')

        plt.close()

        print(f"✓ Saved loss curves: {plots_dir / f'{model_name.lower()}_loss_curves.png'}")

    

    # === PLOT 2: Accuracy Curves ===

    if has_accuracy or has_val_acc:

        fig, ax = plt.subplots(figsize=(10, 6))

        

        if has_accuracy:

            ax.plot(epochs[:len(train_acc)], train_acc, 

                   marker='o', linewidth=2, label='Training Accuracy', 

                   color='#06A77D', markersize=6)

        

        if has_val_acc:

            ax.plot(epochs[:len(val_acc)], val_acc, 

                   marker='s', linewidth=2, label='Validation Accuracy', 

                   color='#D4AF37', markersize=6)

        

        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')

        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')

        ax.set_title(f'{model_name} - Training Accuracy Curves', 

                    fontsize=14, fontweight='bold', pad=20)

        ax.legend(fontsize=11, framealpha=0.9)

        ax.grid(True, alpha=0.3, linestyle='--')

        ax.set_xlim(left=min(epochs) if epochs else 0)

        ax.set_ylim([0, 100])

        

        # Add max accuracy annotation

        if has_val_acc and val_acc:

            max_val_acc = max(val_acc)

            max_epoch = epochs[val_acc.index(max_val_acc)]

            ax.annotate(f'Max: {max_val_acc:.2f}%',

                       xy=(max_epoch, max_val_acc),

                       xytext=(10, -15), textcoords='offset points',

                       bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),

                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        

        plt.tight_layout()

        plt.savefig(plots_dir / f'{model_name.lower()}_accuracy_curves.png', 

                   dpi=300, bbox_inches='tight')

        plt.close()

        print(f"✓ Saved accuracy curves: {plots_dir / f'{model_name.lower()}_accuracy_curves.png'}")

    

    # === PLOT 3: Combined Loss & Accuracy (if both available) ===

    if (has_loss or has_val_loss) and (has_accuracy or has_val_acc):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        

        # Left: Loss

        if has_loss:

            ax1.plot(epochs[:len(train_loss)], train_loss, 

                    marker='o', linewidth=2, label='Train', 

                    color='#2E86AB', markersize=5)

        if has_val_loss:

            ax1.plot(epochs[:len(val_loss)], val_loss, 

                    marker='s', linewidth=2, label='Validation', 

                    color='#A23B72', markersize=5)

        

        ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')

        ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')

        ax1.set_title('Loss', fontsize=12, fontweight='bold')

        ax1.legend(fontsize=10)

        ax1.grid(True, alpha=0.3, linestyle='--')

        

        # Right: Accuracy

        if has_accuracy:

            ax2.plot(epochs[:len(train_acc)], train_acc, 

                    marker='o', linewidth=2, label='Train', 

                    color='#06A77D', markersize=5)

        if has_val_acc:

            ax2.plot(epochs[:len(val_acc)], val_acc, 

                    marker='s', linewidth=2, label='Validation', 

                    color='#D4AF37', markersize=5)

        

        ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')

        ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')

        ax2.set_title('Accuracy', fontsize=12, fontweight='bold')

        ax2.legend(fontsize=10)

        ax2.grid(True, alpha=0.3, linestyle='--')

        ax2.set_ylim([0, 100])

        

        fig.suptitle(f'{model_name} - Training Progress', 

                    fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()

        plt.savefig(plots_dir / f'{model_name.lower()}_combined_curves.png', 

                   dpi=300, bbox_inches='tight')

        plt.close()

        print(f"✓ Saved combined curves: {plots_dir / f'{model_name.lower()}_combined_curves.png'}")





def plot_loss_comparison(models, results_dir: Path, plots_dir: Path):

    """

    Compare final training loss across multiple models.

    

    Args:

        models: List of model names

        results_dir: Directory containing results

        plots_dir: Directory to save plots

    """

    plots_dir.mkdir(parents=True, exist_ok=True)

    

    model_names = []

    final_losses = []

    

    for model in models:

        model_name = model.upper()

        history = load_metrics(model_name, results_dir, 'training_history')

        

        if history and history.get('train_loss'):

            model_names.append(model_name)

            # Get final loss (last epoch)

            train_loss = history['train_loss']

            val_loss = history.get('val_loss', [])

            

            # Use validation loss if available, otherwise training loss

            final_loss = val_loss[-1] if val_loss else train_loss[-1]

            final_losses.append(final_loss)

    

    if not model_names:

        print("⚠️ No loss data found for comparison")

        return

    

    # Create bar plot

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#2E86AB', '#A23B72', '#06A77D', '#D4AF37', '#F18F01']

    bars = ax.bar(model_names, final_losses, 

                  color=colors[:len(model_names)], alpha=0.8, edgecolor='black')

    

    # Add value labels on bars

    for bar in bars:

        height = bar.get_height()

        ax.text(bar.get_x() + bar.get_width()/2., height,

               f'{height:.4f}',

               ha='center', va='bottom', fontweight='bold', fontsize=10)

    

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')

    ax.set_ylabel('Final Loss', fontsize=12, fontweight='bold')

    ax.set_title('Final Training Loss Comparison', 

                fontsize=14, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    

    plt.tight_layout()

    plt.savefig(plots_dir / 'loss_comparison.png', dpi=300, bbox_inches='tight')

    plt.close()

    print(f"✓ Saved loss comparison: {plots_dir / 'loss_comparison.png'}")





def generate_comparison_plots(models, results_dir: Path, plots_dir: Path):

    """Generate comparison plots for multiple models."""

    plots_dir.mkdir(parents=True, exist_ok=True)

    

    # Collect data for all models

    model_data = {}

    for model in models:

        model_name = model.upper()

        model_data[model_name] = {

            'parameters': load_metrics(model_name, results_dir, 'parameters'),

            'memory': load_metrics(model_name, results_dir, 'memory'),

            'latency': load_metrics(model_name, results_dir, 'latency'),

            'performance': load_metrics(model_name, results_dir, 'performance')

        }

    

    # 1. Parameter Comparison

    fig, ax = plt.subplots(figsize=(10, 6))

    model_names = list(model_data.keys())

    total_params = [model_data[m]['parameters']['total_parameters'] / 1e6 

                   for m in model_names if model_data[m]['parameters']]

    trainable_params = [model_data[m]['parameters']['trainable_parameters'] / 1e6 

                       for m in model_names if model_data[m]['parameters']]

    

    x = np.arange(len(model_names))

    width = 0.35

    

    ax.bar(x - width/2, total_params, width, label='Total', alpha=0.8)

    ax.bar(x + width/2, trainable_params, width, label='Trainable', alpha=0.8)

    

    ax.set_xlabel('Model')

    ax.set_ylabel('Parameters (Millions)')

    ax.set_title('Model Parameter Comparison')

    ax.set_xticks(x)

    ax.set_xticklabels(model_names)

    ax.legend()

    ax.grid(True, alpha=0.3)

    

    plt.tight_layout()

    plt.savefig(plots_dir / 'parameter_comparison.png', dpi=300)

    plt.close()

    

    # 2. Memory Usage Comparison

    fig, ax = plt.subplots(figsize=(10, 6))

    memory_values = []

    for m in model_names:

        if model_data[m]['memory']:

            mem = model_data[m]['memory'].get('training_peak_gpu_mb', 0)

            if mem != 'N/A':

                memory_values.append(float(mem))

            else:

                memory_values.append(0)

        else:

            memory_values.append(0)

    

    ax.bar(model_names, memory_values, alpha=0.8, color='coral')

    ax.set_xlabel('Model')

    ax.set_ylabel('Peak GPU Memory (MB)')

    ax.set_title('Training Memory Usage Comparison')

    ax.grid(True, alpha=0.3, axis='y')

    

    plt.tight_layout()

    plt.savefig(plots_dir / 'memory_usage_comparison.png', dpi=300)

    plt.close()

    

    # 3. Latency Comparison

    fig, ax = plt.subplots(figsize=(10, 6))

    latency_values = []

    for m in model_names:

        if model_data[m]['latency']:

            latency_values.append(model_data[m]['latency']['average_ms'])

        else:

            latency_values.append(0)

    

    ax.bar(model_names, latency_values, alpha=0.8, color='skyblue')

    ax.set_xlabel('Model')

    ax.set_ylabel('Average Inference Latency (ms)')

    ax.set_title('Inference Latency Comparison')

    ax.grid(True, alpha=0.3, axis='y')

    

    plt.tight_layout()

    plt.savefig(plots_dir / 'latency_comparison.png', dpi=300)

    plt.close()

    

    # 4. Accuracy Comparison

    fig, ax = plt.subplots(figsize=(10, 6))

    accuracy_values = []

    for m in model_names:

        if model_data[m]['performance']:

            accuracy_values.append(model_data[m]['performance']['accuracy'])

        else:

            accuracy_values.append(0)

    

    ax.bar(model_names, accuracy_values, alpha=0.8, color='lightgreen')

    ax.set_xlabel('Model')

    ax.set_ylabel('Accuracy (%)')

    ax.set_title('Model Accuracy Comparison')

    ax.set_ylim([0, 100])

    ax.grid(True, alpha=0.3, axis='y')

    

    plt.tight_layout()

    plt.savefig(plots_dir / 'accuracy_comparison.png', dpi=300)

    plt.close()

    

    # === NEW: Generate individual training curves for each model ===

    print(f"\n{'='*60}")

    print("Generating Training Curves")

    print(f"{'='*60}")

    for model in models:

        plot_training_curves(model.upper(), results_dir, plots_dir)

    

    # === NEW: Generate loss comparison ===

    plot_loss_comparison(models, results_dir, plots_dir)



    print(f"\n✓ All comparison plots saved to {plots_dir}")





# ============================================================================

#  MANUSCRIPT FIGURES (CSV-driven; physics-aware phase-preservation story)

# ----------------------------------------------------------------------------

#  These read the aggregated result tables produced by collect_metrics.py:

#     results/compiled_training_metrics.csv   (LoRA rank sweep, per-class Dice)

#     results/full_finetune_metrics.csv       (full fine-tuning baselines)

#     results/ablation_metrics.csv            (loss-component ablation)

#     results/compiled_morphology_trends.csv  (area / circularity / opt_volume)

#  and generate the figures referenced by the manuscript placeholders

#  (fig:lora_vs_full, fig:loss_ablation, tab:morphology_milestones).

# ============================================================================



# Shared aesthetic ----------------------------------------------------------

_PALETTE = {

    "lora":    "#2E86AB",   # blue  – parameter-efficient adaptation

    "full_ft": "#C0392B",   # red   – full fine-tuning (collapses minorities)

    "dice":    "#8D99AE",   # grey  – geometry-only baseline

    "pmc":     "#E9C46A",   # amber – contrast term (unstable alone)

    "pmc_bga": "#2A9D8F",   # teal  – + boundary alignment

    "full":    "#264653",   # dark  – full physics-aware loss

    "bf1":     "#2A9D8F",

    "pve":     "#C0392B",

}

_CLASS_ORDER = ["discocyte", "echinocyte", "spherocyte", "stomatocyte"]

_CLASS_LABELS = ["Disco.", "Echino.", "Sphero.", "Stomato."]

_ARCH_LABELS = {

    "EDGE_SAM": "EdgeSAM",

    "MOBILE_SAM": "MobileSAM",

    "MOBILENET_UNET": "MobileNet-UNet",

}





def _apply_style():

    """Best-effort clean publication style (falls back gracefully)."""

    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):

        try:

            plt.style.use(style)

            break

        except (OSError, ValueError):

            continue

    plt.rcParams.update({

        "axes.titleweight": "bold",

        "axes.labelweight": "bold",

        "font.size": 11,

    })





def _read_rows(path: Path):

    """Read a CSV into a list of dict rows; return [] if missing."""

    path = Path(path)

    if not path.exists():

        print(f"⚠️  Missing results file: {path}")

        return []

    with open(path, newline="") as f:

        return list(csv.DictReader(f))





def _f(row, key, default=0.0):

    """Safe float parse (handles '', 'N/A', tiny-collapse values)."""

    try:

        return float(row.get(key, default))

    except (TypeError, ValueError):

        return default





def plot_lora_vs_full_finetune(results_dir: Path, plots_dir: Path):

    """

    Per-class Dice for LoRA (r=8) vs full fine-tuning across all architectures.

    Visually exposes the catastrophic minority-class collapse (echinocyte and

    stomatocyte Dice -> 0) that full fine-tuning induces. Fills fig:lora_vs_full.

    """

    results_dir, plots_dir = Path(results_dir), Path(plots_dir)

    plots_dir.mkdir(parents=True, exist_ok=True)



    lora_rows = {r["Architecture"]: r for r in _read_rows(results_dir / "compiled_training_metrics.csv")

                 if r.get("Rank") == "8"}

    full_rows = {r["Architecture"]: r for r in _read_rows(results_dir / "full_finetune_metrics.csv")}



    archs = [a for a in ["EDGE_SAM", "MOBILE_SAM", "MOBILENET_UNET"]

             if a in lora_rows and a in full_rows]

    if not archs:

        print("⚠️  plot_lora_vs_full_finetune: no matching LoRA/full-FT rows found.")

        return



    fig, axes = plt.subplots(1, len(archs), figsize=(5.2 * len(archs), 4.6), sharey=True)

    if len(archs) == 1:

        axes = [axes]



    x = np.arange(len(_CLASS_ORDER))

    width = 0.38

    for ax, arch in zip(axes, archs):

        lora = [_f(lora_rows[arch], f"dice_{c}") for c in _CLASS_ORDER]

        full = [_f(full_rows[arch], f"dice_{c}") for c in _CLASS_ORDER]

        # tiny collapse values (~1e-13) render as 0

        lora = [0.0 if v < 1e-3 else v for v in lora]

        full = [0.0 if v < 1e-3 else v for v in full]



        b1 = ax.bar(x - width / 2, lora, width, label="LoRA ($r{=}8$)",

                    color=_PALETTE["lora"], edgecolor="black", linewidth=0.6)

        b2 = ax.bar(x + width / 2, full, width, label="Full Fine-Tune",

                    color=_PALETTE["full_ft"], edgecolor="black", linewidth=0.6)



        # mark collapsed (zero) bars explicitly

        for bar, v in list(zip(b1, lora)) + list(zip(b2, full)):

            if v == 0.0:

                ax.text(bar.get_x() + bar.get_width() / 2, 0.015, "0",

                        ha="center", va="bottom", fontsize=8, color="#C0392B",

                        fontweight="bold")



        ax.set_title(_ARCH_LABELS.get(arch, arch))

        ax.set_xticks(x)

        ax.set_xticklabels(_CLASS_LABELS, rotation=0)

        ax.set_ylim(0, 1.0)

        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    axes[0].set_ylabel("Per-Class Dice $\\uparrow$")

    axes[-1].legend(fontsize=9, framealpha=0.9, loc="upper right")

    fig.suptitle("LoRA vs. Full Fine-Tuning: Minority-Class Collapse",

                 fontsize=14, fontweight="bold")

    plt.tight_layout(rect=(0, 0, 1, 0.96))

    out = plots_dir / "fig_lora_vs_full_finetune.png"

    plt.savefig(out, dpi=300, bbox_inches="tight")

    plt.close()

    print(f"✓ Saved: {out}")





def plot_loss_ablation(results_dir: Path, plots_dir: Path):

    """

    Component-wise loss ablation on EdgeSAM (r=8). Left panel: Mean Dice and

    Boundary F1 across the four loss configurations. Right panel: global

    phase-volume error (the physically meaningful conserved quantity), which

    the full loss minimizes. Fills fig:loss_ablation.

    """

    results_dir, plots_dir = Path(results_dir), Path(plots_dir)

    plots_dir.mkdir(parents=True, exist_ok=True)



    rows = {r.get("Loss Config"): r for r in _read_rows(results_dir / "ablation_metrics.csv")

            if r.get("Architecture") == "EDGE_SAM"}

    order = ["dice_only", "pmc", "pmc_bga", "full"]

    labels = ["Dice\nonly", "+PMC", "+PMC\n+BGA", "Full\n$\\mathcal{L}$"]

    order = [k for k in order if k in rows]

    if not order:

        print("⚠️  plot_loss_ablation: no EdgeSAM ablation rows found.")

        return

    labels = labels[:len(order)]



    dice = [_f(rows[k], "mean_dice") for k in order]

    bf1 = [_f(rows[k], "bf1") for k in order]

    pve = [_f(rows[k], "phase_vol_error") * 100 for k in order]

    colors = [_PALETTE["dice"], _PALETTE["pmc"], _PALETTE["pmc_bga"], _PALETTE["full"]][:len(order)]



    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))



    # Left: Dice + Boundary F1 grouped

    x = np.arange(len(order))

    width = 0.38

    ax1.bar(x - width / 2, dice, width, label="Mean Dice", color=_PALETTE["lora"],

            edgecolor="black", linewidth=0.6)

    ax1.bar(x + width / 2, bf1, width, label="Boundary F1", color=_PALETTE["bf1"],

            edgecolor="black", linewidth=0.6)

    for i, (d, b) in enumerate(zip(dice, bf1)):

        ax1.text(i - width / 2, d + 0.01, f"{d:.3f}", ha="center", va="bottom", fontsize=8)

        ax1.text(i + width / 2, b + 0.01, f"{b:.3f}", ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)

    ax1.set_xticklabels(labels)

    ax1.set_ylabel("Score $\\uparrow$")

    ax1.set_ylim(0, 1.08)

    ax1.set_title("Segmentation & Boundary Quality")

    ax1.legend(fontsize=9, framealpha=0.9)

    ax1.grid(True, axis="y", alpha=0.3, linestyle="--")



    # Right: global phase-volume error (lower better)

    bars = ax2.bar(x, pve, color=colors, edgecolor="black", linewidth=0.6, width=0.6)

    for bar, v in zip(bars, pve):

        ax2.text(bar.get_x() + bar.get_width() / 2, v + max(pve) * 0.01,

                 f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.set_xticks(x)

    ax2.set_xticklabels(labels)

    ax2.set_ylabel("Global Phase-Volume Error (\\%) $\\downarrow$")

    ax2.set_title("Optical-Mass Conservation")

    ax2.grid(True, axis="y", alpha=0.3, linestyle="--")

    # annotate the unstable PMC-only spike

    if "pmc" in order:

        j = order.index("pmc")

        ax2.annotate("PMC alone is\nunstable (no scaffold)",

                     xy=(j, pve[j]), xytext=(j + 0.15, pve[j] * 0.7),

                     fontsize=8, ha="left", color="#C0392B",

                     arrowprops=dict(arrowstyle="->", color="#C0392B"))



    fig.suptitle("Physics-Aware Loss Ablation (EdgeSAM, $r{=}8$)",

                 fontsize=14, fontweight="bold")

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    out = plots_dir / "fig_loss_ablation.png"

    plt.savefig(out, dpi=300, bbox_inches="tight")

    plt.close()

    print(f"✓ Saved: {out}")





def plot_morphology_milestones(results_dir: Path, plots_dir: Path,

                               architecture: str = "edge_sam_lora", rank: str = "8"):

    """

    Longitudinal morphology from predicted masks: projected area, circularity,

    and integrated optical volume (mean +/- SD) across storage days. Supports

    the quantitative phase analysis (tab:morphology_milestones).

    """

    results_dir, plots_dir = Path(results_dir), Path(plots_dir)

    plots_dir.mkdir(parents=True, exist_ok=True)



    rows = [r for r in _read_rows(results_dir / "compiled_morphology_trends.csv")

            if architecture in r.get("Architecture", "").lower() and r.get("Rank") == rank]

    if not rows:

        print("⚠️  plot_morphology_milestones: no morphology rows found.")

        return



    days = sorted({int(r["storage_day"]) for r in rows})

    def agg(day, key, scale=1.0):

        vals = [_f(r, key) / scale for r in rows if int(r["storage_day"]) == day]

        return (np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0.0)



    area = [agg(d, "area") for d in days]

    circ = [agg(d, "circularity") for d in days]

    vol = [agg(d, "opt_volume", scale=1e6) for d in days]



    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    for ax, (data, title, ylab, col) in zip(

            axes,

            [(area, "Projected Area", "Area (px$^2$)", _PALETTE["lora"]),

             (circ, "Circularity", "Circularity", _PALETTE["pmc_bga"]),

             (vol, "Integrated Optical Volume", "Opt. Vol. ($\\times10^6$ rad$\\cdot$px$^2$)", _PALETTE["full"])]):

        m = [d[0] for d in data]

        s = [d[1] for d in data]

        ax.errorbar(days, m, yerr=s, marker="o", linewidth=2, capsize=3,

                    color=col, markersize=5)

        ax.set_xlabel("Storage Day")

        ax.set_ylabel(ylab)

        ax.set_title(title)

        ax.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle("Storage-Lesion Morphology Trajectory (EdgeSAM, $r{=}8$)",

                 fontsize=14, fontweight="bold")

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    out = plots_dir / "fig_morphology_milestones.png"

    plt.savefig(out, dpi=300, bbox_inches="tight")

    plt.close()

    print(f"✓ Saved: {out}")





def generate_manuscript_figures(results_dir: Path, plots_dir: Path):

    """

    One-call driver for the physics-aware manuscript figures.



    Usage:

        from utils.plotting import generate_manuscript_figures

        generate_manuscript_figures(Path("results"), Path("global_ablations_and_comparisons"))

    """

    _apply_style()

    print(f"\n{'='*60}\nGenerating manuscript figures\n{'='*60}")

    plot_lora_vs_full_finetune(results_dir, plots_dir)

    plot_loss_ablation(results_dir, plots_dir)

    plot_morphology_milestones(results_dir, plots_dir)

    print(f"\n✓ Manuscript figures saved to {plots_dir}")





# ============================================================================

#  PUBLICATION FIGURES (drop-in replacements for the numbered manuscript figs)

# ----------------------------------------------------------------------------

#  Consistent, colour-blind-safe styling; every value read from the committed

#  result CSVs. Output filenames match the \includegraphics keys in main.tex:

#     Figure 1  -> fig_per_class_dice.png       (per-class Dice @ r=8)

#     Figure 2  -> radar_chart.png              (optimal-model radar @ r=8)

#     Figure 3  -> ablation_curve.png           (LoRA rank vs Dice / params)

#     Figure 4  -> pareto_frontier.png          (accuracy vs throughput)

#     Figure 6  -> fig_phase_scatter.png        (GT vs pred optical volume) *

#     Figure 9  -> fig_population_shift.png     (RBC population shift)

#     Figure 10 -> geometric_trajectory.png     (area-vs-circularity path)

#  * Figure 6 requires REAL paired volumes; see plot_phase_volume_correlation.

# ============================================================================



# Architecture colours (colour-blind safe; EdgeSAM = hero green)

_ARCH_COLORS = {

    "EDGE_SAM":       "#1B9E77",

    "MOBILE_SAM":     "#7570B3",

    "MOBILENET_UNET": "#D95F02",

}

# Morphology class colours (index 1..4)

_CLASS_COLORS = {

    1: "#457B9D",  # discocyte

    2: "#E9C46A",  # echinocyte

    3: "#E76F51",  # spherocyte

    4: "#8E7DBE",  # stomatocyte

}

_CLASS_NAME = {1: "Discocyte", 2: "Echinocyte", 3: "Spherocyte", 4: "Stomatocyte"}





def _label_bars(ax, bars, fmt="{:.3f}", zero_flag=True, fontsize=8):

    for b in bars:

        h = b.get_height()

        if zero_flag and h < 1e-3:

            ax.text(b.get_x() + b.get_width() / 2, 0.012, "0", ha="center",

                    va="bottom", fontsize=fontsize, color="#C0392B", fontweight="bold")

        else:

            ax.text(b.get_x() + b.get_width() / 2, h + 0.012, fmt.format(h),

                    ha="center", va="bottom", fontsize=fontsize)





def fig01_per_class_dice(results_dir: Path, plots_dir: Path):

    """Figure 1 — per-class Dice for the three architectures at LoRA rank 8."""

    results_dir, plots_dir = Path(results_dir), Path(plots_dir)

    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = {r["Architecture"]: r for r in _read_rows(results_dir / "compiled_training_metrics.csv")

            if r.get("Rank") == "8"}

    archs = [a for a in ["MOBILENET_UNET", "EDGE_SAM", "MOBILE_SAM"] if a in rows]

    if not archs:

        print("⚠️  fig01: no rank-8 rows found."); return



    x = np.arange(len(_CLASS_ORDER))

    n = len(archs)

    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(8.5, 5))

    for i, arch in enumerate(archs):

        vals = [max(0.0, _f(rows[arch], f"dice_{c}")) for c in _CLASS_ORDER]

        vals = [0.0 if v < 1e-3 else v for v in vals]

        bars = ax.bar(x + (i - (n - 1) / 2) * width, vals, width,

                      label=_ARCH_LABELS[arch], color=_ARCH_COLORS[arch],

                      edgecolor="black", linewidth=0.5)

        _label_bars(ax, bars)

    ax.set_xticks(x)

    ax.set_xticklabels(_CLASS_LABELS)

    ax.set_ylabel("Dice Score $\\uparrow$", fontsize=12)

    ax.set_ylim(0, 1.05)

    ax.set_title("Per-Class Segmentation Accuracy (LoRA Rank 8)", fontsize=13, fontweight="bold")

    ax.legend(frameon=True, framealpha=0.9, fontsize=10, loc="upper right")

    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()

    out = plots_dir / "fig_per_class_dice.png"

    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()

    print(f"✓ Saved: {out}")





def _hw_row(hw_rows, arch, prefer_onnx=True, precision="FP16"):

    """Pick a hardware row: ONNX if available for this arch, else native."""

    cands = [r for r in hw_rows if r.get("architecture") == arch

             and r.get("precision") == precision and r.get("lora_r") in ("8", "FULL")]

    if prefer_onnx:

        onnx = [r for r in cands if str(r.get("onnx")).lower() == "true"]

        if onnx:

            return onnx[0]

    native = [r for r in cands if str(r.get("onnx")).lower() == "false"]

    return native[0] if native else (cands[0] if cands else None)





def fig02_radar_optimal(results_dir: Path, plots_dir: Path):

    """Figure 2 — radar comparing the three architectures at rank 8 across

    accuracy and efficiency axes (each axis normalized to its best model)."""

    results_dir, plots_dir = Path(results_dir), Path(plots_dir)

    plots_dir.mkdir(parents=True, exist_ok=True)

    seg = {r["Architecture"]: r for r in _read_rows(results_dir / "compiled_training_metrics.csv")

           if r.get("Rank") == "8"}

    hw = _read_rows(results_dir / "benchmarks" / "hardware_benchmark_cuda.csv")

    archs = [a for a in ["EDGE_SAM", "MOBILE_SAM", "MOBILENET_UNET"] if a in seg]

    if not archs or not hw:

        print("⚠️  fig02: missing seg or hardware rows."); return



    axes_labels = ["Mean Dice", "AJI", "Boundary F1", "Throughput\n(FPS)", "Memory Eff.\n(1/VRAM)"]

    raw = {}

    for a in archs:

        h = _hw_row(hw, a)

        if h is None:

            continue

        fps = _f(h, "fps")

        vram = _f(h, "peak_allocated_mb", 1.0) or 1.0

        raw[a] = [_f(seg[a], "mean_dice"), _f(seg[a], "aji"), _f(seg[a], "bf1"),

                  fps, 1.0 / vram]

    if not raw:

        print("⚠️  fig02: no matched rows."); return

    mx = np.max(np.array(list(raw.values())), axis=0)

    mx[mx == 0] = 1.0

    norm = {a: [v / m for v, m in zip(vals, mx)] for a, vals in raw.items()}



    ang = np.linspace(0, 2 * np.pi, len(axes_labels), endpoint=False).tolist()

    ang += ang[:1]

    fig, ax = plt.subplots(figsize=(7.2, 7.2), subplot_kw=dict(polar=True))

    for a, vals in norm.items():

        v = vals + vals[:1]

        ax.plot(ang, v, color=_ARCH_COLORS[a], linewidth=2.2, marker="o",

                markersize=5, label=_ARCH_LABELS[a])

        ax.fill(ang, v, color=_ARCH_COLORS[a], alpha=0.12)

    ax.set_theta_offset(np.pi / 2)

    ax.set_theta_direction(-1)

    ax.set_xticks(ang[:-1])

    ax.set_xticklabels(axes_labels, fontsize=11)

    ax.set_yticks([0.25, 0.5, 0.75, 1.0])

    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], color="grey", fontsize=8)

    ax.set_ylim(0, 1.08)

    ax.set_title("Optimal Model Comparison (LoRA Rank 8)", fontsize=13, fontweight="bold", pad=24)

    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10), fontsize=10, frameon=True)

    plt.tight_layout()

    out = plots_dir / "radar_chart.png"

    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()

    print(f"✓ Saved: {out}  (note: MobileSAM FPS/VRAM are native PyTorch; it has no ONNX path)")





def fig03_rank_dual_axis(results_dir: Path, plots_dir: Path):

    """Figure 3 — EdgeSAM Mean Dice (line) and trainable-parameter overhead

    (bars) as a function of LoRA rank; r=8 highlighted."""

    results_dir, plots_dir = Path(results_dir), Path(plots_dir)

    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = sorted([r for r in _read_rows(results_dir / "compiled_training_metrics.csv")

                   if r.get("Architecture") == "EDGE_SAM"],

                  key=lambda r: int(r["Rank"]))

    if not rows:

        print("⚠️  fig03: no EdgeSAM rows."); return

    ranks = [int(r["Rank"]) for r in rows]

    dice = [_f(r, "mean_dice") for r in rows]

    params_m = [_f(r, "Trainable Params") / 1e6 for r in rows]

    xpos = np.arange(len(ranks))



    fig, ax1 = plt.subplots(figsize=(8.5, 5))

    ax2 = ax1.twinx()

    bars = ax2.bar(xpos, params_m, width=0.55, color="#B8C4CC",

                   edgecolor="black", linewidth=0.5, label="Trainable Params (M)", zorder=1)

    line, = ax1.plot(xpos, dice, color=_ARCH_COLORS["EDGE_SAM"], marker="o",

                     markersize=8, linewidth=2.5, label="Mean Dice", zorder=3)

    # highlight optimal r=8

    if 8 in ranks:

        j = ranks.index(8)

        ax1.scatter([xpos[j]], [dice[j]], s=260, facecolors="none",

                    edgecolors="#C0392B", linewidths=2.2, zorder=4)

        ax1.annotate("optimal $r{=}8$", xy=(xpos[j], dice[j]),

                     xytext=(xpos[j] + 0.15, dice[j] + 0.06), fontsize=10,

                     color="#C0392B", fontweight="bold",

                     arrowprops=dict(arrowstyle="->", color="#C0392B"))

    for xi, d in zip(xpos, dice):

        ax1.text(xi, d + 0.015, f"{d:.3f}", ha="center", va="bottom", fontsize=9,

                 color=_ARCH_COLORS["EDGE_SAM"])

    ax1.set_xticks(xpos)

    ax1.set_xticklabels([f"$r{{=}}{r}$" for r in ranks])

    ax1.set_xlabel("LoRA Rank", fontsize=12)

    ax1.set_ylabel("Mean Dice $\\uparrow$", fontsize=12, color=_ARCH_COLORS["EDGE_SAM"])

    ax1.set_ylim(0, max(dice) * 1.25)

    ax1.tick_params(axis="y", labelcolor=_ARCH_COLORS["EDGE_SAM"])

    ax2.set_ylabel("Trainable Parameters (M) $\\downarrow$", fontsize=12, color="#5A6B75")

    ax2.tick_params(axis="y", labelcolor="#5A6B75")

    ax1.set_title("Effect of LoRA Rank on EdgeSAM Accuracy and Overhead",

                  fontsize=13, fontweight="bold")

    ax1.set_zorder(ax2.get_zorder() + 1); ax1.patch.set_visible(False)

    lns = [line, bars]

    ax1.legend(lns, [l.get_label() for l in lns], loc="upper left", fontsize=10, frameon=True)

    ax1.grid(True, axis="y", alpha=0.25, linestyle="--")

    plt.tight_layout()

    out = plots_dir / "ablation_curve.png"

    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()

    print(f"✓ Saved: {out}")





def fig04_pareto_frontier(results_dir: Path, plots_dir: Path,

                          precision: str = "FP16", rank: str = "8"):

    """Figure 4 - accuracy vs. throughput Pareto frontier at LoRA rank 8.



    Every value is read from the committed result tables:

        results/compiled_training_metrics.csv          -> mean Dice, per-class Dice

        results/benchmarks/hardware_benchmark_cuda.csv -> FPS, peak VRAM



    Design notes:

      * Minimal in-plot text: one label per architecture, everything else

        (speedups, VRAM, why MobileSAM has no ONNX point) belongs in the

        caption. Numbers that a reader needs to quote live in Table III.

      * The Pareto frontier is a step function. Straight segments between

        operating points would imply configurations that do not exist.

      * Native -> ONNX arcs pair the two runtimes of one architecture.

      * Architectures that never predict one or more morphology classes are

        drawn hollow inside a shaded band, so no model can look attractive on

        throughput alone.

      * Sized for a single IEEE column (3.58 in) at 6.5-8 pt, so nothing is

        down-scaled by \\includegraphics[width=\\columnwidth].

    """

    results_dir, plots_dir = Path(results_dir), Path(plots_dir)

    plots_dir.mkdir(parents=True, exist_ok=True)



    seg = {r["Architecture"]: r

           for r in _read_rows(results_dir / "compiled_training_metrics.csv")

           if r.get("Rank") == rank}

    hw = [r for r in _read_rows(results_dir / "benchmarks" / "hardware_benchmark_cuda.csv")

          if r.get("precision") == precision and r.get("lora_r") == rank]

    if not seg or not hw:

        print("\u26a0\ufe0f  fig04: missing segmentation or hardware rows."); return



    pts = []

    for r in hw:

        a = r["architecture"]

        if a not in seg:

            continue

        n_collapsed = sum(1 for c in _CLASS_ORDER if _f(seg[a], f"dice_{c}") < 1e-3)

        pts.append(dict(arch=a,

                        onnx=str(r.get("onnx")).lower() == "true",

                        fps=_f(r, "fps"),

                        dice=_f(seg[a], "mean_dice"),

                        vram=_f(r, "peak_allocated_mb"),

                        collapsed=n_collapsed))

    if not pts:

        print("\u26a0\ufe0f  fig04: no matched rows."); return



    def _dominated(p):

        return any((q["fps"] >= p["fps"] and q["dice"] >= p["dice"]

                    and (q["fps"] > p["fps"] or q["dice"] > p["dice"]))

                   for q in pts if q is not p)

    front = sorted([p for p in pts if not _dominated(p)], key=lambda p: p["fps"])



    # Speedups are printed to stdout so they can be quoted in the caption.

    for a in sorted({p["arch"] for p in pts}):

        nat = next((p for p in pts if p["arch"] == a and not p["onnx"]), None)

        onx = next((p for p in pts if p["arch"] == a and p["onnx"]), None)

        if nat and onx:

            print(f"   {_ARCH_LABELS[a]:<15} ONNX speedup x{onx['fps']/nat['fps']:.1f} "

                  f"({nat['fps']:.0f} -> {onx['fps']:.0f} FPS, "

                  f"{nat['vram']:.0f} -> {onx['vram']:.0f} MB)")



    rc = {"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,

          "ytick.labelsize": 7.5, "legend.fontsize": 6.8,

          "axes.labelweight": "bold", "axes.linewidth": 0.7,

          "font.family": "DejaVu Sans", "axes.grid": False}

    # style.context('default') isolates this figure from _apply_style(), whose

    # seaborn theme changes the axes box and therefore where offset labels land.

    with plt.style.context("default"), plt.rc_context(rc):

        fig, ax = plt.subplots(figsize=(3.58, 2.75))



        xmax = max(p["fps"] for p in pts) * 1.135

        ymin, ymax = 0.355, 0.875

        collapse_top = 0.545



        # --- mode-collapse exclusion band ---------------------------------

        n_bad = max(p["collapsed"] for p in pts)

        if n_bad:

            ax.axhspan(ymin, collapse_top, color="#B3261E", alpha=0.06, lw=0, zorder=0)

            ax.axhline(collapse_top, color="#B3261E", lw=0.7, ls=(0, (3, 2)),

                       alpha=0.6, zorder=1)



        # --- Pareto staircase ---------------------------------------------

        sx, sy = [0.0], [front[0]["dice"]]

        for p in front:

            sx += [p["fps"], p["fps"]]

            sy += [sy[-1], p["dice"]]

        sx.append(xmax); sy.append(sy[-1])

        ax.plot(sx, sy, color="#4F6D7A", ls="--", lw=0.9, zorder=2)



        # --- native -> ONNX compilation arcs (unlabelled; see caption) -----

        for a in sorted({p["arch"] for p in pts}):

            nat = next((p for p in pts if p["arch"] == a and not p["onnx"]), None)

            onx = next((p for p in pts if p["arch"] == a and p["onnx"]), None)

            if not (nat and onx):

                continue

            ax.annotate("", xy=(onx["fps"], onx["dice"]), xytext=(nat["fps"], nat["dice"]),

                        arrowprops=dict(arrowstyle="-|>,head_width=0.16,head_length=0.34",

                                        lw=0.9, color=_ARCH_COLORS[a], alpha=0.75,

                                        shrinkA=6, shrinkB=8,

                                        connectionstyle="arc3,rad=-0.16"), zorder=3)



        # --- selection ring -------------------------------------------------

        sel = next((p for p in pts if p["arch"] == "EDGE_SAM" and p["onnx"]), None)

        if sel:

            ax.scatter(sel["fps"], sel["dice"], s=260, marker="o", facecolor="none",

                       edgecolor=_ARCH_COLORS["EDGE_SAM"], lw=1.1, ls=(0, (2, 1.5)),

                       zorder=3)



        # --- markers ----------------------------------------------------------

        for p in pts:

            c = _ARCH_COLORS[p["arch"]]

            ax.scatter(p["fps"], p["dice"], s=100 if p["onnx"] else 54,

                       marker="*" if p["onnx"] else "o",

                       facecolor="white" if p["collapsed"] else c,

                       edgecolor=c, linewidths=1.2, zorder=5)



        # --- one short label per architecture, anchored to its native point ---

        anchor = {"MOBILE_SAM": (13, -9, "left"),

                  "EDGE_SAM": (-2, -14, "center"),

                  "MOBILENET_UNET": (0, -15, "center")}

        for a in sorted({p["arch"] for p in pts}):

            base = (next((p for p in pts if p["arch"] == a and not p["onnx"]), None)

                    or next(p for p in pts if p["arch"] == a))

            dx, dy, ha = anchor.get(a, (10, 0, "left"))

            ax.annotate(_ARCH_LABELS[a], xy=(base["fps"], base["dice"]),

                        xytext=(dx, dy), textcoords="offset points",

                        fontsize=7.2, ha=ha, va="center",

                        color=_ARCH_COLORS[a], fontweight="bold", zorder=6)



        ax.set_xlim(0, xmax)

        ax.set_ylim(ymin, ymax)

        ax.set_xticks([0, 100, 200, 300, 400])

        ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])

        ax.set_xlabel("Inference throughput (FPS)", labelpad=2)

        ax.set_ylabel("Mean Dice", labelpad=2)

        ax.grid(True, ls=":", lw=0.45, alpha=0.5)

        ax.set_axisbelow(True)

        for s in ("top", "right"):

            ax.spines[s].set_visible(False)



        handles = [

            Line2D([], [], marker="o", ls="", mfc="#555", mec="#555", ms=4.4,

                   label="Native PyTorch"),

            Line2D([], [], marker="*", ls="", mfc="#555", mec="#555", ms=8,

                   label="ONNX Runtime"),

            Line2D([], [], ls="--", color="#4F6D7A", lw=0.9, label="Pareto frontier"),

            Patch(facecolor="#B3261E", alpha=0.14, edgecolor="#B3261E", lw=0.7,

                  ls="--", label="Mode collapse"),

        ]

        ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.012, 1.03),

                  frameon=True, framealpha=0.94, edgecolor="#D0D0D0", fancybox=False,

                  handletextpad=0.5, borderpad=0.4, labelspacing=0.34, handlelength=1.5)



        fig.tight_layout(pad=0.35)

        out = plots_dir / "pareto_frontier.png"

        fig.savefig(out, dpi=600, bbox_inches="tight")

        plt.close(fig)

    print(f"\u2713 Saved: {out}")





def fig09_population_shift(results_dir: Path, plots_dir: Path,

                           architecture: str = "edge_sam", rank: str = "8"):

    """Figure 9 — predicted RBC morphology population proportions over storage days."""

    results_dir, plots_dir = Path(results_dir), Path(plots_dir)

    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = [r for r in _read_rows(results_dir / "compiled_morphology_trends.csv")

            if architecture in r.get("Architecture", "").lower() and r.get("Rank") == rank]

    if not rows:

        print("⚠️  fig09: no morphology rows."); return

    days = sorted({int(r["storage_day"]) for r in rows})

    classes = [1, 2, 3, 4]

    # proportion of each predicted class per day

    frac = {c: [] for c in classes}

    for d in days:

        day_rows = [r for r in rows if int(r["storage_day"]) == d]

        tot = len(day_rows) or 1

        for c in classes:

            frac[c].append(sum(1 for r in day_rows if int(float(r["pred_class"])) == c) / tot)



    fig, ax = plt.subplots(figsize=(9, 5))

    bottom = np.zeros(len(days))

    xpos = np.arange(len(days))

    for c in classes:

        vals = np.array(frac[c])

        ax.bar(xpos, vals, bottom=bottom, width=0.85, label=_CLASS_NAME[c],

               color=_CLASS_COLORS[c], edgecolor="white", linewidth=0.4)

        bottom += vals

    ax.set_xticks(xpos)

    ax.set_xticklabels(days)

    ax.set_xlabel("Storage Day", fontsize=12)

    ax.set_ylabel("Predicted Population Fraction", fontsize=12)

    ax.set_ylim(0, 1.0)

    ax.set_title("Storage-Induced RBC Morphological Population Shift (EdgeSAM, $r{=}8$)",

                 fontsize=13, fontweight="bold")

    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.22), fontsize=10, frameon=False)

    plt.tight_layout()

    out = plots_dir / "fig_population_shift.png"

    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()

    print(f"✓ Saved: {out}")





def fig10_geometric_trajectory(results_dir: Path, plots_dir: Path,

                               architecture: str = "edge_sam", rank: str = "8"):

    """Figure 10 — mean projected area vs circularity per storage day, drawn as

    a time-ordered trajectory (the spherocytic shift: falling area, rising circ.)."""

    results_dir, plots_dir = Path(results_dir), Path(plots_dir)

    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = [r for r in _read_rows(results_dir / "compiled_morphology_trends.csv")

            if architecture in r.get("Architecture", "").lower() and r.get("Rank") == rank]

    if not rows:

        print("⚠️  fig10: no morphology rows."); return

    days = sorted({int(r["storage_day"]) for r in rows})

    ax_mean, circ_mean = [], []

    for d in days:

        dr = [r for r in rows if int(r["storage_day"]) == d]

        ax_mean.append(np.mean([_f(r, "area") for r in dr]))

        circ_mean.append(np.mean([_f(r, "circularity") for r in dr]))



    fig, ax = plt.subplots(figsize=(8.5, 6))

    ax.plot(ax_mean, circ_mean, color="#999999", linewidth=1.4, zorder=1, alpha=0.7)

    sc = ax.scatter(ax_mean, circ_mean, c=days, cmap="viridis", s=160,

                    edgecolor="black", linewidth=0.6, zorder=2)

    for d, ax_, cy in zip(days, ax_mean, circ_mean):

        if d in (days[0], days[-1]) or d in (30, 37):

            ax.annotate(f"Day {d}", (ax_, cy), textcoords="offset points",

                        xytext=(8, 6), fontsize=9, fontweight="bold")

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)

    cbar.set_label("Storage Day", fontsize=11)

    ax.set_xlabel("Projected Area (px$^2$)", fontsize=12)

    ax.set_ylabel("Circularity", fontsize=12)

    ax.set_title("Geometric Degradation Trajectory of RBCs over 47 Days",

                 fontsize=13, fontweight="bold")

    ax.annotate("spherocytic shift", xy=(0.30, 0.86), xycoords="axes fraction",

                fontsize=10, style="italic", color="#555555")

    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()

    out = plots_dir / "geometric_trajectory.png"

    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()

    print(f"✓ Saved: {out}")





def plot_phase_volume_correlation(pairs_csv: Path, plots_dir: Path):

    """

    Figure 6 — ground-truth vs predicted optical phase volume (EdgeSAM r=8).



    HONESTY NOTE: this figure REQUIRES real paired measurements. Provide a CSV

    with columns:  gt_opt_volume, pred_opt_volume, pred_class

    where gt_opt_volume = sum(gt_mask * phase) and

          pred_opt_volume = sum(pred_mask * phase) per cell.

    Generate it with export_phase_volume_pairs() below (needs GPU + checkpoints).

    The previous implementation fabricated predictions via random noise around

    y=x; that is NOT publishable and is intentionally not reproduced here.

    """

    pairs_csv, plots_dir = Path(pairs_csv), Path(plots_dir)

    rows = _read_rows(pairs_csv)

    if not rows:

        print(f"⚠️  fig06: no real paired-volume CSV at {pairs_csv}. "

              f"Run export_phase_volume_pairs() first — figure skipped (not faked).")

        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    gt = np.array([_f(r, "gt_opt_volume") for r in rows]) / 1e6

    pr = np.array([_f(r, "pred_opt_volume") for r in rows]) / 1e6

    cls = [int(float(r.get("pred_class", 0))) for r in rows]



    fig, ax = plt.subplots(figsize=(7, 6.4))

    for c in sorted(set(cls)):

        if c == 0:

            continue

        m = [i for i, cc in enumerate(cls) if cc == c]

        ax.scatter(gt[m], pr[m], s=90, alpha=0.8, edgecolor="white",

                   color=_CLASS_COLORS.get(c, "#333333"), label=_CLASS_NAME.get(c, str(c)))

    lo, hi = float(min(gt.min(), pr.min())), float(max(gt.max(), pr.max()))

    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.8, label="Ideal ($y=x$)")

    # R^2 about the y=x line

    ss_res = float(np.sum((pr - gt) ** 2))

    ss_tot = float(np.sum((gt - gt.mean()) ** 2)) or 1.0

    r2 = 1.0 - ss_res / ss_tot

    mape = float(np.mean(np.abs(pr - gt) / (np.abs(gt) + 1e-9)) * 100)

    ax.text(0.05, 0.92, f"$R^2={r2:.3f}$\nMAPE $={mape:.1f}\\%$",

            transform=ax.transAxes, fontsize=11,

            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.9))

    ax.set_xlabel(r"Ground-Truth Phase Volume ($\times10^6$ rad$\cdot$px$^2$)", fontsize=12)

    ax.set_ylabel(r"Predicted Phase Volume ($\times10^6$ rad$\cdot$px$^2$)", fontsize=12)

    ax.set_title("Optical Volume Conservation (EdgeSAM, $r{=}8$)", fontsize=13, fontweight="bold")

    ax.legend(loc="lower right", fontsize=10, frameon=True)

    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()

    out = plots_dir / "fig_phase_scatter.png"

    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()

    print(f"✓ Saved: {out}")





def export_phase_volume_pairs(results_dir: Path = Path("results"),

                              data_root: str = "./dataset",

                              ckpt: str = "results/edge_sam_lora_r8/checkpoints/best_model.pt",

                              out_csv: Path = Path("results/phase_volume_pairs.csv")):

    """

    Compute REAL per-cell (gt_opt_volume, pred_opt_volume) pairs for Figure 6.

    Requires GPU + trained EdgeSAM r=8 checkpoint + the QPI dataset.

    Integrates phase over the ground-truth and predicted foreground per cell.

    """

    import csv as _csv

    import torch

    from datasets.qpi_dataset import QPIDataset

    from models import get_model



    class _Cfg:

        num_classes = 5; pretrained = False; image_size = 256

        lora_r = 8; lora_alpha = 8.0; insertion_strategy = "encoder_only"



    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model("edge_sam", _Cfg()).to(dev).eval()

    state = torch.load(ckpt, map_location=dev, weights_only=False)

    model.load_state_dict(state.get("model_state", state), strict=False)

    ds = QPIDataset(data_root=data_root, split="val", augment=False)



    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as f:

        w = _csv.writer(f)

        w.writerow(["stem", "gt_opt_volume", "pred_opt_volume", "pred_class"])

        with torch.no_grad():

            for i in range(len(ds)):

                s = ds[i]

                phase = s["phase"]

                gt = (s["mask"] > 0).float().cpu().numpy()

                out = model(phase.unsqueeze(0).to(dev))

                pr = (out.argmax(1).squeeze() > 0).float().cpu().numpy()

                ph = phase.squeeze().cpu().numpy()

                pcls = int(out.argmax(1).squeeze().cpu().numpy().max())

                w.writerow([s.get("stem", i), float((gt * ph).sum()),

                            float((pr * ph).sum()), pcls])

    print(f"✓ Wrote real paired volumes to {out_csv}")





def generate_publication_figures(results_dir: Path, plots_dir: Path,

                                 pairs_csv: Path = None):

    """One-call driver for the numbered manuscript figures (1,2,3,4,6,9,10)."""

    _apply_style()

    print(f"\n{'='*60}\nGenerating publication figures\n{'='*60}")

    fig01_per_class_dice(results_dir, plots_dir)

    fig02_radar_optimal(results_dir, plots_dir)

    fig03_rank_dual_axis(results_dir, plots_dir)

    fig04_pareto_frontier(results_dir, plots_dir)

    fig09_population_shift(results_dir, plots_dir)

    fig10_geometric_trajectory(results_dir, plots_dir)

    plot_phase_volume_correlation(

        pairs_csv or (Path(results_dir) / "phase_volume_pairs.csv"), plots_dir)

    print(f"\n✓ Publication figures saved to {plots_dir}")