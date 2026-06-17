import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import re

# ==========================================
# CONFIGURATION
# ==========================================
CLASS_MAP = {
    1: 'Discocyte',
    2: 'Echinocyte',
    3: 'Spherocyte',
    4: 'Stomatocyte'
}

PALETTE = {
    'Discocyte': '#2ca02c',   # Green
    'Echinocyte': '#ff7f0e',  # Orange
    'Spherocyte': '#d62728',  # Red
    'Stomatocyte': '#1f77b4'  # Blue
}

# Set seaborn style for publication-ready scientific plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# ==========================================
# 1. BIOLOGICAL MORPHOLOGY PLOTTING (CSVs)
# ==========================================
def plot_morphology_trends(csv_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    df = df[df['pred_class'] > 0].copy() # Filter background
    
    if df.empty:
        print(f"  [Skip] No foreground cells found in {csv_path.name}")
        return

    df['Cell Type'] = df['pred_class'].map(CLASS_MAP)
    
    metrics = {
        'area': ('Projected Area (pixels)', 'Projected Area Degradation'),
        'circularity': ('Circularity', 'Circularity Degradation'),
        'opt_volume': ('Phase-Integrated Optical Volume', 'Optical Volume Consistency')
    }

    for col, (y_label, title) in metrics.items():
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df, x='storage_day', y=col, hue='Cell Type', palette=PALETTE, marker='o', linewidth=2.5)
        plt.title(f"{title} Over Storage Time", fontweight='bold', pad=15)
        plt.xlabel('Storage Day', fontweight='bold')
        plt.ylabel(y_label, fontweight='bold')
        plt.legend(title='Predicted Morphology')
        plt.tight_layout()
        plt.savefig(output_dir / f"trend_{col}.png", dpi=300, bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='storage_day', hue='Cell Type', multiple='fill', palette=PALETTE, bins=len(df['storage_day'].unique()), shrink=0.8)
    plt.title("RBC Morphology Population Shift Over Storage Time", fontweight='bold', pad=15)
    plt.xlabel('Storage Day', fontweight='bold')
    plt.ylabel('Proportion of Cells', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "trend_population_shift.png", dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 2. TRAINING & VALIDATION PLOTTING (JSONs)
# ==========================================
def plot_training_metrics(json_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    hist = data.get("training_history", {})
    if not hist or "epochs" not in hist:
        print(f"  [Skip] Invalid training history in {json_path.name}")
        return

    epochs = hist["epochs"]

    # 1. Overall Loss Curve
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=epochs, y=hist["train_loss"], label='Train Loss', linewidth=2.5)
    sns.lineplot(x=epochs, y=hist["val_loss"], label='Val Loss', linewidth=2.5)
    plt.title("Training vs Validation Loss", fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_loss_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Validation Segmentation Metrics over time
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=epochs, y=hist["val_dice"], label='Mean Dice', linewidth=2.5)
    sns.lineplot(x=epochs, y=hist["val_aji"], label='AJI', linewidth=2.5)
    sns.lineplot(x=epochs, y=hist["val_bf1"], label='Boundary F1', linewidth=2.5)
    plt.title("Validation Segmentation Performance", fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_val_metrics_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Physics-Aware Loss Components (Log Scale for visibility)
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=epochs, y=hist["train_L_dice"], label='L_Dice', linewidth=2)
    sns.lineplot(x=epochs, y=hist["train_L_bga"], label='L_BGA (Boundary)', linewidth=2)
    sns.lineplot(x=epochs, y=hist["train_L_pv"], label='L_PV (Phase Vol)', linewidth=2)
    if "train_L_pmc" in hist and sum(hist["train_L_pmc"]) > 0:
        sns.lineplot(x=epochs, y=hist["train_L_pmc"], label='L_PMC (Contrast)', linewidth=2)
    plt.title("Physics-Aware Loss Components Convergence", fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Loss Magnitude (Log Scale)")
    plt.yscale('log') # Log scale because L_PV starts high and drops exponentially
    plt.tight_layout()
    plt.savefig(output_dir / "plot_physics_losses_log.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Best Per-Class Dice & Phase Volume Errors (Bar Charts)
    val_best = data.get("seg_metrics", {}).get("val_best", {})
    if val_best:
        classes = ['discocyte', 'echinocyte', 'spherocyte', 'stomatocyte']
        dice_scores = [val_best.get(f"dice_{c}", 0) for c in classes]
        pv_errors = [val_best.get(f"phase_vol_error_{c}", 0) for c in classes]
        display_names = [CLASS_MAP[i+1] for i in range(4)]
        colors = [PALETTE[name] for name in display_names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Dice Bar Chart
        sns.barplot(x=display_names, y=dice_scores, ax=ax1, palette=colors)
        ax1.set_title("Best Validation Dice per Class", fontweight='bold')
        ax1.set_ylim(0, 1.0)
        ax1.set_ylabel("Dice Score")

        # Phase Volume Error Bar Chart
        sns.barplot(x=display_names, y=pv_errors, ax=ax2, palette=colors)
        ax2.set_title("Phase Volume Error per Class (Lower is Better)", fontweight='bold')
        ax2.set_ylabel("Relative Error")

        plt.tight_layout()
        plt.savefig(output_dir / "plot_best_per_class_bars.png", dpi=300, bbox_inches='tight')
        plt.close()

# ==========================================
# MAIN BATCH PROCESSOR
# ==========================================
def process_all_results(results_base_dir):
    results_path = Path(results_base_dir)
    if not results_path.exists():
        print(f"Error: Directory '{results_base_dir}' does not exist.")
        return

    # Base Output Directories
    bio_out_dir = Path("morphology_analysis")
    train_out_dir = Path("training_validation_plots")

    # 1. Process all JSON files
    json_files = list(results_path.rglob("metrics.json"))
    print(f"Found {len(json_files)} metrics.json files. Plotting AI metrics...")
    
    for json_path in json_files:
        try:
            with open(json_path, 'r') as f:
                model_name = json.load(f).get("model_name", "UNKNOWN_MODEL")
            
            # Formats nicely: e.g., "EDGE_SAM_R16"
            specific_out = train_out_dir / model_name
            plot_training_metrics(json_path, specific_out)
            print(f"  -> Saved AI plots to: {specific_out}")
        except Exception as e:
            print(f"  [Error] Failed on {json_path.name}: {e}")

    # 2. Process all CSV files
    csv_files = list(results_path.rglob("morphology_trends*.csv"))
    print(f"\nFound {len(csv_files)} morphology CSV files. Plotting biological trends...")
    
    for csv_path in csv_files:
        try:
            # Extract rank from filename
            rank_match = re.search(r'rank_([a-zA-Z0-9]+)\.csv$', csv_path.name)
            rank = rank_match.group(1) if rank_match else "unknown"

            # Extract model type from folder structure (fallback cleanup)
            model_dir = csv_path.relative_to(results_path).parts[0]
            model_name = re.sub(r'_r\d+$', '', model_dir).upper()
            
            # Combine into neat name like: MOBILENET_UNET_LORA_RANK_4
            specific_out = bio_out_dir / f"{model_name}_RANK_{rank}"
            plot_morphology_trends(csv_path, specific_out)
            print(f"  -> Saved Bio plots to: {specific_out}")
        except Exception as e:
            print(f"  [Error] Failed on {csv_path.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    
    process_all_results(args.results_dir)
    print("\nAll plotting complete!")