import argparse
import pandas as pd
import numpy as np
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
# 2. INDIVIDUAL TRAINING PLOTTING (JSONs)
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

    # 3. Physics-Aware Loss Components (Log Scale)
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=epochs, y=hist["train_L_dice"], label='L_Dice', linewidth=2)
    sns.lineplot(x=epochs, y=hist["train_L_bga"], label='L_BGA (Boundary)', linewidth=2)
    sns.lineplot(x=epochs, y=hist["train_L_pv"], label='L_PV (Phase Vol)', linewidth=2)
    if "train_L_pmc" in hist and sum(hist["train_L_pmc"]) > 0:
        sns.lineplot(x=epochs, y=hist["train_L_pmc"], label='L_PMC (Contrast)', linewidth=2)
    plt.title("Physics-Aware Loss Components Convergence", fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Loss Magnitude (Log Scale)")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(output_dir / "plot_physics_losses_log.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Best Per-Class Dice & Phase Volume Errors
    val_best = data.get("seg_metrics", {}).get("val_best", {})
    if val_best:
        classes = ['discocyte', 'echinocyte', 'spherocyte', 'stomatocyte']
        dice_scores = [val_best.get(f"dice_{c}", 0) for c in classes]
        pv_errors = [val_best.get(f"phase_vol_error_{c}", 0) for c in classes]
        display_names = [CLASS_MAP[i+1] for i in range(4)]
        colors = [PALETTE[name] for name in display_names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        sns.barplot(x=display_names, y=dice_scores, ax=ax1, palette=colors)
        ax1.set_title("Best Validation Dice per Class", fontweight='bold')
        ax1.set_ylim(0, 1.0)
        ax1.set_ylabel("Dice Score")

        sns.barplot(x=display_names, y=pv_errors, ax=ax2, palette=colors)
        ax2.set_title("Phase Volume Error per Class (Lower is Better)", fontweight='bold')
        ax2.set_ylabel("Relative Error")

        plt.tight_layout()
        plt.savefig(output_dir / "plot_best_per_class_bars.png", dpi=300, bbox_inches='tight')
        plt.close()

# ==========================================
# 3. GLOBAL ABLATION & COMPARISONS
# ==========================================
def plot_global_ablation_and_tradeoffs(results_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_files = list(results_path.rglob("metrics.json"))
    
    # 3.1 Aggregating LoRA Sweep Data
    records = []
    for jpath in json_files:
        try:
            with open(jpath, 'r') as f:
                data = json.load(f)
            
            model_name = data.get("model_name", "")
            if "_R" in model_name:
                arch, rank_str = model_name.rsplit("_R", 1)
                rank = int(rank_str) if rank_str.isdigit() else 0
            else:
                arch = model_name
                rank = 0
            
            val_best = data.get("seg_metrics", {}).get("val_best", {})
            trainable_pct = data.get("parameters", {}).get("trainable_percentage", 0)
            
            records.append({
                "Architecture": arch.replace("_", "-"), 
                "Rank": rank,
                "Mean Dice": val_best.get("mean_dice", 0),
                "AJI": val_best.get("aji", 0),
                "Boundary F1": val_best.get("bf1", 0),
                "Phase Vol Error": val_best.get("phase_vol_error", 0),
                "Trainable %": trainable_pct
            })
        except Exception:
            pass
            
    if records:
        df_ablation = pd.DataFrame(records)
        df_ablation = df_ablation[df_ablation['Rank'] > 0] # Filter out Rank 0 sweeps
        df_ablation = df_ablation.sort_values('Rank')
        df_ablation['Rank_str'] = df_ablation['Rank'].astype(str) 
        
        # [PREVIOUS] Plot 1: Rank vs. Dice
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df_ablation, x='Rank_str', y='Mean Dice', hue='Architecture', marker='o', linewidth=2.5)
        plt.title("Effect of LoRA Rank on Segmentation Accuracy", fontweight='bold')
        plt.xlabel("LoRA Rank (r)")
        plt.ylabel("Mean Dice")
        plt.tight_layout()
        plt.savefig(output_dir / "ablation_rank_vs_dice.png", dpi=300)
        plt.close()

        # [NEW] Plot 2: Rank vs. Boundary F1 (Physics Quality)
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df_ablation, x='Rank_str', y='Boundary F1', hue='Architecture', marker='o', linewidth=2.5)
        plt.title("Effect of LoRA Rank on Boundary Preservation", fontweight='bold')
        plt.xlabel("LoRA Rank (r)")
        plt.ylabel("Boundary F1 Score")
        plt.tight_layout()
        plt.savefig(output_dir / "ablation_rank_vs_boundary_f1.png", dpi=300)
        plt.close()

        # [PREVIOUS] Plot 3: Rank vs. Phase Vol Error
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df_ablation, x='Rank_str', y='Phase Vol Error', hue='Architecture', marker='o', linewidth=2.5)
        plt.title("Effect of LoRA Rank on Phase Preservation", fontweight='bold')
        plt.xlabel("LoRA Rank (r)")
        plt.ylabel("Relative Phase Volume Error (Lower is Better)")
        plt.tight_layout()
        plt.savefig(output_dir / "ablation_rank_vs_phase_error.png", dpi=300)
        plt.close()

        # [NEW] Plot 4: The Optimal Configuration Comparison (Grouped Bar Chart for Rank 8)
        df_optimal = df_ablation[df_ablation['Rank'] == 8].copy()
        if not df_optimal.empty:
            df_melted = df_optimal.melt(
                id_vars=['Architecture'], 
                value_vars=['Mean Dice', 'AJI', 'Boundary F1'],
                var_name='Metric', 
                value_name='Score'
            )
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_melted, x='Architecture', y='Score', hue='Metric', palette='viridis')
            plt.title("Optimal Model Performance Comparison (LoRA Rank = 8)", fontweight='bold')
            plt.xlabel("Lightweight Architecture")
            plt.ylabel("Score")
            plt.ylim(0, 1.05)
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(output_dir / "comparison_optimal_models_rank8.png", dpi=300)
            plt.close()

        # [PREVIOUS] Plot 5: Trainable % vs Dice (Parameter Efficiency)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_ablation, x='Trainable %', y='Mean Dice', hue='Architecture', size='Rank', sizes=(50, 250), alpha=0.8)
        plt.title("Parameter Efficiency Trade-off", fontweight='bold')
        plt.xlabel("Trainable Parameters (%)")
        plt.ylabel("Mean Dice")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_dir / "ablation_trainable_vs_dice.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 3.2 Accuracy vs Efficiency Trade-offs (If Benchmark CSV exists)
    bench_csv = list(results_path.rglob("hardware_benchmark*.csv"))
    if bench_csv:
        df_bench = pd.read_csv(bench_csv[0])
        
        # [PREVIOUS] FPS vs Dice Pareto Front
        plt.figure(figsize=(9, 6))
        sns.scatterplot(data=df_bench, x='fps', y='mean_dice', hue='architecture', style='tensorrt', s=150, alpha=0.8)
        plt.title("Accuracy vs. Efficiency Trade-off", fontweight='bold')
        plt.xlabel("Inference Speed (FPS)")
        plt.ylabel("Mean Dice")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Arch / TensorRT")
        plt.tight_layout()
        plt.savefig(output_dir / "tradeoff_fps_vs_dice.png", dpi=300, bbox_inches='tight')
        plt.close()

        # [NEW] Trainable Parameters vs. FPS Scatter
        if 'overall_trainable_ratio_%' in df_bench.columns:
            plt.figure(figsize=(9, 6))
            sns.scatterplot(data=df_bench, x='overall_trainable_ratio_%', y='fps', hue='architecture', style='tensorrt', s=150, alpha=0.8)
            plt.title("Trainable Footprint vs. Inference Speed", fontweight='bold')
            plt.xlabel("Trainable Parameters (%)")
            plt.ylabel("Inference Speed (FPS)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Arch / TensorRT")
            plt.tight_layout()
            plt.savefig(output_dir / "tradeoff_params_vs_fps.png", dpi=300, bbox_inches='tight')
            plt.close()

        print(f"  -> Generated Global Ablation & Trade-off plots at {output_dir}")

# ==========================================
# MAIN BATCH PROCESSOR
# ==========================================
def process_all_results(results_base_dir):
    results_path = Path(results_base_dir)
    if not results_path.exists():
        print(f"Error: Directory '{results_base_dir}' does not exist.")
        return

    bio_out_dir = Path("morphology_analysis")
    train_out_dir = Path("training_validation_plots")
    global_out_dir = Path("global_ablations_and_comparisons")

    # 1. Process all JSON files (Individual Models)
    json_files = list(results_path.rglob("metrics.json"))
    print(f"Found {len(json_files)} metrics.json files. Plotting AI metrics...")
    for json_path in json_files:
        try:
            with open(json_path, 'r') as f:
                model_name = json.load(f).get("model_name", "UNKNOWN_MODEL")
            specific_out = train_out_dir / model_name
            plot_training_metrics(json_path, specific_out)
        except Exception as e:
            pass

    # 2. Process all CSV files (Morphology)
    csv_files = list(results_path.rglob("morphology_trends*.csv"))
    print(f"Found {len(csv_files)} morphology CSV files. Plotting biological trends...")
    for csv_path in csv_files:
        try:
            rank_match = re.search(r'rank_([a-zA-Z0-9]+)\.csv$', csv_path.name)
            rank = rank_match.group(1) if rank_match else "unknown"
            model_dir = csv_path.relative_to(results_path).parts[0]
            model_name = re.sub(r'_r\d+$', '', model_dir).upper()
            specific_out = bio_out_dir / f"{model_name}_RANK_{rank}"
            plot_morphology_trends(csv_path, specific_out)
        except Exception as e:
            pass

    # 3. Generate Global Ablations and Hardware Comparisons
    print(f"Aggregating global sweeps and benchmarking data...")
    plot_global_ablation_and_tradeoffs(results_path, global_out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    
    process_all_results(args.results_dir)
    print("\nAll plotting complete!")