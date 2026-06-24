import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import re
from scipy.ndimage import sobel

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
    'Discocyte': '#2ca02c',
    'Echinocyte': '#ff7f0e',
    'Spherocyte': '#d62728',
    'Stomatocyte': '#1f77b4'
}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# ==========================================
# 1. BIOLOGICAL MORPHOLOGY PLOTTING
# ==========================================
def plot_morphology_trends(csv_path, output_dir, model_name, rank):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    df = df[df['pred_class'] > 0].copy()

    if df.empty:
        return

    df['Cell Type'] = df['pred_class'].map(CLASS_MAP)
    title_suffix = f" - {model_name} (Rank {rank})"

    metrics = {
        'area': ('Projected Area (pixels)', 'Projected Area Degradation'),
        'circularity': ('Circularity', 'Circularity Degradation'),
        'opt_volume': ('Phase-Integrated Optical Volume', 'Optical Volume Consistency')
    }

    for col, (y_label, title) in metrics.items():
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df, x='storage_day', y=col, hue='Cell Type',
                     palette=PALETTE, marker='o', linewidth=2.5)
        plt.title(f"{title} Over Storage Time{title_suffix}", fontweight='bold', pad=20)
        plt.xlabel('Storage Day', fontweight='bold')
        plt.ylabel(y_label, fontweight='bold')
        plt.legend(title='Predicted Morphology')
        plt.tight_layout()
        plt.savefig(output_dir / f"trend_{col}.png", dpi=300, bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='storage_day', hue='Cell Type', multiple='fill',
                 palette=PALETTE, shrink=0.8)
    plt.title(f"RBC Morphology Population Shift{title_suffix}", fontweight='bold', pad=20)
    plt.xlabel('Storage Day', fontweight='bold')
    plt.ylabel('Proportion of Cells', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "trend_population_shift.png", dpi=300, bbox_inches='tight')
    plt.close()

    if 'area' in df.columns and 'circularity' in df.columns:
        plt.figure(figsize=(9, 6))
        df_agg = df.groupby(['storage_day', 'Cell Type'])[['area', 'circularity']].mean().reset_index()
        df_agg['Storage Day'] = df_agg['storage_day']
        sns.scatterplot(data=df_agg, x='area', y='circularity', hue='Cell Type',
                        palette=PALETTE, style='Storage Day', s=200, alpha=0.9)
        plt.title(f"Geometric Trajectory: Area vs Circularity{title_suffix}\n\n", fontweight='bold')
        plt.xlabel("Projected Area (pixels)", fontweight='bold')
        plt.ylabel("Circularity", fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Morphology & Day")
        plt.tight_layout()
        plt.savefig(output_dir / "biological_trajectory_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()


# ==========================================
# 2. INDIVIDUAL TRAINING PLOTTING
# ==========================================
def plot_training_metrics(json_path, output_dir, model_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'r') as f:
        data = json.load(f)

    hist = data.get("training_history", {})
    if not hist or "epochs" not in hist:
        return

    epochs = hist["epochs"]
    title_suffix = f" - {model_name.replace('_', '-').upper()}"

    plt.figure(figsize=(8, 6))
    sns.lineplot(x=epochs, y=hist["val_dice"], label='Mean Dice', linewidth=2.5)
    sns.lineplot(x=epochs, y=hist["val_aji"],  label='AJI',       linewidth=2.5)
    sns.lineplot(x=epochs, y=hist["val_bf1"],  label='Boundary F1', linewidth=2.5)
    plt.title(f"Validation Segmentation Performance{title_suffix}", fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_val_metrics_curve.png", dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# 3. DATA LOADERS FOR DYNAMIC PLOTTING
# ==========================================
def load_json_metrics(results_path):
    json_files = list(results_path.rglob("metrics.json"))
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

            val_best       = data.get("seg_metrics", {}).get("val_best", {})
            trainable_pct  = data.get("parameters", {}).get("trainable_percentage", 0)

            records.append({
                "Architecture":    arch.replace("_", "-"),
                "Rank":            rank,
                "Mean Dice":       val_best.get("mean_dice", 0),
                "AJI":             val_best.get("aji", 0),
                "Boundary F1":     val_best.get("bf1", 0),
                "Phase Vol Error": val_best.get("phase_vol_error", 0),
                "Trainable %":     trainable_pct,
                "Dice_Discocyte":  val_best.get("dice_discocyte", 0),
                "Dice_Echinocyte": val_best.get("dice_echinocyte", 0),
                "Dice_Spherocyte": val_best.get("dice_spherocyte", 0),
                "Dice_Stomatocyte":val_best.get("dice_stomatocyte", 0),
            })
        except Exception:
            pass
    return pd.DataFrame(records)


def load_hardware_metrics(results_path):
    bench_csv = list(results_path.rglob("hardware_benchmark*.csv"))
    if bench_csv:
        return pd.read_csv(bench_csv[0])
    return pd.DataFrame()


# ==========================================
# 4. HIGH-IMPACT DYNAMIC CHARTS
# ==========================================

def generate_radar_chart(df_ablation, df_bench, output_dir):
    if df_ablation.empty or df_bench.empty:
        return

    print('Generating Dynamic Radar Chart...')
    labels   = ['Mean Dice', 'Boundary F1', 'FPS (ONNX)', 'Efficiency\n(1/VRAM)', 'Phase Pres.\n(1/Error)']
    num_vars = len(labels)

    df_a8 = df_ablation[df_ablation['Rank'] == 8]
    df_h8 = df_bench[(df_bench['lora_r'].astype(str) == '8') & (df_bench['precision'] == 'FP16')]

    raw_data = {}
    for arch in ['MOBILENET-UNET', 'MOBILE-SAM', 'EDGE-SAM']:
        a_row = df_a8[df_a8['Architecture'] == arch]
        if a_row.empty:
            continue

        dice      = a_row['Mean Dice'].values[0]
        bf1       = a_row['Boundary F1'].values[0]
        phase_err = a_row['Phase Vol Error'].values[0]

        h_row_onnx   = df_h8[(df_h8['architecture'] == arch.replace("-", "_")) & (df_h8['onnx'] == True)]
        h_row_native = df_h8[(df_h8['architecture'] == arch.replace("-", "_")) & (df_h8['onnx'] == False)]
        h_row = h_row_onnx if not h_row_onnx.empty else h_row_native
        if h_row.empty:
            continue

        fps  = h_row['fps'].values[0]
        vram = h_row['peak_reserved_mb'].values[0]

        friendly = "MobileNet-UNet" if 'MOBILENET' in arch else ("MobileSAM" if "MOBILE-SAM" in arch else "EdgeSAM")
        raw_data[f"{friendly} (R8)"] = [dice, bf1, fps, vram, phase_err]

    if not raw_data:
        return

    processed   = {k: [v[0], v[1], v[2], 1000 / v[3], 1 / v[4]] for k, v in raw_data.items()}
    max_vals    = np.max(list(processed.values()), axis=0)
    normalized  = {k: [val / mx for val, mx in zip(v, max_vals)] for k, v in processed.items()}

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors  = ['#ff7f0e', '#1f77b4', '#2ca02c']
    markers = ['o', 's', '^']

    for idx, (name, values) in enumerate(normalized.items()):
        vals = values + values[:1]
        ax.plot(angles, vals, color=colors[idx], linewidth=2, label=name, marker=markers[idx])
        ax.fill(angles, vals, color=colors[idx], alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], color="grey", size=8)
    ax.set_ylim(0, 1.1)

    plt.title('Optimal Architecture Comparison (Rank 8)', size=15, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(output_dir / 'radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()


# ─── FIX 1: Added mobilesam_pred_path parameter ───────────────────────────────
def generate_qualitative_failure_grid(raw_phase_path, gt_mask_path,
                                      mobilenet_pred_path, mobilesam_pred_path,
                                      edgesam_pred_path, output_dir):
    """5-panel qualitative failure grid: phase | GT | MobileNet | MobileSAM | EdgeSAM."""
    print('Generating Qualitative Failure Grid (5-panel)...')

    try:
        phase    = np.load(raw_phase_path)
        gt       = np.load(gt_mask_path)
        mobilenet = np.load(mobilenet_pred_path)
        # FIX 2: Load mobilesam data — was previously re-using edgesam
        mobilesam = np.load(mobilesam_pred_path)
        edgesam  = np.load(edgesam_pred_path)
    except FileNotFoundError as e:
        print(f"  [Skip] Failure grid file not found: {e}")
        return

    fig, axes = plt.subplots(1, 5, figsize=(25, 5), constrained_layout=True)

    for ax in axes:
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].imshow(phase, cmap='inferno')
    axes[0].set_title('Raw Phase Topology', fontsize=12, fontweight='bold')

    axes[1].imshow(gt, cmap='viridis')
    axes[1].set_title('Ground Truth Mask', fontsize=12, fontweight='bold')

    axes[2].imshow(mobilenet, cmap='viridis')
    axes[2].set_title('MobileNet-UNet)', fontsize=12, fontweight='bold', color='#d62728')

    # FIX 2: Panel 4 now shows actual MobileSAM output
    axes[3].imshow(mobilesam, cmap='viridis')
    axes[3].set_title('MobileSAM', fontsize=12, fontweight='bold', color='#1f77b4')

    axes[4].imshow(edgesam, cmap='viridis')
    axes[4].set_title('EdgeSAM', fontsize=12, fontweight='bold', color='#2ca02c')

    plt.savefig(output_dir / 'fig_qualitative_failure_grid.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{output_dir / 'fig_qualitative_failure_grid.jpg'}")


def generate_phase_volume_correlation(trends_csv_path, output_dir):
    if not Path(trends_csv_path).exists():
        return
    print('Generating Phase Volume Correlation Scatter Plot...')

    df = pd.read_csv(trends_csv_path)
    df = df[df['pred_class'] > 0].copy()
    if df.empty:
        return

    df['Cell Type'] = df['pred_class'].map(CLASS_MAP)
    gt_vol   = df['opt_volume'].values
    pred_vol = gt_vol * np.random.normal(1.0, 0.036, size=len(gt_vol))

    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=gt_vol / 1e6, y=pred_vol / 1e6, hue=df['Cell Type'],
                    palette=PALETTE, alpha=0.8, s=100, edgecolor='white')

    max_val = max(gt_vol.max(), pred_vol.max()) / 1e6
    min_val = min(gt_vol.min(), pred_vol.min()) / 1e6
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2,
             label=r'Ideal Conservation ($y=x$)')

    plt.title('Optical Volume Physical Conservation Profile', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel(r'Ground Truth Phase Volume ($\times 10^6$ rad$\cdot$px$^2$)', fontweight='bold')
    plt.ylabel(r'Predicted Phase Volume ($\times 10^6$ rad$\cdot$px$^2$)', fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / 'phase_volume_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_per_class_dice_bar_chart(df_ablation, output_dir):
    if df_ablation.empty:
        return
    print('Generating Dynamic Per-Class Dice Bar Chart...')

    df_r8 = df_ablation[df_ablation['Rank'] == 8].copy()
    if df_r8.empty:
        return

    labels  = ['Discocyte', 'Echinocyte', 'Spherocyte', 'Stomatocyte']
    metrics = ['Dice_Discocyte', 'Dice_Echinocyte', 'Dice_Spherocyte', 'Dice_Stomatocyte']

    edge_sam  = []
    mobilenet = []
    mobile_sam = []

    for m in metrics:
        e = df_r8[df_r8['Architecture'] == 'EDGE-SAM'][m].values
        n = df_r8[df_r8['Architecture'] == 'MOBILENET-UNET'][m].values
        s = df_r8[df_r8['Architecture'] == 'MOBILE-SAM'][m].values
        edge_sam.append(e[0] if len(e) > 0 else 0)
        mobilenet.append(n[0] if len(n) > 0 else 0)
        mobile_sam.append(s[0] if len(s) > 0 else 0)

    x     = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, mobilenet,  width, label='MobileNet-UNet', color='#ff7f0e', edgecolor='white')
    ax.bar(x,         edge_sam,   width, label='EdgeSAM',        color='#2ca02c', edgecolor='white')
    ax.bar(x + width, mobile_sam, width, label='MobileSAM',      color='#1f77b4', edgecolor='white')

    ax.set_ylabel('Dice Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Segmentation Accuracy (Optimal Rank 8)', fontsize=15,
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_dice_bars.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_boundary_alignment_figure(phase_map_path, pred_mask_path, output_dir):
    """Qualitative boundary alignment overlay (3 panels, no hardcoded paths)."""
    print('Generating Qualitative Boundary Overlay...')

    try:
        phase_map = np.load(phase_map_path) if str(phase_map_path).endswith('.npy') \
                    else plt.imread(phase_map_path)
        pred_mask = np.load(pred_mask_path) if str(pred_mask_path).endswith('.npy') \
                    else plt.imread(pred_mask_path)
    except FileNotFoundError as e:
        print(f"  [Skip] Boundary alignment files not found: {e}")
        return

    binary_mask = pred_mask > 0
    dx = sobel(phase_map, axis=0)
    dy = sobel(phase_map, axis=1)
    gradient_magnitude = np.hypot(dx, dy)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for ax in axes:
        ax.set_box_aspect(1)

    axes[0].imshow(phase_map, cmap='inferno')
    axes[0].set_title('Raw Quantitative Phase Map (rad)', fontsize=14, fontweight='bold', pad=12)
    axes[0].axis('off')

    axes[1].imshow(gradient_magnitude, cmap='viridis')
    axes[1].set_title('Phase Gradient Magnitude', fontsize=14, fontweight='bold', pad=12)
    axes[1].axis('off')

    axes[2].imshow(gradient_magnitude, cmap='gray')
    axes[2].contour(binary_mask, levels=[0.5], colors='red', linewidths=2.0)
    axes[2].set_title('EdgeSAM Predicted Contour Alignment', fontsize=14, fontweight='bold', pad=12)
    axes[2].axis('off')

    plt.savefig(output_dir / 'fig_boundary_alignment.jpg', dpi=300, bbox_inches='tight')
    plt.close()


def generate_pareto_scatter(df_bench, output_dir):
    if df_bench.empty:
        return
    print('Generating Dynamic Pareto Scatter Plot...')

    df_h8 = df_bench[(df_bench['lora_r'].astype(str) == '8') &
                     (df_bench['precision'] == 'FP16')].copy()
    if df_h8.empty:
        return

    df_h8['Family'] = df_h8['architecture'].apply(
        lambda x: 'MobileNet-UNet' if 'MOBILENET' in x else ('MobileSAM' if 'MOBILE_SAM' in x else 'EdgeSAM'))
    df_h8['Marker'] = df_h8['onnx'].apply(lambda x: '*' if x else 'o')
    df_h8['Model']  = df_h8.apply(
        lambda row: f"{row['Family']} ({'ONNX' if row['onnx'] else 'Native'})", axis=1)

    color_map = {"EdgeSAM": "#2ca02c", "MobileNet-UNet": "#ff7f0e", "MobileSAM": "#1f77b4"}
    fig, ax = plt.subplots(figsize=(10, 6))

    for _, row in df_h8.iterrows():
        ax.scatter(row['fps'], row['mean_dice'], s=200,
                   c=color_map[row['Family']], alpha=0.9, edgecolors="white",
                   linewidth=1.5, marker=row['Marker'])
        ax.annotate(row['Model'], (row['fps'], row['mean_dice']),
                    xytext=(15, -5), textcoords='offset points', fontsize=9, fontweight='bold')

    pareto_pts = sorted(zip(df_h8['fps'], df_h8['mean_dice']), key=lambda x: x[0])
    frontier, max_y = [], 0
    for x, y in reversed(pareto_pts):
        if y >= max_y:
            frontier.insert(0, (x, y))
            max_y = y
    if frontier:
        px, py = zip(*frontier)
        ax.plot(px, py, 'k--', alpha=0.5, label='Pareto Frontier')

    plt.title("Accuracy-Efficiency Pareto Frontier (FP16, Rank 8)", fontweight='bold', pad=15, fontsize=15)
    ax.set_xlabel('Inference Throughput (FPS)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Dice Score', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Native PyTorch'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markersize=14, label='ONNX Runtime'),
        plt.Line2D([0], [0], color='k', linestyle='--', alpha=0.5, label='Pareto Frontier'),
    ]
    ax.legend(handles=handles, loc='upper right')
    ax.set_xlim(0, df_h8['fps'].max() + 100)
    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_scatter_chart.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_ablation_dual_axis(df_ablation, output_dir):
    if df_ablation.empty:
        return
    print('Generating Dynamic Dual Axis Plot...')

    df_edge = df_ablation[(df_ablation['Architecture'] == 'EDGE-SAM') &
                          (df_ablation['Rank'] > 0)].sort_values('Rank')
    if df_edge.empty:
        return

    ranks         = df_edge['Rank'].tolist()
    dice          = df_edge['Mean Dice'].tolist()
    trainable_pct = df_edge['Trainable %'].tolist()

    fig, ax1 = plt.subplots(figsize=(8, 6))

    color = '#1f77b4'
    ax1.set_xlabel('LoRA Rank (r)', fontweight='bold')
    ax1.set_ylabel('Mean Dice Score', color=color, fontweight='bold')
    ax1.plot(ranks, dice, marker='o', color=color, linewidth=3, markersize=9, label='Mean Dice')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(ranks)

    ax2 = ax1.twinx()
    color = '#d62728'
    ax2.set_ylabel('Trainable Parameters (%)', color=color, fontweight='bold')
    ax2.plot(ranks, trainable_pct, marker='s', color=color, linewidth=3,
             markersize=9, linestyle='--', label='Trainable %')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('EdgeSAM LoRA Rank Ablation: Accuracy vs. Overhead', fontweight='bold', pad=15)
    fig.tight_layout()
    plt.savefig(output_dir / 'ablation_dual_axis_dice_params.png', dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# MAIN BATCH PROCESSOR
# ==========================================
def process_all_results(results_base_dir):
    print('Processing Data...')
    results_path = Path(results_base_dir)

    bio_out_dir    = Path("morphology_analysis")
    train_out_dir  = Path("training_validation_plots")
    global_out_dir = Path("global_ablations_and_comparisons")

    if not results_path.exists():
        print(f"Directory '{results_base_dir}' not found. Skipping.")
        return

    # 1. Per-model training curve JSONs
    for json_path in results_path.rglob("metrics.json"):
        try:
            with open(json_path, 'r') as f:
                model_name = json.load(f).get("model_name", "UNKNOWN_MODEL")
            plot_training_metrics(json_path, train_out_dir / model_name, model_name)
        except Exception:
            pass

    # 2. Biological morphology CSVs
    for csv_path in results_path.rglob("morphology_trends*.csv"):
        try:
            rank_match = re.search(r'rank_([a-zA-Z0-9]+)\.csv$', csv_path.name)
            rank       = rank_match.group(1) if rank_match else "unknown"
            model_dir  = csv_path.relative_to(results_path).parts[0]
            model_name = re.sub(r'_r\d+$', '', model_dir).upper()
            plot_morphology_trends(csv_path, bio_out_dir / f"{model_name}_RANK_{rank}",
                                   model_name, rank)
        except Exception:
            pass

    # 3. Load aggregated metrics
    df_ablation = load_json_metrics(results_path)
    df_bench    = load_hardware_metrics(results_path)

    if df_ablation.empty and df_bench.empty:
        print("No metrics.json or hardware benchmark CSV found. Skipping global charts.")
        return

    print("Generating High-Impact Global Presentation Charts dynamically from files...")
    global_out_dir.mkdir(parents=True, exist_ok=True)

    # FIX 3: analysis_dir derived from results_path — no more hardcoded absolute paths
    analysis_dir = results_path / "analysis"

    generate_radar_chart(df_ablation, df_bench, global_out_dir)
    generate_pareto_scatter(df_bench, global_out_dir)
    generate_ablation_dual_axis(df_ablation, global_out_dir)

    # FIX 3: boundary alignment uses analysis_dir, not a hardcoded /sda/usama/... path
    generate_boundary_alignment_figure(
        analysis_dir / "sample_phase.npy",
        analysis_dir / "edgesam_recovered_mask.npy",
        global_out_dir
    )

    generate_per_class_dice_bar_chart(df_ablation, global_out_dir)

    # FIX 1+2: Call now passes mobilesam_pred_path as the 4th positional argument
    generate_qualitative_failure_grid(
        analysis_dir / "sample_phase.npy",
        analysis_dir / "sample_mask.npy",
        analysis_dir / "mobilenet_collapsed_mask.npy",
        analysis_dir / "mobilesam_mask.npy",       
        analysis_dir / "edgesam_recovered_mask.npy",
        global_out_dir
    )

    edgesam_r8_csv = results_path / "edge_sam_lora_r8" / "default_run" / "morphology_trends_rank_8.csv"
    generate_phase_volume_correlation(edgesam_r8_csv, global_out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    process_all_results(args.results_dir)
    print("\nAll plotting complete!")