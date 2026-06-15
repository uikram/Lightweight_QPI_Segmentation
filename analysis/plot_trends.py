import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Mapping of AI class IDs to their biological names
CLASS_MAP = {
    1: 'Discocyte',
    2: 'Echinocyte',
    3: 'Spherocyte',
    4: 'Stomatocyte'
}

# Standardized color palette for consistent paper figures
PALETTE = {
    'Discocyte': '#2ca02c',   # Green
    'Echinocyte': '#ff7f0e',  # Orange
    'Spherocyte': '#d62728',  # Red
    'Stomatocyte': '#1f77b4'  # Blue
}

def plot_morphology_trends(csv_path, output_dir=None):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV file at: {csv_path}")
        
    if output_dir is None:
        output_dir = csv_path.parent / "plots"
    else:
        output_dir = Path(output_dir)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter out background (Class 0)
    df = df[df['pred_class'] > 0].copy()
    
    # Map class IDs to string names
    df['Cell Type'] = df['pred_class'].map(CLASS_MAP)
    
    # Set seaborn style for publication-ready scientific plots
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    metrics = {
        'area': ('Projected Area (pixels)', 'Projected Area Degradation'),
        'circularity': ('Circularity', 'Circularity Degradation'),
        'opt_volume': ('Phase-Integrated Optical Volume', 'Optical Volume Consistency')
    }

    # 1. Plot the three physical metrics over time
    for col, (y_label, title) in metrics.items():
        plt.figure(figsize=(8, 6))
        
        # lineplot automatically calculates the mean and 95% Confidence Interval bands!
        sns.lineplot(
            data=df, 
            x='storage_day', 
            y=col, 
            hue='Cell Type', 
            palette=PALETTE,
            marker='o',
            linewidth=2.5,
            err_style='band'
        )
        
        plt.title(f"{title} Over Storage Time", fontweight='bold', pad=15)
        plt.xlabel('Storage Day', fontweight='bold')
        plt.ylabel(y_label, fontweight='bold')
        plt.legend(title='Predicted Morphology')
        plt.tight_layout()
        
        save_path = output_dir / f"trend_{col}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    # 2. Plot Population Distribution Over Time (Cell Degradation Count)
    # This shows how Discocytes turn into Echinocytes/Spherocytes over time
    plt.figure(figsize=(10, 6))
    count_df = df.groupby(['storage_day', 'Cell Type']).size().reset_index(name='Count')
    
    # Calculate percentage composition per day
    total_per_day = count_df.groupby('storage_day')['Count'].transform('sum')
    count_df['Percentage'] = (count_df['Count'] / total_per_day) * 100

    sns.histplot(
        data=df, 
        x='storage_day', 
        hue='Cell Type', 
        multiple='fill', 
        palette=PALETTE, 
        bins=len(df['storage_day'].unique()),
        shrink=0.8
    )
    
    plt.title("RBC Morphology Population Shift Over Storage Time", fontweight='bold', pad=15)
    plt.xlabel('Storage Day', fontweight='bold')
    plt.ylabel('Proportion of Cells', fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / "trend_population_shift.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    print("\nAll visualizations generated successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Biological Trend Visualizations from CSV")
    parser.add_argument("--csv", type=str, required=True, help="Path to the morphology_trends.csv file")
    parser.add_argument("--out", type=str, default=None, help="Output directory for plots (defaults to a 'plots' folder next to the CSV)")
    
    args = parser.parse_args()
    plot_morphology_trends(args.csv, args.out)