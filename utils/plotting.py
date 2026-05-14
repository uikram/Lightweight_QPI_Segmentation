"""
Visualization utilities for generating comparison plots and training curves.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


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
