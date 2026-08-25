#!/usr/bin/env python3
"""
visualization.py - regenerate every data-driven figure in the manuscript.

    python visualization.py

Reads the JSON files in ./data/ and writes the PNGs one directory up, into the
manuscript folder, overwriting the existing figures.

Only figures produced from data are handled here. Figure 1 (framework overview)
and Figures 8-10 (microscopy montages) are not data-driven and are left alone.

    Fig. 2   per_class_dice.png          per-class Dice at r=8
    Fig. 3   radar_chart.png             aggregate performance radar
    Fig. 4   ablation_curve.png          LoRA rank sweep
    Fig. 5   pareto_frontier.png         accuracy vs throughput
    Fig. 6   lora_vs_full_finetune.png   LoRA versus full fine-tuning
    Fig. 7   loss_ablation.png           loss-component ablation
    Fig. 11  population_shift.png        morphology composition over storage
    Fig. 12  drymass_validation.png      dry-mass agreement and trajectory
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ===========================================================================
#  STYLE - Matching exact manuscript parameters
# ===========================================================================

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent/"visualization/figures"

STYLE = {
    "width_single_col": 3.50,
    "width_double_col": 7.16,
    "dpi": 600,
    "facecolor": "white",

    # --- typography -------------------------------------------------------
    "font_family": "DejaVu Sans",
    "font_size_base": 8.0,
    "font_size_axis_label": 8.5,
    "font_size_tick": 7.5,
    "font_size_legend": 7.5,
    "font_size_title": 8.5,
    "font_size_annotation": 7.0,
    "font_size_bar_value": 5.4,          
    "font_size_bar_value_paired": 6.6,   
    "font_size_zero_marker": 6.0,        
    "font_size_zero_marker_paired": 8.0,
    "font_size_secondary_axis_label": 8.0,   
    "font_size_radar_tick": 7.0,
    "font_size_pareto_label": 7.2,
    "font_size_panel_title": 8.5,
    "axis_label_weight": "bold",

    # --- colours ----------------------------------------------------------
    "colour_by_architecture": {
        "EDGE_SAM":       "#1B9E77",
        "MOBILE_SAM":     "#7570B3",
        "MOBILENET_UNET": "#D95F02",
    },
    "colour_by_class": {           
        1: "#457B9D",
        2: "#E9C46A",
        3: "#E76F51",
        4: "#8E7DBE",
    },
    "colour_full_finetune": "#C0564F",
    "colour_error":         "#B3261E",   
    "colour_frontier":      "#4F6D7A",
    "colour_neutral":       "#777777",
    "colour_reference_line": "#888888",

    # --- labels -----------------------------------------------------------
    "architecture_labels": {
        "EDGE_SAM": "EdgeSAM",
        "MOBILE_SAM": "MobileSAM",
        "MOBILENET_UNET": "MobileNet-UNet",
    },
    "class_labels_short": ["Disco.", "Echino.", "Sphero.", "Stomato."],
    "class_labels_full": ["Discocyte", "Echinocyte", "Spherocyte", "Stomatocyte"],
}

def _rc() -> dict:
    return {
        "font.family": STYLE["font_family"],
        "font.size": STYLE["font_size_base"],
        "axes.labelsize": STYLE["font_size_axis_label"],
        "axes.titlesize": STYLE["font_size_title"],
        "xtick.labelsize": STYLE["font_size_tick"],
        "ytick.labelsize": STYLE["font_size_tick"],
        "legend.fontsize": STYLE["font_size_legend"],
        "axes.labelweight": STYLE["axis_label_weight"],
        "axes.linewidth": 0.7,
        "axes.grid": False,
    }


def load(name: str) -> dict:
    with open(DATA_DIR / name) as fh:
        return json.load(fh)

def apply_axis_style(ax, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis, ls=":", lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

def save(fig, filename: str) -> None:
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight",
                facecolor=STYLE["facecolor"])
    plt.close(fig)
    print(f"  wrote {path}")

def architecture_colour(arch: str) -> str:
    return STYLE["colour_by_architecture"][arch]

def architecture_label(arch: str) -> str:
    return STYLE["architecture_labels"][arch]


# ===========================================================================
#  Figure 2 - per-class Dice
# ===========================================================================

def figure02_per_class_dice() -> None:
    d = load("figure02.json")
    order = ["MOBILENET_UNET", "EDGE_SAM", "MOBILE_SAM"]
    width = 0.26
    x = np.arange(len(d["classes"]))

    fig, ax = plt.subplots(figsize=(STYLE["width_single_col"], 2.35))
    for i, arch in enumerate(order):
        values = [0.0 if v < 1e-3 else v for v in d["series"][arch]]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=architecture_label(arch),
                      color=architecture_colour(arch),
                      edgecolor="black", linewidth=0.4)
        for b in bars:
            h = b.get_height()
            if h < 1e-3:
                ax.text(b.get_x() + b.get_width() / 2, 0.015, "0", ha="center",
                        va="bottom", fontsize=6, color=STYLE["colour_error"], 
                        fontweight="bold")
            else:
                ax.text(b.get_x() + b.get_width() / 2, h + 0.015, f"{h:.2f}",
                        ha="center", va="bottom", fontsize=5.4, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(STYLE["class_labels_short"])
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=True, framealpha=0.92, edgecolor="#ccc",
              loc="upper right", ncol=1, handlelength=1.2, borderpad=0.35)
    apply_axis_style(ax, "y")
    fig.tight_layout(pad=0.3)
    save(fig, "per_class_dice.png")


# ===========================================================================
#  Figure 3 - aggregate performance radar
# ===========================================================================

def figure03_radar() -> None:
    d = load("figure03.json")
    order = ["EDGE_SAM", "MOBILE_SAM", "MOBILENET_UNET"]
    raw = np.array([d["series"][a] for a in order], dtype=float)
    best = raw.max(axis=0)                      
    best[best == 0] = 1.0
    normalised = raw / best

    n_axes = raw.shape[1]
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    side = STYLE["width_single_col"]
    fig, ax = plt.subplots(figsize=(side, side * 0.92), subplot_kw=dict(polar=True))
    for arch, values in zip(order, normalised):
        closed = values.tolist() + [values[0]]
        ax.plot(angles, closed, color=architecture_colour(arch),
                lw=1.4, marker="o", ms=3, label=architecture_label(arch))
        ax.fill(angles, closed, color=architecture_colour(arch), alpha=0.10)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    
    # Enforce labels to match paper exactly
    ax.set_xticklabels(["Mean Dice", "AJI", "Boundary F1", "Throughput", "Memory eff."], 
                       fontsize=STYLE["font_size_radar_tick"])
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([], fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.grid(lw=0.4, alpha=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3,
              frameon=False, columnspacing=1.0, handlelength=1.2)
    fig.tight_layout(pad=0.3)
    save(fig, "radar_chart.png")


# ===========================================================================
#  Figure 4 - LoRA rank sweep
# ===========================================================================

def figure04_rank_sweep() -> None:
    d = load("figure04.json")
    ranks = d["ranks"]
    dice = d["mean_dice"]
    params = d["trainable_params_millions"]
    selected = d["selected_rank"]
    idx = ranks.index(selected)
    positions = np.arange(len(ranks))

    fig, ax = plt.subplots(figsize=(STYLE["width_single_col"], 2.3))
    ax_params = ax.twinx()
    ax_params.bar(positions, params, 0.5, color="#cccccc",
                  edgecolor="#999999", lw=0.4, zorder=1)
    ax_params.set_ylabel("Trainable params (M)", fontsize=7.5)
    ax_params.set_ylim(0, 1.8)
    ax_params.tick_params(labelsize=7)
    ax_params.spines["top"].set_visible(False)

    colour = architecture_colour("EDGE_SAM")
    ax.plot(positions, dice, "o-", color=colour, lw=1.6, ms=4.5, zorder=3)
    ax.scatter([positions[idx]], [dice[idx]], s=90, facecolor="none",
               edgecolor=colour, lw=1.2, ls=(0, (2, 1.4)), zorder=4)
    ax.annotate("selected", (positions[idx], dice[idx]),
                textcoords="offset points", xytext=(0, 13), ha="center",
                fontsize=6.5, fontweight="bold", color=colour)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"$r$={k}" for k in ranks])
    ax.set_ylabel("Mean Dice")
    ax.set_ylim(0.58, 0.80)
    ax.set_zorder(ax_params.get_zorder() + 1)
    ax.patch.set_visible(False)
    apply_axis_style(ax, "y")
    fig.tight_layout(pad=0.3)
    save(fig, "ablation_curve.png")


# ===========================================================================
#  Figure 5 - accuracy versus throughput
# ===========================================================================

def figure05_pareto() -> None:
    d = load("figure05.json")
    pts = d["points"]

    def dominated(p):
        return any(q["fps"] >= p["fps"] and q["mean_dice"] >= p["mean_dice"]
                   and (q["fps"] > p["fps"] or q["mean_dice"] > p["mean_dice"])
                   for q in pts if q is not p)

    frontier = sorted([p for p in pts if not dominated(p)], key=lambda p: p["fps"])

    fig, ax = plt.subplots(figsize=(STYLE["width_single_col"], 2.75))
    x_max = max(p["fps"] for p in pts) * 1.135
    y_min, y_max, band = 0.325, 0.885, 0.505

    ax.axhspan(y_min, band, color=STYLE["colour_error"], alpha=0.06, lw=0, zorder=0)
    ax.axhline(band, color=STYLE["colour_error"], lw=0.6, ls=(0, (3, 2)),
               alpha=0.55, zorder=1)

    step_x, step_y = [0.0], [frontier[0]["mean_dice"]]
    for p in frontier:
        step_x += [p["fps"], p["fps"]]
        step_y += [step_y[-1], p["mean_dice"]]
    step_x.append(x_max)
    step_y.append(step_y[-1])
    ax.plot(step_x, step_y, color=STYLE["colour_frontier"], ls="--", lw=0.9, zorder=2)

    for arch in sorted({p["architecture"] for p in pts}):
        native = next((p for p in pts if p["architecture"] == arch and p["runtime"] == "native"), None)
        onnx = next((p for p in pts if p["architecture"] == arch and p["runtime"] == "ONNX"), None)
        if native and onnx:
            ax.annotate("", xy=(onnx["fps"], onnx["mean_dice"]),
                        xytext=(native["fps"], native["mean_dice"]),
                        arrowprops=dict(arrowstyle="-|>,head_width=0.16,head_length=0.34",
                                        lw=0.9, color=architecture_colour(arch),
                                        alpha=0.75, shrinkA=6, shrinkB=8,
                                        connectionstyle="arc3,rad=-0.16"), zorder=3)

    sel = next(p for p in pts if p["architecture"] == d["selected"]["architecture"]
               and p["runtime"] == d["selected"]["runtime"])
    ax.scatter(sel["fps"], sel["mean_dice"], s=260, marker="o", facecolor="none",
               edgecolor=architecture_colour(sel["architecture"]), lw=1.1,
               ls=(0, (2, 1.5)), zorder=3)

    for p in pts:
        colour = architecture_colour(p["architecture"])
        ax.scatter(p["fps"], p["mean_dice"],
                   s=100 if p["runtime"] == "ONNX" else 54,
                   marker="*" if p["runtime"] == "ONNX" else "o",
                   facecolor="white" if p["collapsed_classes"] else colour,
                   edgecolor=colour, linewidths=1.2, zorder=5)

    label_offsets = {"MOBILE_SAM": (13, 1, "left"),
                     "EDGE_SAM": (-2, -14, "center"),
                     "MOBILENET_UNET": (0, -15, "center")}
    for arch in sorted({p["architecture"] for p in pts}):
        anchor = next((p for p in pts if p["architecture"] == arch and p["runtime"] == "native"),
                      next(p for p in pts if p["architecture"] == arch))
        dx, dy, ha = label_offsets[arch]
        ax.annotate(architecture_label(arch), xy=(anchor["fps"], anchor["mean_dice"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=STYLE["font_size_pareto_label"], ha=ha, va="center",
                    color=architecture_colour(arch), fontweight="bold", zorder=6)

    ax.set_xlim(0, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_xlabel("Inference throughput (FPS)")
    ax.set_ylabel("Mean Dice")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc="#555555", mec="#555555", ms=4.4,
               label="Native PyTorch"),
        Line2D([], [], marker="*", ls="", mfc="#555555", mec="#555555", ms=8,
               label="ONNX Runtime"),
        Line2D([], [], ls="--", color=STYLE["colour_frontier"],
               lw=0.9, label="Pareto frontier"),
        Patch(facecolor=STYLE["colour_error"], alpha=0.14,
              edgecolor=STYLE["colour_error"], lw=0.7, ls="--", label="Mode collapse")],
        loc="upper right", bbox_to_anchor=(1.012, 1.03), frameon=True,
        framealpha=0.94, edgecolor="#D0D0D0", fancybox=False,
        handletextpad=0.45, borderpad=0.38, labelspacing=0.3, handlelength=1.5)
    apply_axis_style(ax)
    fig.tight_layout(pad=0.35)
    save(fig, "pareto_frontier.png")


# ===========================================================================
#  Figure 6 - LoRA versus full fine-tuning
# ===========================================================================

def figure06_lora_vs_full() -> None:
    d = load("figure06.json")
    order = ["EDGE_SAM", "MOBILE_SAM", "MOBILENET_UNET"]
    x = np.arange(len(d["classes"]))

    fig, axes = plt.subplots(1, 3, figsize=(STYLE["width_double_col"], 2.95), sharey=True)
    for ax, arch in zip(axes, order):
        lora = [0 if v < 1e-3 else v for v in d["series"][arch]["lora"]]
        full = [0 if v < 1e-3 else v for v in d["series"][arch]["full_finetune"]]
        
        ax.bar(x - 0.19, lora, 0.36, color=architecture_colour(arch),
               edgecolor="black", linewidth=0.4, label="LoRA ($r$=8)")
        ax.bar(x + 0.19, full, 0.36, color=STYLE["colour_full_finetune"],
               edgecolor="black", linewidth=0.4, label="Full fine-tune")
        
        for xi, (lo, fu) in enumerate(zip(lora, full)):
            for off, val, dy in ((-0.19, lo, 0.022), (0.19, fu, 0.075)):
                if val < 1e-3:
                    ax.text(xi + off, 0.025, "0", ha="center", fontsize=8,
                            color=STYLE["colour_error"], fontweight="bold")
                else:
                    ax.text(xi + off, val + dy, f"{val:.2f}", ha="center", fontsize=6.6)

        ax.set_xticks(x)
        ax.set_xticklabels(STYLE["class_labels_short"], fontsize=8)
        ax.set_title(architecture_label(arch), fontsize=9.5, fontweight="bold")
        ax.set_ylim(0, 1.14)
        ax.tick_params(labelsize=8)
        apply_axis_style(ax, "y")

    axes[0].set_ylabel("Per-class Dice", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.015), handlelength=1.3,
               columnspacing=1.6, fontsize=8.5)
    fig.tight_layout(pad=0.3, rect=(0, 0.08, 1, 1))
    save(fig, "lora_vs_full_finetune.png")


# ===========================================================================
#  Figure 7 - loss-component ablation
# ===========================================================================

def figure07_loss_ablation() -> None:
    d = load("figure07.json")
    labels = ["$\\mathcal{L}_{Dice}$", "$+\\mathcal{L}_{PMC}$",
              "$+\\mathcal{L}_{BGA}$", "Full $\\mathcal{L}$"]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(STYLE["width_double_col"], 2.55))
    for ax, arch in zip(axes, d["panel_order"]):
        bf1 = d["series"][arch]["boundary_f1"]
        pve = d["series"][arch]["phase_volume_error_pct"]
        ax.bar(x, bf1, 0.55, color=architecture_colour(arch),
               edgecolor="black", linewidth=0.4)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_title(architecture_label(arch), fontsize=9.5, fontweight="bold")
        ax.tick_params(labelsize=7.5)
        apply_axis_style(ax, "y")

        ax_err = ax.twinx()
        ax_err.plot(x, pve, "s--", color=STYLE["colour_error"], lw=1.2, ms=4, zorder=5)
        ax_err.set_ylim(0, max(pve) * 1.35)
        ax_err.tick_params(labelsize=7, colors=STYLE["colour_error"])
        ax_err.spines["top"].set_visible(False)
        for xi, value in zip(x, pve):
            ax_err.annotate(f"{value:.1f}", (xi, value), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=7,
                            color=STYLE["colour_error"], fontweight="bold",
                            bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
        if arch == d["panel_order"][-1]:
            ax_err.set_ylabel("PVE (%)", color=STYLE["colour_error"], fontsize=8)

    axes[0].set_ylabel("Boundary F1", fontsize=9)
    fig.legend(handles=[
        Line2D([], [], color=STYLE["colour_neutral"], lw=4, label="Boundary F1"),
        Line2D([], [], color=STYLE["colour_error"], ls="--", marker="s", ms=3.5,
               lw=1.2, label="Phase-volume error (%)")],
        loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.015),
        handlelength=1.4, columnspacing=1.6, fontsize=8.5)
    fig.tight_layout(pad=0.3, rect=(0, 0.08, 1, 1))
    save(fig, "loss_ablation.png")

# ===========================================================================
#  Figure 11 - morphology composition across storage
# ===========================================================================

def figure11_population_shift() -> None:
    d = load("figure11.json")
    days = d["storage_days"]
    totals = np.array(d["total_per_day"], dtype=float)
    positions = np.arange(len(days))

    # 1. Increased the image size to 5.0 x 3.2 inches for more breathing room
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    bottom = np.zeros(len(days))
    
    for class_id, name in zip(d["class_ids"], STYLE["class_labels_full"]):
        counts = np.array(d["cell_counts"][str(class_id)], dtype=float)
        fraction = counts / totals
        # 2. Increased bar width from 0.78 to 0.88 to make the bars bigger
        ax.bar(positions, fraction, 0.88, bottom=bottom,
               color=STYLE["colour_by_class"][class_id], edgecolor="white",
               linewidth=0.4, label=name)
        bottom += fraction

    ax.set_xticks(positions)
    
    # 3. Made the tick labels (the text in the middle) smaller
    ax.set_xticklabels(days, fontsize=6.5)
    ax.tick_params(axis='y', labelsize=6.5)
    
    ax.set_xlabel("Storage day", fontsize=7.5)
    
    # With the larger image size, we can safely reset y to center without clipping
    ax.set_ylabel("Predicted population fraction", fontsize=7.5)
    
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.6, len(days) - 0.4)
    
    # Legend positioned to fit nicely under the larger canvas
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4,
              frameon=False, handlelength=1.0, columnspacing=0.9, 
              fontsize=6.5)
              
    apply_axis_style(ax, "y")
    fig.tight_layout(pad=0.5)
    
    path = OUTPUT_DIR / "population_shift.png"
    fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight", pad_inches=0.05, facecolor=STYLE["facecolor"])
    plt.close(fig)
    print(f"  wrote {path} (with increased figure size and wider bars)")

# ===========================================================================
#  Figure 12 - dry-mass validation
# ===========================================================================

def figure12_drymass_validation() -> None:
    d = load("figure12.json")
    a, b = d["panel_a"], d["panel_b"]

    fig, axes = plt.subplots(1, 2, figsize=(STYLE["width_double_col"], 2.85))

    # --- (a) prediction against annotation, coloured by storage day ---------
    ax = axes[0]
    ground_truth = np.array(a["ground_truth_pg"])
    predicted = np.array(a["predicted_pg"])
    limits = [min(ground_truth.min(), predicted.min()) * 0.9,
              max(ground_truth.max(), predicted.max()) * 1.05]
    
    ax.plot(limits, limits, ls="--", lw=0.8, color=STYLE["colour_reference_line"], zorder=1)
    scatter = ax.scatter(ground_truth, predicted, c=a["storage_day"],
                         cmap="viridis", s=22, edgecolor="k", linewidth=0.4, zorder=3)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Dry mass, ground-truth masks (pg)")
    ax.set_ylabel("Dry mass, EdgeSAM (pg)")
    
    # Text override using explicit biases formatting observed from the sample plot
    # The slash has been removed so it formats as a clean percentage (e.g. +1.5%)
    bias_str = f"{a['median_bias_pct']:+.1f}%"
    ax.text(0.04, 0.94, f"$r$ = {a['pearson_r']:.3f}\nmedian bias {bias_str}",
            transform=ax.transAxes, va="top", fontsize=8, fontweight="bold")
    ax.set_title("(a) agreement with annotation", fontsize=8.5, fontweight="bold")
    colourbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03)
    colourbar.set_label("storage day", fontsize=7.0)
    colourbar.ax.tick_params(labelsize=6.5)

    # --- (b) area and mass relative to day 0 --------------------------------
    ax = axes[1]
    ax.axhline(100, color="#bbb", lw=0.7, ls="--", zorder=1)
    ax.plot(b["storage_days"], b["area_pct_of_day0"], "o-",
            color=architecture_colour("EDGE_SAM"), lw=1.4, ms=4.2, label="Projected area")
    ax.plot(b["storage_days"], b["dry_mass_pct_of_day0"], "s-",
            color=STYLE["colour_full_finetune"], lw=1.4, ms=4.2, label="Dry mass")
    ax.set_xlabel("Storage day")
    ax.set_ylabel("% of day 0 (median)")
    ax.set_xlim(-3, 52)
    ax.set_ylim(50, 148)
    
    ax.legend(loc="lower left", frameon=True, framealpha=0.94, edgecolor="#ccc",
              fancybox=False, handlelength=1.5, borderpad=0.4, fontsize=8)
    ax.set_title("(b) area contracts faster than mass", fontsize=8.5, fontweight="bold")

    for ax in axes:
        apply_axis_style(ax)
    fig.tight_layout(pad=0.4)
    save(fig, "drymass_validation.png")


# ===========================================================================
#  Entry point
# ===========================================================================

FIGURES = [
    ("Fig. 2  per-class Dice",            figure02_per_class_dice),
    ("Fig. 3  aggregate radar",           figure03_radar),
    ("Fig. 4  LoRA rank sweep",           figure04_rank_sweep),
    ("Fig. 5  accuracy vs throughput",    figure05_pareto),
    ("Fig. 6  LoRA vs full fine-tuning",  figure06_lora_vs_full),
    ("Fig. 7  loss-component ablation",   figure07_loss_ablation),
    ("Fig. 11 morphology composition",    figure11_population_shift),
    ("Fig. 12 dry-mass validation",       figure12_drymass_validation),
]

def main() -> None:
    print(f"data   : {DATA_DIR.resolve()}")
    print(f"output : {OUTPUT_DIR.resolve()}\n")
    with plt.style.context("default"), plt.rc_context(_rc()):
        for description, build in FIGURES:
            print(description)
            build()
    print(f"\n{len(FIGURES)} data-driven figures regenerated.")

if __name__ == "__main__":
    main()