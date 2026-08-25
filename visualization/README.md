# visualization/

Regenerates every **data-driven** figure in the manuscript from JSON, with a
single command and no CSV dependency.

```bash
python visualization.py
```

PNGs are written one directory up, into the manuscript folder, overwriting the
existing files.

## What is covered

| Manuscript | Output file | Content |
|---|---|---|
| Fig. 2  | `per_class_dice.png` | per-class Dice at r = 8 |
| Fig. 3  | `radar_chart.png` | aggregate performance radar |
| Fig. 4  | `ablation_curve.png` | LoRA rank sweep |
| Fig. 5  | `pareto_frontier.png` | accuracy vs throughput |
| Fig. 6  | `lora_vs_full_finetune.png` | LoRA vs full fine-tuning |
| Fig. 7  | `loss_ablation.png` | loss-component ablation |
| Fig. 11 | `population_shift.png` | morphology composition over storage |
| Fig. 12 | `drymass_validation.png` | dry-mass agreement and trajectory |

**Not covered**, because they are not generated from data: Fig. 1 (framework
overview, a hand-built diagram) and Figs. 8–10 (microscopy montages, which are
measured image data).

## Layout

```
visualization/
├── visualization.py     all plotting logic, one function per figure
├── data/
│   ├── figure02.json    each file carries the numbers for one figure,
│   ├── figure03.json    plus a "description" and "source" field saying
│   ├── ...              which result table it came from
│   └── figure12.json
└── README.md
```

Data files are named after the manuscript figure number, so `figure07.json`
holds Fig. 7. Values are stored as they are plotted; the only derived quantity
computed at draw time is the radar normalisation (each axis divided by the best
model), which is done in the script so the raw values stay inspectable.

## Restyling

Every visual parameter is in the `STYLE` dictionary at the top of
`visualization.py` — colours per architecture and per cell class, font family
and eight separate font sizes, line widths, marker sizes and styles, bar widths,
grid appearance, spine visibility, figure widths and DPI, and the printed labels
for architectures and classes.

Changing a colour everywhere is one edit:

```python
"colour_by_architecture": {
    "EDGE_SAM":       "#1B9E77",   # <- change here, applies to all figures
    ...
}
```

Nothing inside the plotting functions hard-codes a colour, size or font, so the
appearance can be changed without touching the data or the drawing logic.

## Reproducibility notes

- Figure widths are the true IEEE column widths (3.50 in single, 7.16 in
  double), so `\includegraphics[width=\columnwidth]` applies no scaling and
  text renders at the size it was set in.
- Plotting runs inside `plt.style.context("default")`, so an active
  `matplotlibrc` or seaborn theme in the user's environment cannot change the
  result.
- Verified on a clean copy with no CSVs present: all eight figures regenerate.

## Requirements

```bash
pip install matplotlib numpy
```
