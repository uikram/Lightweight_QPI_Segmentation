# Physics-Aware LoRA-Adapted Lightweight AI for Edge-Deployable QPI Segmentation

This repository contains the official PyTorch implementation of the Physics-Aware LoRA-Adapted Lightweight AI framework for Quantitative Phase Imaging (QPI).

This project bridges the domain gap between natural RGB-pretrained models and holographic phase topologies. By applying Low-Rank Adaptation (LoRA) to ultra-efficient architectures (EdgeSAM, MobileSAM, and MobileNet-UNet) and enforcing a custom Physics-Aware Phase Consistency Loss, the framework enables real-time, edge-deployable cell segmentation while strictly preserving physically meaningful biological metrics (e.g., phase-integrated optical volume and cellular dry mass).

## ✨ Key Features

* **Lightweight Architectures:** Implements highly distilled base models including EdgeSAM (RepViT), MobileSAM (TinyViT), and MobileNet-UNet.

* **Parameter-Efficient Fine-Tuning (PEFT):** Integrates customizable LoRA layers into model bottlenecks and attention blocks to bypass full-network fine-tuning.

* **Physics-Aware Optimization:** Features a custom compound loss function combining Multi-Class Dice, Phase-Mask Contrast, Boundary-Gradient Alignment, and Phase-Volume Preservation.

* **Edge Benchmarking:** Dedicated profiling scripts to measure PyTorch/ONNX inference latency, throughput (FPS), and peak VRAM allocation across different precisions (FP32/FP16).

* **Biological Morphology Tracking:** Automated extraction of 2D/3D physical traits (projected area, circularity, optical volume) to map the timeline of the red blood cell storage lesion.

## 📂 Repository Structure

```text
Lightweight_QPI_Segmentation/
│
├── main.py                     # Primary entry point for training, evaluation, and sweeps
├── collect_metrics.py          # Script to aggregate output metrics to JSON
│
├── configs/                    # YAML configuration files for each architecture
│   ├── edge_sam_lora.yaml
│   ├── mobile_sam_lora.yaml
│   └── mobilenet_unet_lora.yaml
│
├── datasets/                   # Data loading and augmentation pipelines
│   ├── qpi_dataset.py          # QPI dataloader and mask parsing
│   └── qpi_augmentation.py     # Physics-preserving offline augmentations (rotations, flips)
│
├── models/                     # Model architectures and LoRA integration
│   ├── edge_sam.py             # RepViT-based EdgeSAM implementation
│   ├── mobile_sam.py           # TinyViT-based MobileSAM implementation
│   ├── mobilenet_unet.py       # MobileNetV2-UNet baseline implementation
│   └── lora_utils.py           # LoRA rank injection and weight freezing logic
│
├── training/                   # Core training and optimization loops
│   ├── train.py                # Setup for optimizers and schedulers
│   ├── trainer_seg.py          # Iteration loops and mixed-precision (FP16) logic
│   └── losses.py               # Implementation of Physics-Aware Phase Consistency Loss
│
├── evaluation/                 # Testing and validation metrics
│   ├── evaluate.py             # Evaluation loop for validation sets
│   ├── metrics.py              # Centralized metric calculations
│   ├── seg_metrics.py          # Mean Dice, AJI, IoU, and Boundary F1 calculations
│   └── profiling.py            # Hardware tracking (VRAM, MACs, Params)
│
├── benchmark/                  # Edge deployment simulations
│   ├── benchmark.py            # Hardware execution script (Latency, FPS, Memory)
│
├── analysis/                   # Biological and statistical tracking
│   ├── morphology_analysis.py  # Area, circularity, and phase-volume extraction
│   ├── plot_trends.py          # Generates longitudinal storage lesion visuals
│   └── sample_*.npy            # Sample phase maps and masks for quick testing
│
└── utils/                      # Helper functions
    ├── config.py               # YAML parser
    ├── helpers.py              # General file I/O and random seeding
    ├── plotting.py             # Qualitative grid and scatter plot generators
    └── transforms.py           # Image normalizations
```

## 📁 Dataset Preparation

The framework expects single-channel `.tif` or `.npy` files. Phase images should be `float32` (radians), and masks should be integers (values 0-4 for multi-class morphology).

Organize your data exactly as follows:

```text
dataset/
├── X_train/             # Training phase maps
├── Y_train/             # Training semantic masks (0=bg, 1=disco, 2=echino, 3=sphero, 4=stomato)
├── X_val/               # Validation phase maps
├── Y_val/               # Validation semantic masks
├── labels_train.csv     # (Optional) CSV with 'filename' and 'morphology_class'
└── labels_val.csv       # (Optional) CSV with 'filename' and 'morphology_class'
```

## 🚀 Getting Started & Usage

### 1. Installation

Clone the repository and install the required dependencies (requires Python 3.8+ and PyTorch):

```bash
git clone https://github.com/YourUsername/Lightweight_QPI_Segmentation.git
cd Lightweight_QPI_Segmentation
pip install torch torchvision numpy scipy tifffile pillow PyYAML tqdm
```

> **Note:** For ONNX hardware benchmarking, ensure `onnxruntime-gpu` is installed. For SAM models, ensure their respective pip packages are installed if utilizing official backbones.

### 2. Training & LoRA Rank Sweeps

To initiate training or sweep across multiple LoRA ranks to find the Pareto-optimal configuration, use `main.py` along with the desired architecture configuration. For example, to sweep across ranks 2, 4, 8, 16, and 32 for MobileSAM on GPU 2:

```bash
CUDA_VISIBLE_DEVICES=2 python main.py --mode sweep --config configs/mobile_sam_lora.yaml --ranks 2 4 8 16 32 --gpu 0
```

### 3. Aggregating Metrics

Once training or rank sweeps are complete, run the metric aggregation script to collect and compile the segmentation accuracy and physics-aware metrics into a unified JSON file:

```bash
python collect_metrics.py
```

### 4. Hardware Benchmarking

To evaluate inference latency, throughput (FPS), and memory footprint across different configurations, use the benchmarking script. You can test various ranks and precisions (e.g., FP32 vs. FP16) to evaluate ONNX vs. PyTorch native execution:

```bash
CUDA_VISIBLE_DEVICES=2 python benchmark/benchmark.py --ranks 0 2 4 8 16 32 --precisions fp32 fp16
```

### 5. Biological Morphology Analysis

To extract geometric trajectories and plot the longitudinal storage lesion trends based on model predictions (e.g., area vs. circularity, population shift):

```bash
python analysis/plot_trends.py
```

## ⚙️ Configuration Example

Hyperparameters and loss weights are strictly controlled via YAML files in the `configs/` directory. Example structure (`edge_sam_lora.yaml`):

```yaml
model:
  architecture: "edge_sam"
  in_channels: 1
  num_classes: 5

lora:
  lora_r: 8
  lora_alpha: 8.0
  insertion_strategy: "bottleneck"

training:
  epochs: 50
  learning_rate: 0.0005
  mixed_precision: "fp16"

loss_weights:
  lambda1_pmc: 0.1
  lambda2_bga: 0.05
  lambda3_pv: 0.1
```

## 🔬 Core Paper Message

> "Physics-aware LoRA adaptation enables lightweight and edge-deployable quantitative holographic cell analysis while preserving biologically meaningful phase information."
