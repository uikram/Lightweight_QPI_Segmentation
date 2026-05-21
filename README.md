# Physics-Aware LoRA-Adapted Lightweight AI for Edge-Deployable QPI Cell Analysis

This repository contains the official PyTorch implementation of the framework proposed for lightweight, physics-aware holographic cell analysis. The codebase adapts highly efficient, natural-image pretrained vision models (MobileNet-UNet, MobileSAM, EdgeSAM) to single-channel Quantitative Phase Imaging (QPI) data using Low-Rank Adaptation (LoRA).

A core contribution of this repository is the **Physics-Aware Phase Consistency Loss**, which forces the neural network to preserve biologically meaningful optical volumes and phase gradients during segmentation.

## ✨ Features

* **Lightweight Architectures:** Implements `MobileNet-UNet`, `MobileSAM` (TinyViT), and `EdgeSAM` adapted for 1-channel phase maps.
* **Parameter-Efficient Fine-Tuning (PEFT):** Custom LoRA injection modules supporting `encoder_only`, `attention_blocks`, and `bottleneck` strategies. Includes zero-overhead weight merging for edge deployment.
* **Physics-Aware Loss:** A composite loss function incorporating Phase-Mask Contrast (PMC), Boundary-Gradient Alignment (BGA), and Phase-Volume Preservation (PV).
* **Automated Rank Sweeping:** Built-in benchmarking pipeline to evaluate the latency-accuracy trade-offs across different LoRA ranks ($r = 2, 4, 8, 16, 32$).
* **Biological Morphological Metrics:** Computes standard AI metrics (Dice, IoU) alongside advanced biological metrics (Aggregated Jaccard Index, Boundary F1, and Phase Volume Error).

---

## 🛠️ Installation & Requirements

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install torch torchvision numpy scipy tifffile pillow PyYAML tqdm

```

For MobileSAM and EdgeSAM support, install their respective official packages:

```bash
pip install mobile-sam edge-sam

```

---

## 📁 Dataset Preparation

The framework expects single-channel `.tif` / `.npy` files. Phase images should be `float32` (radians), and masks should be `int` (values 0-4 for multi-class morphology).

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

---

## 🚀 Usage

The framework is controlled via a central `main.py` entry point, which accepts specific modes: `train`, `evaluate`, or `sweep`. Configurations are managed via YAML files located in the `configs/` directory.

### 1. Standard Training

To train a model using the settings defined in its configuration file:

```bash
python main.py --mode train --config configs/mobile_sam_lora.yaml --gpu 0

```

### 2. Evaluation

To evaluate a trained checkpoint (loads `best_model.pt` from the configured results directory):

```bash
python main.py --mode evaluate --config configs/edge_sam_lora.yaml --gpu 0

```

### 3. LoRA Rank Sweep (Hyperparameter Search)

To automatically train and evaluate multiple LoRA ranks sequentially (ideal for latency/accuracy trade-off analysis):

```bash
python main.py --mode sweep --config configs/mobilenet_unet_lora.yaml --ranks 2 4 8 16 32 --gpu 0

```

*Note: The script dynamically handles GPU memory clearing between ranks.*

---

## ⚙️ Configuration Files

Hyperparameters and loss weights are strictly controlled via YAML files. Example structure (`configs/mobile_sam_lora.yaml`):

```yaml
model:
  architecture: "mobile_sam"
  in_channels: 1
  num_classes: 5

lora:
  lora_r: 8
  lora_alpha: 8.0
  insertion_strategy: "attention_blocks"

training:
  epochs: 50
  learning_rate: 0.0005
  mixed_precision: "fp16"

loss_weights:
  lambda1_pmc: 0.1
  lambda2_bga: 0.05
  lambda3_pv: 0.1

```

---

## 🔬 Core Paper Message

> *"Physics-aware LoRA adaptation enables lightweight and edge-deployable quantitative holographic cell analysis while preserving biologically meaningful phase information."*
