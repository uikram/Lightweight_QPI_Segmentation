"""
QPI Dataset: Loader for single-channel quantitative phase images.

Actual directory structure (X_train / Y_train layout):
    data_root/
        X_train/    ← .tif float32 phase maps (single-channel, radians)
        Y_train/    ← .tif uint8/int semantic masks, pixel values 0-4
        X_val/      ← used as validation during training (also serves as test set)
        Y_val/

Mask pixel values:
    0: Background
    1: Discocyte
    2: Echinocyte
    3: Spherocyte
    4: Stomatocyte

Phase maps: float32, shape (H, W), values in radians.
"""

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datasets.qpi_augmentation import QPIAugmentation, QPIValTransform
from datetime import datetime

try:
    import tifffile
    TIFF_AVAILABLE = True
except ImportError:
    TIFF_AVAILABLE = False

# ── Label mapping ─────────────────────────────────────────────────────────────
# Indices match dataset pixel values exactly (no offset).
MORPHOLOGY_CLASSES = {
    "background":  0,
    "discocyte":   1,
    "echinocyte":  2,
    "spherocyte":  3,
    "stomatocyte": 4,
}

# CrossEntropyLoss class weights: [bg, disco, echino, sphero, stomato]
# Higher weight for rarer / clinically important degradation classes.
CLASS_WEIGHTS = [0.5, 1.0, 1.5, 2.0, 2.0]

NUM_CLASSES = 5


class QPIDataset(Dataset):
    """
    Dataset for quantitative phase image segmentation.

    Returns:
        phase:   (1, H, W) float32 tensor  – normalized phase map
        mask:    (H, W)    long tensor      – class labels 0-4
        meta:    dict with stem, morphology_class, storage_day
    """

    def __init__(self,
                 data_root: str,
                 split: str = "train",
                 image_size: Optional[int] = None,
                 augment: bool = True,
                 return_phase_raw: bool = True):
        """
        Args:
            data_root:        Root directory containing X_train, Y_train, X_val, Y_val.
            split:            'train' or 'val'.
            image_size:       Resize to (image_size, image_size) if set.
            augment:          Apply physics-preserving augmentation (train only).
            return_phase_raw: If True, also return unnormalized phase for loss computation.
        """
        self.data_root        = Path(data_root)
        self.split            = split
        self.image_size       = image_size
        self.return_phase_raw = return_phase_raw

        # ── Directory layout: X_{split} / Y_{split} ──────────────────────────
        self.phase_dir = self.data_root / f"X_{split}"
        self.mask_dir  = self.data_root / f"Y_{split}"

        if not self.phase_dir.exists():
            raise FileNotFoundError(
                f"Phase directory not found: {self.phase_dir}\n"
                f"Expected layout: {{data_root}}/X_{{split}} and Y_{{split}}"
            )
        if not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Mask directory not found: {self.mask_dir}"
            )

        self.samples = self._collect_samples()

        # labels.csv is optional — falls back to empty dict gracefully
        label_file    = self.data_root / f"labels_{split}.csv"
        self.labels   = self._load_labels(label_file) if label_file.exists() else {}
        if not self.labels:
            print(f"[QPIDataset] No labels CSV found. morphology_class defaults to 0.")

        # ── Transforms ────────────────────────────────────────────────────────
        if augment and split == "train":
            self.transform = QPIAugmentation(
                flip_h=True, flip_v=True, rotate=True,
                translate=True, zoom=False, normalize=True
            )
        else:
            self.transform = QPIValTransform()

        print(f"[QPIDataset] {split}: {len(self.samples)} samples "
              f"from {self.phase_dir.name} / {self.mask_dir.name}")

    # ── Sample collection ─────────────────────────────────────────────────────
    def _collect_samples(self) -> List[Dict]:
        # Search for both lowercase and uppercase phase map extensions
        phase_files = sorted(
            list(self.phase_dir.glob("*.tif")) +
            list(self.phase_dir.glob("*.tiff")) +
            list(self.phase_dir.glob("*.TIF")) +
            list(self.phase_dir.glob("*.TIFF")) +
            list(self.phase_dir.glob("*.npy"))
        )

        samples = []
        missing = 0

        for pf in phase_files:
            stem = pf.stem
            mask_path = None
            
            # Look for exact match, or common mask prefixes/suffixes
            possible_stems = [
                stem,
                f"mask_{stem}",
                f"{stem}_mask",
                f"output_mask_{stem}",
                stem.replace("image", "mask"),
                stem.replace("image", "output_mask")
            ]
            
            for p_stem in possible_stems:
                # ADDED: Uppercase .TIFF, .TIF, .PNG to handle Linux case-sensitivity
                for ext in [".tif", ".tiff", ".npy", ".png", ".TIFF", ".TIF", ".PNG"]:
                    candidate = self.mask_dir / (p_stem + ext)
                    if candidate.exists():
                        mask_path = candidate
                        break
                if mask_path:
                    break

            if mask_path is None:
                missing += 1
                continue

            # Extract storage day from filename prefix
            raw_dates = []
            for pf in phase_files:
                stem = pf.stem
                mask_path = None
                for ext in [".tif", ".tiff", ".npy", ".png"]:
                    candidate = self.mask_dir / (stem + ext)
                    if candidate.exists():
                        mask_path = candidate
                        break

                if mask_path is None:
                    missing += 1
                    continue
                    
                try:
                    raw_dates.append(datetime.strptime(stem.split("_")[0], "%Y%m%d"))
                except ValueError:
                    raw_dates.append(None)
                    
                samples.append({
                    "phase_path": pf,
                    "mask_path": mask_path,
                    "stem": stem,
                    "storage_day": None # Will fill in next step
                })

            # Compute relative days
            valid_dates = [d for d in raw_dates if d is not None]
            base_date = min(valid_dates) if valid_dates else None

            for s, d in zip(samples, raw_dates):
                if d and base_date:
                    s["storage_day"] = (d - base_date).days

        if missing:
            print(f"[QPIDataset] Warning: {missing} phase files had no matching mask.")
        if not samples:
            raise RuntimeError(
                f"No matched phase/mask pairs found.\n"
                f"  phase_dir: {self.phase_dir}\n"
                f"  mask_dir:  {self.mask_dir}\n"
                f"  Make sure your image and mask filenames share the same base name!"
            )
        return samples

    # ── Label CSV ─────────────────────────────────────────────────────────────
    def _load_labels(self, label_file: Path) -> Dict[str, int]:
        labels = {}
        with open(label_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stem = Path(row["filename"]).stem
                cls  = row.get("morphology_class", "background").strip().lower()
                labels[stem] = MORPHOLOGY_CLASSES.get(cls, 0)
        print(f"[QPIDataset] Loaded {len(labels)} morphology labels.")
        return labels

    # ── Loaders ───────────────────────────────────────────────────────────────
    def _load_phase(self, path: Path) -> np.ndarray:
        if path.suffix in [".tif", ".tiff"]:
            if not TIFF_AVAILABLE:
                raise ImportError("Install tifffile: pip install tifffile")
            phase = tifffile.imread(str(path)).astype(np.float32)
        elif path.suffix == ".npy":
            phase = np.load(path).astype(np.float32)
        else:
            raise ValueError(f"Unsupported phase format: {path.suffix}")

        # Ensure 2-D (H, W)
        if phase.ndim == 3:
            phase = phase[0]
        return phase

    def _load_mask(self, path: Path) -> np.ndarray:
        """
        Load semantic mask and return as (H, W) int64 array with values 0-4.

        Handles two TIFF encodings:
          • Single-channel (H, W): values 0-4 — ground-truth format, used as-is.
          • Multi-channel (H, W, C): treated as single-channel by taking channel 0.
            This covers the case where GT files were accidentally saved as RGB
            but all channels carry the same class-index plane.
            A warning is printed if channels differ so you can diagnose the files.
        """
        if path.suffix in [".tif", ".tiff"]:
            if not TIFF_AVAILABLE:
                raise ImportError("Install tifffile: pip install tifffile")
            mask = tifffile.imread(str(path))
        elif path.suffix == ".npy":
            mask = np.load(path)
        elif path.suffix == ".png":
            try:
                from PIL import Image
                mask = np.array(Image.open(path))
            except ImportError:
                raise ImportError("Install Pillow: pip install Pillow")
        else:
            raise ValueError(f"Unsupported mask format: {path.suffix}")

        # ── Handle multi-channel TIFFs ────────────────────────────────────────
        if mask.ndim == 3:
            if not np.array_equal(mask[:, :, 0], mask[:, :, 1]):
                print(
                    f"[QPIDataset] WARNING: mask {path.name} has 3 channels "
                    f"with differing values. Taking channel 0. "
                    f"Verify your Y_train TIFF format is single-channel int (0-4)."
                )
            mask = mask[:, :, 0]

        return mask.astype(np.int64)

    # ── Resize ────────────────────────────────────────────────────────────────
    def _resize(self, phase: np.ndarray, mask: np.ndarray):
        ph = torch.from_numpy(phase).unsqueeze(0).unsqueeze(0)       # (1,1,H,W)
        mk = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()

        ph = torch.nn.functional.interpolate(
            ph, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False
        ).squeeze()
        mk = torch.nn.functional.interpolate(
            mk, size=(self.image_size, self.image_size),
            mode="nearest"
        ).squeeze().long()
        return ph.numpy(), mk.numpy()

    # ── Dataset interface ─────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        try:
            phase = self._load_phase(sample["phase_path"])
            mask  = self._load_mask(sample["mask_path"])
        except Exception as e:
            print(f"[QPIDataset] Error loading {sample['stem']}: {e}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))

        if self.image_size is not None:
            phase, mask = self._resize(phase, mask)

        # Raw phase before augmentation — used by physics-aware loss
        phase_raw = torch.from_numpy(phase.copy()).unsqueeze(0)   # (1, H, W)

        phase_t = torch.from_numpy(phase).unsqueeze(0)            # (1, H, W)
        mask_t  = torch.from_numpy(mask).long()                   # (H, W)

        phase_t, mask_t = self.transform(phase_t, mask_t)

        morphology_class = self.labels.get(sample["stem"], 0)
        storage_day      = sample["storage_day"] if sample["storage_day"] is not None else -1

        result = {
            "phase":            phase_t,
            "mask":             mask_t,
            "morphology_class": morphology_class,
            "storage_day":      storage_day,
            "stem":             sample["stem"],
        }
        if self.return_phase_raw:
            result["phase_raw"] = phase_raw

        return result


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_qpi_loaders(config, num_workers: int = 4):
    """
    Build train / val DataLoaders from X_train/Y_train and X_val/Y_val.
    The val loader doubles as the test loader (no separate test split).

    Returns: train_loader, val_loader, None
    """
    data_root  = str(getattr(config, "data_root", "./dataset"))
    image_size = getattr(config, "image_size",  None)
    batch_size = getattr(config, "batch_size",  8)
    debug_mode = getattr(config, "debug_mode",  False)

    train_ds = QPIDataset(data_root, split="train", image_size=image_size, augment=True)
    val_ds   = QPIDataset(data_root, split="val",   image_size=image_size, augment=False)

    if debug_mode:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(min(32, len(train_ds))))
        val_ds   = Subset(val_ds,   range(min(16, len(val_ds))))

    nw = num_workers if not debug_mode else 0

    # Access the underlying dataset if it's wrapped in a Subset (debug_mode)
    ds_for_weights = train_ds.dataset if hasattr(train_ds, 'dataset') else train_ds

    # Only use the sampler if we actually have labels to balance
    if not debug_mode and len(ds_for_weights) > 0 and len(ds_for_weights.labels) > 0:
        class_counts = torch.zeros(5) # 5 classes
        for s in ds_for_weights.samples:
            cls = ds_for_weights.labels.get(s["stem"], 0)
            class_counts[cls] += 1
            
        class_counts = class_counts.clamp(min=1)
        class_weights = 1.0 / class_counts
        
        sample_weights = torch.tensor(
            [class_weights[ds_for_weights.labels.get(s["stem"], 0)] for s in ds_for_weights.samples]
        )
        # len(train_ds) ensures we sample the correct amount if debug_mode modified the length
        sampler = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=nw, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True,
    )

    print(f"[QPIDataLoader] Train: {len(train_loader)} batches | "
          f"Val: {len(val_loader)} batches")

    # Val set serves as test set per dataset specification.
    # Returning None for test_loader so existing main.py logic is unaffected.
    return train_loader, val_loader, None
