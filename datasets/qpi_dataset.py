"""
QPI Dataset: Loader for single-channel quantitative phase images.

Expected directory structure:
    data_root/
        train/
            phase/          ← .npy or .tiff float32 phase maps (radians)
            masks/          ← .npy uint8 instance masks
            labels.csv      ← optional morphology class labels
        val/
            phase/
            masks/
            labels.csv
        test/
            phase/
            masks/
            labels.csv

Phase maps: float32, shape (H, W), values in radians.
Masks: uint8, shape (H, W), 0=background, 1=cell instance (binary)
       OR instance-level integer labels for instance segmentation.

Morphology classes (for RBC):
    0: discocyte
    1: echinocyte
    2: stomatocyte
    3: spherocyte
"""

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datasets.qpi_augmentation import QPIAugmentation, QPIValTransform

# Optional TIFF support
try:
    import tifffile
    TIFF_AVAILABLE = True
except ImportError:
    TIFF_AVAILABLE = False

MORPHOLOGY_CLASSES = {
    "discocyte":  0,
    "echinocyte": 1,
    "stomatocyte": 2,
    "spherocyte": 3,
}

CLASS_WEIGHTS = [1.0, 1.5, 1.5, 2.0]  # Higher weight for rarer classes


class QPIDataset(Dataset):
    """
    Dataset for quantitative phase image segmentation.

    Returns:
        phase:  (1, H, W) float32 tensor  – normalized phase map
        mask:   (H, W) long tensor         – binary or instance mask
        meta:   dict with filename, morphology_class, storage_day
    """

    def __init__(self,
                 data_root: str,
                 split: str = "train",
                 image_size: Optional[int] = None,
                 augment: bool = True,
                 return_phase_raw: bool = True):
        """
        Args:
            data_root:        Root directory containing train/val/test splits.
            split:            One of "train", "val", "test".
            image_size:       Resize phase maps to (image_size, image_size) if set.
            augment:          Apply physics-preserving augmentation (train only).
            return_phase_raw: If True, also return unnormalized phase for loss computation.
        """
        self.data_root       = Path(data_root)
        self.split           = split
        self.image_size      = image_size
        self.return_phase_raw = return_phase_raw

        split_dir = self.data_root / split
        self.phase_dir = split_dir / "phase"
        self.mask_dir  = split_dir / "masks"

        if not self.phase_dir.exists():
            raise FileNotFoundError(f"Phase directory not found: {self.phase_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        # Collect sample paths
        self.samples = self._collect_samples()

        # Load morphology labels if available
        label_file = split_dir / "labels.csv"
        self.labels = self._load_labels(label_file) if label_file.exists() else {}

        # Transforms
        if augment and split == "train":
            self.transform = QPIAugmentation(
                flip_h=True, flip_v=True, rotate=True,
                translate=True, zoom=False, normalize=True
            )
        else:
            self.transform = QPIValTransform()

        print(f"[QPIDataset] {split}: {len(self.samples)} samples loaded "
              f"from {split_dir}")

    def _collect_samples(self) -> List[Dict]:
        """Collect matched phase/mask file pairs."""
        phase_files = sorted(
            list(self.phase_dir.glob("*.npy")) +
            list(self.phase_dir.glob("*.tiff")) +
            list(self.phase_dir.glob("*.tif"))
        )

        samples = []
        missing_masks = 0

        for pf in phase_files:
            stem = pf.stem
            # Look for matching mask
            mask_path = None
            for ext in [".npy", ".tiff", ".tif", ".png"]:
                candidate = self.mask_dir / (stem + ext)
                if candidate.exists():
                    mask_path = candidate
                    break

            if mask_path is None:
                missing_masks += 1
                continue

            # Extract storage day from filename if present (e.g., day_14_cell_001)
            storage_day = None
            parts = stem.split("_")
            for i, p in enumerate(parts):
                if p == "day" and i + 1 < len(parts):
                    try:
                        storage_day = int(parts[i + 1])
                    except ValueError:
                        pass

            samples.append({
                "phase_path":   pf,
                "mask_path":    mask_path,
                "stem":         stem,
                "storage_day":  storage_day,
            })

        if missing_masks > 0:
            print(f"[QPIDataset] Warning: {missing_masks} phase files had no matching mask.")

        if len(samples) == 0:
            raise RuntimeError(
                f"No matched phase/mask pairs found in {self.phase_dir}. "
                f"Check file naming convention."
            )

        return samples

    def _load_labels(self, label_file: Path) -> Dict[str, int]:
        """Load morphology class labels from CSV."""
        labels = {}
        with open(label_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stem  = Path(row["filename"]).stem
                cls   = row.get("morphology_class", "discocyte").strip().lower()
                labels[stem] = MORPHOLOGY_CLASSES.get(cls, 0)
        print(f"[QPIDataset] Loaded {len(labels)} morphology labels.")
        return labels

    def _load_phase(self, path: Path) -> np.ndarray:
        """Load phase map as float32 array."""
        if path.suffix == ".npy":
            phase = np.load(path).astype(np.float32)
        elif path.suffix in [".tiff", ".tif"]:
            if not TIFF_AVAILABLE:
                raise ImportError("tifffile required for .tiff files: pip install tifffile")
            phase = tifffile.imread(str(path)).astype(np.float32)
        else:
            raise ValueError(f"Unsupported phase file format: {path.suffix}")

        # Ensure 2D
        if phase.ndim == 3:
            phase = phase[0]  # Take first channel if multi-channel

        return phase

    def _load_mask(self, path: Path) -> np.ndarray:
        """Load segmentation mask as uint8 array."""
        if path.suffix == ".npy":
            mask = np.load(path)
        elif path.suffix in [".tiff", ".tif"]:
            if not TIFF_AVAILABLE:
                raise ImportError("tifffile required: pip install tifffile")
            mask = tifffile.imread(str(path))
        elif path.suffix == ".png":
            try:
                from PIL import Image
                mask = np.array(Image.open(path))
            except ImportError:
                raise ImportError("Pillow required for .png masks: pip install Pillow")
        else:
            raise ValueError(f"Unsupported mask format: {path.suffix}")

        # Binarize: any non-zero value = cell
        return (mask > 0).astype(np.int64)

    def _resize(self, phase: np.ndarray, mask: np.ndarray):
        """Resize phase and mask to target image_size."""
        import torch.nn.functional as F
        ph = torch.from_numpy(phase).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        mk = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()

        ph = F.interpolate(ph, size=(self.image_size, self.image_size),
                           mode="bilinear", align_corners=False).squeeze()
        mk = F.interpolate(mk, size=(self.image_size, self.image_size),
                           mode="nearest").squeeze().long()
        return ph.numpy(), mk.numpy()

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

        # Store raw phase BEFORE augmentation for physics-aware loss
        phase_raw = torch.from_numpy(phase.copy()).unsqueeze(0)  # (1, H, W)

        # Convert to tensors
        phase_t = torch.from_numpy(phase).unsqueeze(0)  # (1, H, W)
        mask_t  = torch.from_numpy(mask).long()          # (H, W)

        # Apply transforms (returns normalized phase + mask)
        phase_t, mask_t = self.transform(phase_t, mask_t)

        morphology_class = self.labels.get(sample["stem"], 0)
        storage_day      = sample["storage_day"] if sample["storage_day"] is not None else -1

        result = {
            "phase":            phase_t,         # (1, H, W) normalized
            "mask":             mask_t,           # (H, W) long
            "morphology_class": morphology_class, # int
            "storage_day":      storage_day,      # int (-1 if unknown)
            "stem":             sample["stem"],
        }

        if self.return_phase_raw:
            result["phase_raw"] = phase_raw      # (1, H, W) unnormalized

        return result


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------

def get_qpi_loaders(config, num_workers: int = 4):
    """
    Build train/val/test DataLoaders for QPI segmentation.

    Returns:
        train_loader, val_loader, test_loader (test_loader may be None)
    """
    data_root  = str(getattr(config, "data_root", config.train_image_dir))
    image_size = getattr(config, "image_size", None)
    batch_size = getattr(config, "batch_size", 8)
    debug_mode = getattr(config, "debug_mode", False)

    train_ds = QPIDataset(data_root, split="train", image_size=image_size,
                          augment=True)
    val_ds   = QPIDataset(data_root, split="val",   image_size=image_size,
                          augment=False)

    if debug_mode:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(min(50, len(train_ds))))
        val_ds   = Subset(val_ds,   range(min(20, len(val_ds))))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers if not debug_mode else 0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if not debug_mode else 0,
        pin_memory=True,
    )

    # Optional test set
    test_loader = None
    test_dir = Path(data_root) / "test"
    if test_dir.exists():
        test_ds = QPIDataset(data_root, split="test", image_size=image_size,
                             augment=False)
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers if not debug_mode else 0,
            pin_memory=True,
        )

    print(f"[QPIDataLoader] Train: {len(train_loader)} batches | "
          f"Val: {len(val_loader)} batches | "
          f"Test: {len(test_loader) if test_loader else 0} batches")

    return train_loader, val_loader, test_loader