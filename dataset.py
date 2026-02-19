"""
dataset.py
────────────────────────────────────────────────────────────
PyTorch Dataset for NIfTI brain MRI stroke classification.
Handles loading, preprocessing, and augmentation.
"""

import os
import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy.ndimage import rotate, zoom


# ─── Transforms ───────────────────────────────────────────────────────────────

class RandomFlip:
    """Random flipping along each axis."""
    def __call__(self, volume):
        for axis in range(3):
            if random.random() > 0.5:
                volume = np.flip(volume, axis=axis).copy()
        return volume


class RandomRotate:
    """Small random rotation in the axial plane (±15°)."""
    def __call__(self, volume):
        angle = random.uniform(-15, 15)
        return rotate(volume, angle, axes=(0, 1), reshape=False, order=1)


class RandomIntensityShift:
    """Additive brightness + multiplicative contrast jitter."""
    def __call__(self, volume):
        shift   = random.uniform(-0.1, 0.1)
        scale   = random.uniform(0.9,  1.1)
        volume  = volume * scale + shift
        return np.clip(volume, 0, 1)


class RandomZoom:
    """Zoom in/out by ±10 %."""
    def __call__(self, volume):
        factor = random.uniform(0.9, 1.1)
        zoomed = zoom(volume, factor, order=1)
        # Crop or pad back to original shape
        orig = np.array(volume.shape)
        new  = np.array(zoomed.shape)
        out  = np.zeros_like(volume)

        # Slices for source (zoomed)
        src_slices = []
        dst_slices = []
        for o, n in zip(orig, new):
            if n >= o:
                start = (n - o) // 2
                src_slices.append(slice(start, start + o))
                dst_slices.append(slice(0, o))
            else:
                start = (o - n) // 2
                src_slices.append(slice(0, n))
                dst_slices.append(slice(start, start + n))

        out[tuple(dst_slices)] = zoomed[tuple(src_slices)]
        return out


class AddGaussianNoise:
    def __init__(self, std=0.02):
        self.std = std
    def __call__(self, volume):
        return np.clip(volume + np.random.normal(0, self.std, volume.shape), 0, 1)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, volume):
        for t in self.transforms:
            volume = t(volume)
        return volume


# ─── Dataset ──────────────────────────────────────────────────────────────────

class BrainStrokeDataset(Dataset):
    """
    Folder structure expected:
        root/
            train/stroke/scan_*.nii.gz
            train/healthy/scan_*.nii.gz
            val/...
            test/...

    Args:
        root      : path to dataset root
        split     : "train" | "val" | "test"
        target_shape : (D, H, W) to resize all scans to
        augment   : apply augmentation (train only)
    """

    CLASSES = {"healthy": 0, "stroke": 1}

    def __init__(self, root, split="train", target_shape=(64, 64, 32), augment=False):
        self.root         = Path(root)
        self.split        = split
        self.target_shape = target_shape
        self.augment      = augment

        self.samples = []  # list of (path, label)
        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for class_name, label in self.CLASSES.items():
            class_dir = split_dir / class_name
            if class_dir.exists():
                for f in sorted(class_dir.glob("*.nii*")):
                    self.samples.append((f, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No NIfTI files found in {split_dir}")

        self.augment_transforms = Compose([
            RandomFlip(),
            RandomRotate(),
            RandomZoom(),
            RandomIntensityShift(),
            AddGaussianNoise(std=0.02),
        ])

        print(f"[Dataset] {split:5s} | {len(self.samples):4d} scans "
              f"| stroke: {sum(1 for _,l in self.samples if l==1)} "
              f"| healthy: {sum(1 for _,l in self.samples if l==0)}")

    def __len__(self):
        return len(self.samples)

    def _load_nii(self, path):
        img  = nib.load(str(path))
        data = img.get_fdata(dtype=np.float32)
        return data

    def _resize(self, volume):
        factors = [t / s for t, s in zip(self.target_shape, volume.shape)]
        resized = zoom(volume, factors, order=1)
        return resized.astype(np.float32)

    def _normalize(self, volume):
        mn, mx = volume.min(), volume.max()
        if mx - mn < 1e-8:
            return volume
        return (volume - mn) / (mx - mn)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        volume = self._load_nii(path)

        # Resize to uniform shape
        if volume.shape != tuple(self.target_shape):
            volume = self._resize(volume)

        volume = self._normalize(volume)

        if self.augment:
            volume = self.augment_transforms(volume)

        # Add channel dim → (1, D, H, W)
        volume = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)
        label  = torch.tensor(label, dtype=torch.long)
        return volume, label, str(path)


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_dataloaders(data_dir, target_shape=(64, 64, 32),
                    batch_size=8, num_workers=2):
    train_ds = BrainStrokeDataset(data_dir, "train", target_shape, augment=True)
    val_ds   = BrainStrokeDataset(data_dir, "val",   target_shape, augment=False)
    test_ds  = BrainStrokeDataset(data_dir, "test",  target_shape, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
