"""
preprocess.py
────────────────────────────────────────────────────────────
Robust MRI preprocessing pipeline used at BOTH training time
and inference time — ensuring the model always sees the same
input distribution.

Functions
─────────
  preprocess_scan(path, target_shape)  ← use this everywhere
  validate_scan(path)                  ← call before inference
  inspect_scan(path)                   ← print raw stats

The problem it solves
─────────────────────
  Raw MRI voxel values can range from 0–4000+ (T1), 0–32000 (T2/FLAIR).
  The model was trained on [0, 1] normalised scans.
  Feeding raw unnormalised scans → garbage probability outputs (e.g. 90%)
  even for completely healthy brains.

Pipeline steps
──────────────
  1.  Load NIfTI
  2.  Squeeze extra dims  (e.g. 4D → 3D)
  3.  Brain masking       (non-zero voxels only)
  4.  Percentile clipping (p1 → p99.5) to remove outlier hot/dark spots
  5.  Min-Max normalisation → [0, 1]
  6.  Spatial resize       → target_shape via trilinear zoom
  7.  Return float32 tensor (1, 1, D, H, W)
"""

import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import zoom
from pathlib import Path


# ─── Core Pipeline ────────────────────────────────────────────────────────────

def preprocess_scan(path, target_shape=(64, 64, 32), verbose=True):
    """
    Load, normalise, resize a NIfTI scan and return a model-ready tensor.

    Parameters
    ----------
    path         : str or Path to .nii / .nii.gz
    target_shape : (D, H, W) tuple — must match what the model was trained on
    verbose      : print normalisation stats

    Returns
    -------
    tensor : torch.FloatTensor of shape (1, 1, D, H, W) in [0, 1]
    meta   : dict with raw stats (useful for logging / debugging)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scan not found: {path}")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    img  = nib.load(str(path))
    vol  = np.asarray(img.dataobj, dtype=np.float32)

    # ── 2. Squeeze extra dims (e.g. (X,Y,Z,1) from some scanners) ────────────
    while vol.ndim > 3:
        vol = vol[..., 0]

    raw_min  = float(vol.min())
    raw_max  = float(vol.max())
    raw_mean = float(vol.mean())
    raw_std  = float(vol.std())

    if verbose:
        print(f"\n[Preprocess] {path.name}")
        print(f"  Raw shape   : {vol.shape}")
        print(f"  Raw dtype   : {img.header.get_data_dtype()}")
        print(f"  Raw range   : [{raw_min:.1f}, {raw_max:.1f}]")
        print(f"  Raw mean/std: {raw_mean:.1f} / {raw_std:.1f}")

    # ── 3. Brain mask — use non-zero / non-background voxels ─────────────────
    brain_mask = vol > (raw_max * 0.01)   # exclude background (air ≈ 0)
    if brain_mask.sum() < 1000:
        brain_mask = vol > raw_min

    # ── 4. Percentile clipping (removes scanner noise / hot spots) ───────────
    p_low  = float(np.percentile(vol[brain_mask], 0.5))
    p_high = float(np.percentile(vol[brain_mask], 99.5))
    vol    = np.clip(vol, p_low, p_high)

    if verbose:
        print(f"  Clipped to  : [{p_low:.1f}, {p_high:.1f}]  (p0.5–p99.5 of brain)")

    # ── 5. Min-Max normalisation → [0, 1] ────────────────────────────────────
    v_min = vol[brain_mask].min()
    v_max = vol[brain_mask].max()
    if v_max - v_min < 1e-8:
        raise ValueError(f"Scan has no intensity variation (min=max={v_min}). "
                         f"Check if the file is corrupt.")
    vol = (vol - v_min) / (v_max - v_min)
    vol[~brain_mask] = 0.0   # keep background at 0 after normalisation
    vol = np.clip(vol, 0.0, 1.0)

    if verbose:
        print(f"  Normalised  : [{vol.min():.4f}, {vol.max():.4f}]  ✓")

    # ── 6. Spatial resize → target_shape ─────────────────────────────────────
    if vol.shape != tuple(target_shape):
        factors = [t / s for t, s in zip(target_shape, vol.shape)]
        vol     = zoom(vol, factors, order=1)   # bilinear
        if verbose:
            print(f"  Resized     : {vol.shape}  (target {target_shape})")

    vol = vol.astype(np.float32)

    # ── 7. To tensor (1, 1, D, H, W) ─────────────────────────────────────────
    tensor = torch.tensor(vol).unsqueeze(0).unsqueeze(0)

    meta = {
        "path":     str(path),
        "raw_shape": img.shape,
        "raw_range": (raw_min, raw_max),
        "raw_mean":  raw_mean,
        "raw_std":   raw_std,
        "clip_range": (p_low, p_high),
        "normalised": True,
        "target_shape": target_shape,
    }

    if verbose:
        print(f"  Tensor shape: {tuple(tensor.shape)}  dtype={tensor.dtype}\n")

    return tensor, meta


# ─── Scan Validator ───────────────────────────────────────────────────────────

def validate_scan(path, target_shape=(64, 64, 32)):
    """
    Run checks on a NIfTI file before inference.
    Raises ValueError with a clear message if something is wrong.
    Returns (tensor, meta) if all checks pass.
    """
    path = Path(path)

    # Check file exists and is NIfTI
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix not in (".nii", ".gz"):
        raise ValueError(f"Expected .nii or .nii.gz file, got: {path.suffix}")

    # Try loading
    try:
        img = nib.load(str(path))
    except Exception as e:
        raise ValueError(f"Could not read NIfTI file: {e}")

    vol = np.asarray(img.dataobj, dtype=np.float32)
    while vol.ndim > 3:
        vol = vol[..., 0]

    # Check for empty / all-zero volume
    if vol.max() == 0:
        raise ValueError("Scan appears to be all zeros — file may be corrupt.")

    # Check for NaN or Inf
    if not np.isfinite(vol).all():
        n_bad = (~np.isfinite(vol)).sum()
        raise ValueError(f"Scan contains {n_bad} NaN/Inf values. "
                         "Consider re-exporting from DICOM.")

    # Warn if unnormalised (raw MRI values, NOT already [0,1])
    raw_max = vol.max()
    if raw_max > 10.0:
        print(f"  ⚠  Raw max = {raw_max:.1f}  →  Will normalise automatically.")
    else:
        print(f"  ✓  Looks pre-normalised (max={raw_max:.4f})")

    # All good — preprocess
    tensor, meta = preprocess_scan(path, target_shape, verbose=True)
    return tensor, meta


# ─── Inspect utility ──────────────────────────────────────────────────────────

def inspect_scan(path):
    """Print full statistics about a raw NIfTI scan (no modification)."""
    path = Path(path)
    img  = nib.load(str(path))
    vol  = np.asarray(img.dataobj, dtype=np.float32)

    print(f"\n{'='*55}")
    print(f"  SCAN INSPECTION: {path.name}")
    print(f"{'='*55}")
    print(f"  Shape        : {img.shape}")
    print(f"  Data dtype   : {img.header.get_data_dtype()}")
    print(f"  Voxel size   : {img.header.get_zooms()}")
    print(f"  Min value    : {vol.min():.2f}")
    print(f"  Max value    : {vol.max():.2f}")
    print(f"  Mean         : {vol.mean():.2f}")
    print(f"  Std          : {vol.std():.2f}")
    print(f"  p1 / p99     : {np.percentile(vol, 1):.2f} / {np.percentile(vol, 99):.2f}")
    print(f"  Non-zero vox : {(vol>0).sum():,}  of  {vol.size:,}")
    if vol.max() > 10:
        print(f"\n  ⚠  NOT NORMALISED — raw scanner values detected.")
        print(f"     Use preprocess_scan() before feeding to the model.")
    else:
        print(f"\n  ✓  Appears normalised (max ≤ 10).")
    print(f"{'='*55}\n")
