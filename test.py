"""
test.py
────────────────────────────────────────────────────────────

Modes:
  1. test   — evaluate the trained model on the test set
  2. predict — run inference on a single .nii.gz file
  3. compare — compare two .nii.gz scans (e.g. baseline vs follow-up)

Run:
    # Evaluate on test set
    python test.py --mode test --checkpoint checkpoints/best_model.pth --data_dir data/

    # Single scan inference
    python test.py --mode predict --checkpoint checkpoints/best_model.pth --scan patient.nii.gz

    # Longitudinal comparison
    python test.py --mode compare \
        --checkpoint checkpoints/best_model.pth \
        --scan followup.nii.gz \
        --baseline baseline.nii.gz
"""

import os
import argparse
import json
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.ndimage import zoom
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
)

from model      import StrokeCNN3D, GradCAM3D
from preprocess import validate_scan, inspect_scan
from dataset import get_dataloaders, BrainStrokeDataset


# ─── Colours ──────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":      "#0D0D1A",
    "panel":   "#141428",
    "accent1": "#FF3B30",  # high risk
    "accent2": "#FF9F0A",  # moderate
    "accent3": "#30D158",  # low risk
    "text":    "#E8E8FF",
    "muted":   "#6E6E8F",
}


# ─── Loading helpers ──────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path, device):
    # PyTorch 2.6+ changed default weights_only=True; use False for full dicts
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = StrokeCNN3D(
        base_filters=args.get("base_filters", 16),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()
    print(f"✓ Loaded checkpoint: {checkpoint_path}")
    if "val_auc" in ckpt:
        print(f"  Saved at epoch {ckpt.get('epoch','?')}  |  val AUC={ckpt['val_auc']:.4f}")
    return model, args


def preprocess_scan(path, target_shape=(64, 64, 32)):
    """Validates + applies full normalisation pipeline before inference."""
    tensor, meta = validate_scan(path, target_shape)
    return tensor


def stroke_probability(model, volume_tensor, device):
    volume_tensor = volume_tensor.to(device)
    with torch.no_grad():
        logits = model(volume_tensor)
    return F.softmax(logits, dim=1)[0, 1].item()


def risk_label(prob):
    if prob >= 0.75: return "HIGH RISK",   PALETTE["accent1"]
    if prob <= 0.45 and prob >= 20: return "MODERATE",    PALETTE["accent2"]
    return "LOW RISK", PALETTE["accent3"]


# ─── Mode 1: Test-set evaluation ──────────────────────────────────────────────

def evaluate_test_set(model, data_dir, args, device, out_dir):
    target_shape = tuple(args.get("input_shape", [64, 64, 32]))
    test_ds = BrainStrokeDataset(data_dir, "test", target_shape, augment=False)
    from torch.utils.data import DataLoader
    loader  = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2)

    all_labels, all_probs, all_preds = [], [], []
    with torch.no_grad():
        for volumes, labels, _ in loader:
            volumes = volumes.to(device)
            logits  = model(volumes)
            probs   = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds   = (probs >= 0.5).astype(int)
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())

    labels = np.array(all_labels)
    probs  = np.array(all_probs)
    preds  = np.array(all_preds)

    auc = roc_auc_score(labels, probs)
    f1  = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    cm  = confusion_matrix(labels, preds, labels=[0, 1])

    print("\n" + "="*60)
    print("  TEST SET RESULTS")
    print("="*60)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print("\n" + classification_report(labels, preds, target_names=["Healthy", "Stroke"]))
    print("  Confusion Matrix:")
    print(f"             Pred Healthy  Pred Stroke")
    print(f"  True Healthy  {cm[0,0]:6d}       {cm[0,1]:6d}")
    print(f"  True Stroke   {cm[1,0]:6d}       {cm[1,1]:6d}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor=PALETTE["bg"])
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ROC
    ax1 = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(labels, probs)
    ax1.plot(fpr, tpr, color=PALETTE["accent3"], lw=2, label=f"AUC = {auc:.3f}")
    ax1.plot([0,1],[0,1],"--", color=PALETTE["muted"], lw=1)
    _style_ax(ax1, "ROC Curve", "FPR", "TPR")
    ax1.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["muted"],
               labelcolor=PALETTE["text"])

    # PR Curve
    ax2 = fig.add_subplot(gs[0, 1])
    precision, recall, _ = precision_recall_curve(labels, probs)
    ax2.plot(recall, precision, color=PALETTE["accent2"], lw=2)
    _style_ax(ax2, "Precision–Recall", "Recall", "Precision")

    # Confusion Matrix
    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(cm, cmap="YlOrRd", aspect="auto")
    ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Healthy", "Stroke"], color=PALETTE["text"])
    ax3.set_yticklabels(["Healthy", "Stroke"], color=PALETTE["text"])
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white", fontsize=14, fontweight="bold")
    ax3.set_title("Confusion Matrix", color=PALETTE["text"], fontsize=11, pad=8)
    ax3.set_xlabel("Predicted",  color=PALETTE["muted"])
    ax3.set_ylabel("True Label", color=PALETTE["muted"])
    ax3.tick_params(colors=PALETTE["muted"])
    for spine in ax3.spines.values():
        spine.set_edgecolor(PALETTE["muted"])

    # Score distribution
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(probs[labels == 0], bins=20, color=PALETTE["accent3"],
             alpha=0.7, label="Healthy", density=True)
    ax4.hist(probs[labels == 1], bins=20, color=PALETTE["accent1"],
             alpha=0.7, label="Stroke",  density=True)
    ax4.axvline(0.5, color="white", ls="--", lw=1)
    _style_ax(ax4, "Score Distribution", "Stroke Probability", "Density")
    ax4.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["muted"],
               labelcolor=PALETTE["text"])

    # Metrics bar
    ax5 = fig.add_subplot(gs[1, 1])
    names  = ["Accuracy", "AUC-ROC", "F1"]
    values = [acc, auc, f1]
    colors = [PALETTE["accent3"], PALETTE["accent2"], PALETTE["accent1"]]
    bars = ax5.barh(names, values, color=colors, height=0.5)
    for bar, v in zip(bars, values):
        ax5.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{v:.3f}", va="center", color=PALETTE["text"], fontsize=11)
    ax5.set_xlim(0, 1.15)
    _style_ax(ax5, "Test Metrics", "Score", "")

    # Summary text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    summary = (
        f"TEST SUMMARY\n\n"
        f"Total scans :  {len(labels)}\n"
        f"Healthy     :  {(labels==0).sum()}\n"
        f"Stroke      :  {(labels==1).sum()}\n\n"
        f"TP : {cm[1,1]}    TN : {cm[0,0]}\n"
        f"FP : {cm[0,1]}    FN : {cm[1,0]}\n\n"
        f"Sensitivity : {cm[1,1]/(cm[1,1]+cm[1,0]+1e-8):.3f}\n"
        f"Specificity : {cm[0,0]/(cm[0,0]+cm[0,1]+1e-8):.3f}"
    )
    ax6.text(0.1, 0.9, summary, transform=ax6.transAxes,
             color=PALETTE["text"], fontsize=11, va="top",
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor=PALETTE["panel"],
                       edgecolor=PALETTE["muted"], alpha=0.9))

    fig.suptitle("Brain Stroke Predictor — Test Evaluation",
                 color=PALETTE["text"], fontsize=15, fontweight="bold", y=1.01)
    out_path = out_dir / "test_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"\n✓ Saved evaluation plot → {out_path}")


# ─── Mode 2: Single scan prediction ──────────────────────────────────────────

def predict_single(model, scan_path, args, device, out_dir):
    target_shape = tuple(args.get("input_shape", [64, 64, 32]))
    volume = preprocess_scan(scan_path, target_shape)
    prob   = stroke_probability(model, volume, device)
    label, color = risk_label(prob)

    # Grad-CAM
    cam = GradCAM3D(model)(volume.to(device), class_idx=1)

    _plot_prediction(volume.squeeze().numpy(), cam, prob, label, color,
                     title=f"Scan: {Path(scan_path).name}",
                     out_path=out_dir / "prediction.png")

    print(f"\n{'='*50}")
    print(f"  PREDICTION RESULT")
    print(f"  File        : {scan_path}")
    print(f"  Probability : {prob:.4f}  ({prob*100:.1f}%)")
    print(f"  Risk Level  : {label}")
    print(f"{'='*50}\n")
    return prob


# ─── Mode 3: Compare two scans ────────────────────────────────────────────────

def compare_scans(model, scan_path, baseline_path, args, device, out_dir):
    target_shape = tuple(args.get("input_shape", [64, 64, 32]))

    vol_base = preprocess_scan(baseline_path, target_shape)
    vol_scan = preprocess_scan(scan_path,     target_shape)

    prob_base = stroke_probability(model, vol_base, device)
    prob_scan = stroke_probability(model, vol_scan, device)

    delta = prob_scan - prob_base
    direction = "▲ INCREASED" if delta > 0.05 else ("▼ DECREASED" if delta < -0.05 else "→ STABLE")

    lbl_base, col_base = risk_label(prob_base)
    lbl_scan, col_scan = risk_label(prob_scan)

    cam_base = GradCAM3D(model)(vol_base.to(device), class_idx=1)
    cam_scan = GradCAM3D(model)(vol_scan.to(device), class_idx=1)

    _plot_comparison(
        vol_base.squeeze().numpy(), cam_base, prob_base, lbl_base, col_base,
        vol_scan.squeeze().numpy(), cam_scan, prob_scan, lbl_scan, col_scan,
        delta, direction,
        out_path=out_dir / "comparison.png"
    )

    print(f"\n{'='*60}")
    print(f"  LONGITUDINAL COMPARISON")
    print(f"  Baseline  : {baseline_path}  →  {prob_base*100:.1f}%  [{lbl_base}]")
    print(f"  Follow-up : {scan_path}  →  {prob_scan*100:.1f}%  [{lbl_scan}]")
    print(f"  Delta     : {delta:+.4f}  {direction}")
    print(f"{'='*60}\n")


# ─── Plot helpers ─────────────────────────────────────────────────────────────

def _style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(PALETTE["panel"])
    ax.set_title(title,   color=PALETTE["text"],  fontsize=11, pad=6)
    ax.set_xlabel(xlabel, color=PALETTE["muted"], fontsize=9)
    ax.set_ylabel(ylabel, color=PALETTE["muted"], fontsize=9)
    ax.tick_params(colors=PALETTE["muted"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["muted"])


def _mid_slices(volume):
    """Return axial, coronal, sagittal mid-slices."""
    d, h, w = volume.shape
    return (volume[d//2, :, :],
            volume[:, h//2, :],
            volume[:, :, w//2])


def _plot_prediction(volume, cam, prob, label, color, title, out_path):
    fig = plt.figure(figsize=(14, 6), facecolor=PALETTE["bg"])
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.25)

    slices = _mid_slices(volume)
    cam_sl = _mid_slices(cam)
    views  = ["Axial", "Coronal", "Sagittal"]

    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(slices[i].T, cmap="gray", origin="lower", aspect="auto")
        ax.set_title(views[i], color=PALETTE["text"], fontsize=9)
        ax.axis("off")

        ax2 = fig.add_subplot(gs[1, i])
        ax2.imshow(slices[i].T,  cmap="gray",  origin="lower", aspect="auto")
        ax2.imshow(cam_sl[i].T,  cmap="hot",   origin="lower", aspect="auto", alpha=0.5)
        ax2.set_title(f"Grad-CAM {views[i]}", color=PALETTE["text"], fontsize=9)
        ax2.axis("off")

    # Gauge panel
    ax_g = fig.add_subplot(gs[:, 3])
    ax_g.set_facecolor(PALETTE["panel"])
    ax_g.set_xlim(0, 1); ax_g.set_ylim(0, 1)
    ax_g.axis("off")

    # Draw simple bar gauge
    ax_g.add_patch(plt.Rectangle((0.15, 0.2), 0.7, 0.08,
                                  facecolor=PALETTE["bg"], edgecolor=PALETTE["muted"], lw=1))
    ax_g.add_patch(plt.Rectangle((0.15, 0.2), 0.7 * prob, 0.08,
                                  facecolor=color))
    ax_g.text(0.5, 0.55, f"{prob*100:.1f}%", ha="center", va="center",
              color="white", fontsize=28, fontweight="bold",
              fontfamily="monospace", transform=ax_g.transAxes)
    ax_g.text(0.5, 0.42, label, ha="center", va="center",
              color=color, fontsize=13, fontweight="bold", transform=ax_g.transAxes)
    ax_g.text(0.5, 0.15, "Stroke Probability", ha="center",
              color=PALETTE["muted"], fontsize=9, transform=ax_g.transAxes)
    for spine in ax_g.spines.values():
        spine.set_edgecolor(PALETTE["muted"])

    fig.suptitle(title, color=PALETTE["text"], fontsize=13, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"✓ Saved prediction plot → {out_path}")


def _plot_comparison(vol_b, cam_b, prob_b, lbl_b, col_b,
                     vol_s, cam_s, prob_s, lbl_s, col_s,
                     delta, direction, out_path):
    fig = plt.figure(figsize=(16, 8), facecolor=PALETTE["bg"])
    gs  = gridspec.GridSpec(3, 5, figure=fig, hspace=0.35, wspace=0.25)

    slb = _mid_slices(vol_b);  csb = _mid_slices(cam_b)
    sls = _mid_slices(vol_s);  css = _mid_slices(cam_s)
    views = ["Axial", "Coronal", "Sagittal"]

    for i in range(3):
        # Baseline row
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(slb[i].T, cmap="gray", origin="lower", aspect="auto")
        ax.set_title(f"Base {views[i]}", color=PALETTE["text"], fontsize=8)
        ax.axis("off")

        ax2 = fig.add_subplot(gs[1, i])
        ax2.imshow(sls[i].T, cmap="gray", origin="lower", aspect="auto")
        ax2.set_title(f"Follow-up {views[i]}", color=PALETTE["text"], fontsize=8)
        ax2.axis("off")

        # Difference
        ax3 = fig.add_subplot(gs[2, i])
        diff = np.abs(css[i] - csb[i])
        ax3.imshow(diff.T, cmap="RdYlGn_r", origin="lower", aspect="auto", vmin=0, vmax=1)
        ax3.set_title(f"Δ {views[i]}", color=PALETTE["text"], fontsize=8)
        ax3.axis("off")

    # Summary panel
    ax_s = fig.add_subplot(gs[:, 3:])
    ax_s.set_facecolor(PALETTE["panel"]); ax_s.axis("off")

    def bar(ax, y, prob, color, label):
        ax.add_patch(plt.Rectangle((0.05, y), 0.90 * prob, 0.06,
                                    facecolor=color, transform=ax.transAxes))
        ax.add_patch(plt.Rectangle((0.05, y), 0.90, 0.06,
                                    facecolor="none", edgecolor=PALETTE["muted"],
                                    lw=1, transform=ax.transAxes))
        ax.text(0.5, y + 0.085, f"{label}  {prob*100:.1f}%  [{risk_label(prob)[0]}]",
                ha="center", color=color, fontsize=10, fontweight="bold",
                transform=ax.transAxes)

    bar(ax_s, 0.72, prob_b, col_b, "Baseline")
    bar(ax_s, 0.52, prob_s, col_s, "Follow-up")

    delta_color = PALETTE["accent1"] if delta > 0 else PALETTE["accent3"]
    ax_s.text(0.5, 0.36, f"Δ = {delta:+.3f}", ha="center",
              color=delta_color, fontsize=16, fontweight="bold", transform=ax_s.transAxes)
    ax_s.text(0.5, 0.26, direction, ha="center",
              color=delta_color, fontsize=12, transform=ax_s.transAxes)

    for spine in ax_s.spines.values():
        spine.set_edgecolor(PALETTE["muted"])

    fig.suptitle("Brain Stroke — Longitudinal Comparison",
                 color=PALETTE["text"], fontsize=14, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"✓ Saved comparison plot → {out_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       type=str, required=True,
                        choices=["test", "predict", "compare"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir",   type=str, default="data/")
    parser.add_argument("--scan",       type=str, default=None,
                        help="Path to .nii / .nii.gz scan for predict / compare")
    parser.add_argument("--baseline",   type=str, default=None,
                        help="Path to baseline scan for compare mode")
    parser.add_argument("--out_dir",    type=str, default="results/")
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, model_args = load_checkpoint(args.checkpoint, device)

    if args.mode == "test":
        evaluate_test_set(model, args.data_dir, model_args, device, out_dir)

    elif args.mode == "predict":
        assert args.scan, "--scan is required for predict mode"
        predict_single(model, args.scan, model_args, device, out_dir)

    elif args.mode == "compare":
        assert args.scan and args.baseline, "--scan and --baseline required for compare mode"
        compare_scans(model, args.scan, args.baseline, model_args, device, out_dir)
