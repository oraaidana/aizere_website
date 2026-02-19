"""
model.py
────────────────────────────────────────────────────────────
3D-CNN model for brain stroke binary classification from NIfTI MRI.

Architecture:
  Encoder:  4 × (Conv3D → BN → ReLU → MaxPool3D)
  Bottleneck: Global Average Pooling
  Head:     FC(256) → Dropout → FC(2)

Also includes:
  - StrokePredictor wrapper (sigmoid probability output)
  - GradCAM3D for saliency visualisation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Building Blocks ──────────────────────────────────────────────────────────

class ConvBlock3D(nn.Module):
    """Conv3D → BatchNorm → ReLU (optionally repeated twice)."""
    def __init__(self, in_ch, out_ch, double=False):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if double:
            layers += [
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation channel attention (3-D)."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1, 1)
        return x * w


# ─── Main Network ─────────────────────────────────────────────────────────────

class StrokeCNN3D(nn.Module):
    """
    Lightweight 3-D CNN for stroke vs. healthy classification.

    Input : (B, 1, D, H, W)   float32 in [0, 1]
    Output: (B, 2)             raw logits
    """

    def __init__(self, in_channels=1, num_classes=2, base_filters=16, dropout=0.5):
        super().__init__()
        f = base_filters  # 16 → 32 → 64 → 128

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = nn.Sequential(ConvBlock3D(in_channels, f,   double=True), SEBlock3D(f))
        self.enc2 = nn.Sequential(ConvBlock3D(f,  f * 2,  double=True), SEBlock3D(f * 2))
        self.enc3 = nn.Sequential(ConvBlock3D(f * 2, f * 4, double=True), SEBlock3D(f * 4))
        self.enc4 = nn.Sequential(ConvBlock3D(f * 4, f * 8, double=False), SEBlock3D(f * 8))

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool3d(1)

        # ── Classifier ────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(f * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder with downsampling
        x = self.pool(self.enc1(x))   # /2
        x = self.pool(self.enc2(x))   # /4
        x = self.pool(self.enc3(x))   # /8
        x = self.enc4(x)              # keep spatial (no pool after last enc)

        # Save last feature map for Grad-CAM hook
        self._last_feature = x        # (B, f*8, D', H', W')

        x = self.gap(x)               # (B, f*8, 1, 1, 1)
        x = x.view(x.size(0), -1)    # (B, f*8)
        return self.classifier(x)     # (B, num_classes)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Probability Wrapper ──────────────────────────────────────────────────────

class StrokePredictor(nn.Module):
    """
    Wraps StrokeCNN3D and returns stroke probability (scalar per sample).
    Use for inference only.
    """
    def __init__(self, backbone: StrokeCNN3D):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        logits = self.backbone(x)               # (B, 2)
        probs  = F.softmax(logits, dim=1)       # (B, 2)
        return probs[:, 1]                      # (B,) stroke probability


# ─── Grad-CAM 3D ─────────────────────────────────────────────────────────────

class GradCAM3D:
    """
    Grad-CAM for StrokeCNN3D: produces a 3-D saliency map.

    Usage:
        cam = GradCAM3D(model)
        heatmap = cam(volume_tensor)   # (D, H, W) numpy
    """

    def __init__(self, model: StrokeCNN3D):
        self.model      = model
        self.gradients  = None
        self.activations= None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, input, output):
            self.activations = output.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        # Hook on the last encoder block (enc4)
        self.model.enc4.register_forward_hook(fwd_hook)
        self.model.enc4.register_full_backward_hook(bwd_hook)

    def __call__(self, x: torch.Tensor, class_idx: int = 1) -> "np.ndarray":
        import numpy as np
        from scipy.ndimage import zoom

        self.model.eval()
        x = x.clone().requires_grad_(True)
        logits = self.model(x)

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Pool gradients over spatial dims
        weights  = self.gradients.mean(dim=(2, 3, 4), keepdim=True)   # (1, C, 1, 1, 1)
        cam      = (weights * self.activations).sum(dim=1).squeeze()   # (D', H', W')
        cam      = F.relu(cam).cpu().numpy()

        # Upsample to input spatial size
        target_size = x.shape[2:]   # (D, H, W)
        factors     = [t / s for t, s in zip(target_size, cam.shape)]
        cam         = zoom(cam, factors, order=1)
        cam         = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.astype(np.float32)


# ─── Quick smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = StrokeCNN3D(base_filters=16)
    print(f"Parameters: {model.count_parameters():,}")

    dummy = torch.zeros(2, 1, 64, 64, 32)
    out   = model(dummy)
    print(f"Output shape: {out.shape}")          # (2, 2)
    print(f"Logits: {out}")

    predictor = StrokePredictor(model)
    probs = predictor(dummy)
    print(f"Stroke probabilities: {probs}")      # (2,)
