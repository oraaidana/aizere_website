import io, os, base64, torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List
import torch.nn.functional as F
# Import your local architecture
from model import StrokeCNN3D, GradCAM3D
from preprocess import validate_scan
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Brain Stroke Diagnostic API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your website's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "checkpoints/best_model.pth"


# Load model globally to keep it in VRAM
def get_model():
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    args = ckpt.get("args", {})
    m = StrokeCNN3D(base_filters=args.get("base_filters", 16)).to(DEVICE)
    m.load_state_dict(ckpt.get("model_state", ckpt))
    m.eval()
    # Apply Temperature T if saved in checkpoint, otherwise default to 1.5 for stability
    T = ckpt.get("temperature", 1.5)
    return m, args, T


MODEL, MODEL_ARGS, TEMP = get_model()


# --- Response Schema ---
class ScanResult(BaseModel):
    id: str
    probability: float
    risk_level: str
    risk_color: str
    heatmap_b64: str


# --- Internal Logic ---
def get_risk_meta(p):
    if p > 0.20: return "HIGH RISK", "#FF3B30"
    if 0.09 <= p <= 0.020: return "MODERATE", "#FF9F0A"
    return "LOW RISK", "#30D158"


def create_viz_b64(tensor, cam):
    vol = tensor.squeeze().numpy()
    slices = [vol[32, :, :], vol[:, 32, :], vol[:, :, 16]]  # Mid-slices
    cam_sl = [cam[32, :, :], cam[:, 32, :], cam[:, :, 16]]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), facecolor="#0D0D1A")
    for i in range(3):
        axes[i].imshow(slices[i].T, cmap="gray", origin="lower")
        axes[i].imshow(cam_sl[i].T, cmap="hot", origin="lower", alpha=0.4)
        axes[i].axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# --- Endpoints ---
@app.get("/")
async def root():
    return {
        "message": "Brain Stroke Diagnostic API is running",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "docs": "/docs (Swagger UI)"
        }
    }
@app.post("/predict", response_model=ScanResult)
async def predict_stroke(file: UploadFile = File(...)):
    # 1. Validate extension
    if not file.filename.endswith(('.nii', '.nii.gz')):
        raise HTTPException(400, "File must be NIfTI format")

    # 2. Save temporary file for Nibabel
    temp_path = f"tmp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        # 3. Process & Calibrated Inference
        target = tuple(MODEL_ARGS.get("input_shape", [64, 64, 32]))
        tensor, _ = validate_scan(temp_path, target)

        with torch.no_grad():
            logits = MODEL(tensor.to(DEVICE))
            # Apply Temperature Scaling to fix the 99% vs 4% issue
            # Apply softmax to get probabilities for [Healthy, Stroke]
            #probs = F.softmax(logits / TEMP, dim=1)

            # Extract the probability for the Stroke class (index 1)
            prob = F.softmax(logits, dim=1)[0, 1].item()

            # 4. Generate Explainability Heatmap
        cam = GradCAM3D(MODEL)(tensor.to(DEVICE), class_idx=1)

        risk_lvl, risk_col = get_risk_meta(prob)
        img_str = create_viz_b64(tensor, cam)

        return {
            "id": file.filename,
            "probability": prob,
            "risk_level": risk_lvl,
            "risk_color": risk_col,
            "heatmap_b64": img_str
        }

    finally:
        if os.path.exists(temp_path): os.remove(temp_path)


@app.get("/health")
def health():
    return {"status": "ready", "gpu": torch.cuda.is_available()}