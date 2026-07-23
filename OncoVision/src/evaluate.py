"""
evaluate.py — OncoVision Clinical Evaluation
Avalia o modelo treinado com métricas médicas reais.

Métricas:
  - Accuracy       : acerto geral
  - AUC-ROC        : capacidade de separar benigno/maligno
  - Sensitivity    : taxa de detecção de câncer (recall dos malignos) — crítico
  - Specificity    : taxa de acerto nos benignos
  - Precision      : dos que o modelo disse maligno, quantos realmente são
  - F1-Score       : equilíbrio precision/recall
  - Confusion Matrix

Em medicina, Sensitivity é a métrica mais importante:
  → falso negativo (câncer não detectado) é muito pior que falso positivo.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
import matplotlib.pyplot as plt

from dataset import CBISDDSMDataset
from model import OncoVisionNet

# ── Configuração ──────────────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 512
BATCH_SIZE   = 4
THRESHOLD    = 0.55

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# aceita best_model ou last_model
MODEL_PATH = MODELS_DIR / "best_model.pth"
if not MODEL_PATH.exists():
    MODEL_PATH = MODELS_DIR / "last_model.pth"

print(f"Device     : {DEVICE}")
print(f"Modelo     : {MODEL_PATH}")

# ── Transforms (mesmos do treino) ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# ── Dataset e Loader ─────────────────────────────────────────────────────────
dataset = CBISDDSMDataset(transform=transform, use_cropped=True)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Modelo ────────────────────────────────────────────────────────────────────
model = OncoVisionNet(pretrained=False).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ── Inferência ────────────────────────────────────────────────────────────────
y_true, y_prob = [], []

print("\nRodando inferência...")
with torch.no_grad():
    for i, (images, labels) in enumerate(loader):
        if images.dim() == 3:
            images = images.unsqueeze(1)
        images = images.to(DEVICE)

        logits = model(images)
        probs  = torch.sigmoid(logits).cpu().numpy().flatten()

        y_true.extend(labels.numpy().tolist())
        y_prob.extend(probs.tolist())

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(loader)} batches processados...")

y_true = np.array(y_true)
y_prob = np.array(y_prob)
y_pred = (y_prob >= THRESHOLD).astype(int)

# ── Métricas clínicas ─────────────────────────────────────────────────────────
acc = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn + 1e-8)   # recall dos malignos — mais importante
specificity = tn / (tn + fp + 1e-8)
precision   = tp / (tp + fp + 1e-8)
f1          = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-8)

print(f"\n{'='*45}")
print(f"  RESULTADOS CLÍNICOS — OncoVision")
print(f"{'='*45}")
print(f"  Accuracy    : {acc:.4f}  ({acc*100:.1f}%)")
print(f"  AUC-ROC     : {auc:.4f}")
print(f"  Sensitivity : {sensitivity:.4f}  ({sensitivity*100:.1f}%)  ← detecção de câncer")
print(f"  Specificity : {specificity:.4f}  ({specificity*100:.1f}%)")
print(f"  Precision   : {precision:.4f}  ({precision*100:.1f}%)")
print(f"  F1-Score    : {f1:.4f}")
print(f"{'='*45}")
print(f"\n  Confusion Matrix:")
print(f"                 Pred Benigno  Pred Maligno")
print(f"  Real Benigno       {tn:>5}         {fp:>5}")
print(f"  Real Maligno       {fn:>5}         {tp:>5}")
print(f"\n  Falsos Negativos (câncer não detectado): {fn}")
print(f"  Falsos Positivos (alarme falso)         : {fp}")

print(f"\n{classification_report(y_true, y_pred, target_names=['Benigno', 'Maligno'])}")

# ── Curva ROC ─────────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_true, y_prob)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="crimson", lw=2, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("OncoVision — Curva ROC")
plt.legend(loc="lower right")
plt.tight_layout()

roc_path = OUTPUTS_DIR / "roc_curve.png"
plt.savefig(roc_path, dpi=150)
print(f"Curva ROC salva em: {roc_path}")
plt.show()
