"""
threshold_analysis.py — OncoVision Clinical Threshold Analysis

Gera:
  1. Tabela Sensitivity vs Specificity por threshold
  2. Gráfico publicável: Sensitivity/Specificity vs Threshold
  3. Curva ROC com AUC
  4. Ponto ótimo clínico (Sensitivity > 92% e Specificity máxima)

Execute após o treino:
  python threshold_analysis.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve

from dataset import CBISDDSMDataset
from model import OncoVisionNet

# ── Configuração ──────────────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 224
BATCH_SIZE   = 8
SEED         = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = PROJECT_ROOT / "models" / "best_model.pth"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

print(f"Device : {DEVICE}")
print(f"Modelo : {MODEL_PATH}\n")

# ── Dataset — mesmo split do treino ──────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

ds       = CBISDDSMDataset(transform=transform, use_cropped=True)
patients = ds.df["patient_id"].unique().to_numpy()
rng      = np.random.default_rng(SEED)
rng.shuffle(patients)

val_patients = set(patients[:int(len(patients) * 0.2)])
val_idx      = ds.df[ds.df["patient_id"].isin(val_patients)].index.tolist()
val_loader   = DataLoader(Subset(ds, val_idx), batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

print(f"Validação: {len(val_idx)} imagens\n")

# ── Modelo ────────────────────────────────────────────────────────────────────
model = OncoVisionNet(pretrained=False).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ── Inferência — coleta probabilidades ───────────────────────────────────────
y_true, y_prob = [], []

print("Rodando inferência...")
with torch.no_grad(), torch.amp.autocast(device_type=DEVICE, enabled=False):
    for images, labels in val_loader:
        if images.dim() == 3:
            images = images.unsqueeze(1)
        images = images.to(DEVICE)
        logits = model(images)
        probs  = torch.sigmoid(logits).cpu().numpy().flatten()
        y_true.extend(labels.numpy().tolist())
        y_prob.extend(probs.tolist())

y_true = np.array(y_true)
y_prob = np.array(y_prob)

auc = roc_auc_score(y_true, y_prob)
print(f"AUC-ROC: {auc:.4f}\n")

# ── Análise por threshold ─────────────────────────────────────────────────────
thresholds  = np.arange(0.10, 0.91, 0.05)
sensitivities, specificities, f1s = [], [], []

print(f"{'Threshold':>10}  {'Sensitivity':>11}  {'Specificity':>11}  {'F1':>6}  {'FN':>4}  {'FP':>4}")
print("-" * 60)

best_threshold = 0.5
best_score     = -1

for t in thresholds:
    pred = (y_prob >= t).astype(int)
    tp = ((pred == 1) & (y_true == 1)).sum()
    tn = ((pred == 0) & (y_true == 0)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()

    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    f1   = 2 * (prec * sens) / (prec + sens + 1e-8)

    sensitivities.append(sens)
    specificities.append(spec)
    f1s.append(f1)

    # ponto ótimo: sensitivity > 92% com maior specificity
    if sens >= 0.92 and spec > best_score:
        best_score     = spec
        best_threshold = t

    marker = " ← ótimo clínico" if (sens >= 0.92 and spec == best_score and t == best_threshold) else ""
    print(f"  {t:>8.2f}   {sens*100:>9.1f}%   {spec*100:>9.1f}%  {f1:.3f}  {fn:>4}  {fp:>4}{marker}")

print(f"\n✅ Threshold clínico recomendado: {best_threshold:.2f}")
print(f"   Sensitivity : {sensitivities[list(thresholds).index(best_threshold) if best_threshold in thresholds else 0]*100:.1f}%")
print(f"   Specificity : {best_score*100:.1f}%")

# ── Gráfico publicável ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5))
gs  = gridspec.GridSpec(1, 2, figure=fig)

# — Plot 1: Sensitivity / Specificity vs Threshold —
ax1 = fig.add_subplot(gs[0])
ax1.plot(thresholds, [s * 100 for s in sensitivities],
         color="#e63946", lw=2.5, marker="o", markersize=5, label="Sensitivity (Recall Maligno)")
ax1.plot(thresholds, [s * 100 for s in specificities],
         color="#457b9d", lw=2.5, marker="s", markersize=5, label="Specificity")
ax1.plot(thresholds, [f * 100 for f in f1s],
         color="#2a9d8f", lw=2, linestyle="--", label="F1-Score")

ax1.axvline(x=best_threshold, color="gray", linestyle=":", lw=1.5,
            label=f"Threshold ótimo = {best_threshold:.2f}")
ax1.axhline(y=92, color="#e63946", linestyle=":", lw=1, alpha=0.5)

ax1.set_xlabel("Threshold de Decisão", fontsize=12)
ax1.set_ylabel("Métrica (%)", fontsize=12)
ax1.set_title("OncoVision — Análise de Threshold Clínico", fontsize=13, fontweight="bold")
ax1.legend(fontsize=10)
ax1.set_xlim(0.10, 0.90)
ax1.set_ylim(0, 105)
ax1.grid(True, alpha=0.3)

# — Plot 2: Curva ROC —
fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
ax2 = fig.add_subplot(gs[1])
ax2.plot(fpr, tpr, color="#e63946", lw=2.5, label=f"AUC = {auc:.3f}")
ax2.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)

# marca o threshold ótimo na curva ROC
idx_opt = np.argmin(np.abs(roc_thresh - best_threshold))
ax2.scatter(fpr[idx_opt], tpr[idx_opt], color="#e63946", s=120, zorder=5,
            label=f"Threshold = {best_threshold:.2f}")

ax2.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
ax2.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
ax2.set_title("OncoVision — Curva ROC", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = OUTPUTS_DIR / "threshold_analysis.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nGráfico salvo: {out_path}")
plt.show()
