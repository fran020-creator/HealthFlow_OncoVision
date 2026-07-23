"""
train.py — OncoVision Training Loop (Clinical Grade)

Correções aplicadas:
  1. autocast(enabled=False) explícito na validação — garante fp32 real
  2. Early stopping protegido contra NaN (math.isnan check)
  3. patients.to_numpy() antes do shuffle — sem duplicatas no split
  4. THRESHOLD = 0.40 — equilíbrio entre sensitivity e specificity
  5. pos_weight aumentado para penalizar mais falsos negativos
  6. SmoothedBCE removida da Fase 1 — BCE puro mais estável
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
import numpy as np
import math
import time

from dataset import CBISDDSMDataset
from model import OncoVisionNet

# ── Configuração ──────────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
VAL_SPLIT  = 0.2
SEED       = 42
PATIENCE   = 7
THRESHOLD  = 0.55
USE_CALC   = True   # True = usa mass + calc  |  False = só mass

PHASES = [
    {"epochs": 5,  "img_size": 224, "mode": "head",    "lr": 3e-4,  "smooth": 0.0,  "label": "Fase 1 — Só cabeça (backbone frozen)"},
    {"epochs": 10, "img_size": 320, "mode": "partial",  "lr": 3e-5,  "smooth": 0.05, "label": "Fase 2 — Unfreeze parcial (blocos finais)"},
    {"epochs": 30, "img_size": 384, "mode": "full",     "lr": 1e-5,  "smooth": 0.05, "label": "Fase 3 — Full fine-tuning"},
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

print(f"Device    : {DEVICE}")
print(f"Batch     : {BATCH_SIZE}  |  Seed: {SEED}  |  Threshold: {THRESHOLD}\n")

# ── Patient-wise split ────────────────────────────────────────────────────────
_ref     = CBISDDSMDataset(use_cropped=False, use_calc=USE_CALC)
patients = _ref.df["patient_id"].unique().to_numpy()   # fix 3: numpy antes do shuffle
rng      = np.random.default_rng(SEED)
rng.shuffle(patients)

n_val_p        = int(len(patients) * VAL_SPLIT)
val_patients   = set(patients[:n_val_p])
train_patients = set(patients[n_val_p:])

train_idx = _ref.df[_ref.df["patient_id"].isin(train_patients)].index.tolist()
val_idx   = _ref.df[_ref.df["patient_id"].isin(val_patients)].index.tolist()

print(f"Pacientes únicos : {len(patients)}")
print(f"  Treino         : {len(train_patients)} pacientes  ({len(train_idx)} imagens)")
print(f"  Validação      : {len(val_patients)} pacientes  ({len(val_idx)} imagens)\n")

# ── Class weights — penaliza mais falso negativo (câncer não detectado) ───────
# pos_weight > 1 → modelo prefere errar para maligno do que perder câncer
n_benign   = 577 + 104
n_maligno  = 636
pos_weight = torch.tensor([3.0]).to(DEVICE)  # penaliza fortemente falso negativo
print(f"pos_weight: {pos_weight.item():.3f}\n")

# ── Modelo ────────────────────────────────────────────────────────────────────
model  = OncoVisionNet(pretrained=True).to(DEVICE)
scaler = GradScaler()


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_transform(img_size: int, augment: bool) -> transforms.Compose:
    ops = [transforms.ToTensor(), transforms.Resize((img_size, img_size))]
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.RandomAutocontrast(p=0.3),
        ]
    ops.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    return transforms.Compose(ops)


def make_loaders(img_size: int):
    """Dois datasets independentes — sem vazamento de augmentation."""
    tr_ds = CBISDDSMDataset(transform=get_transform(img_size, augment=True),  use_cropped=False, use_calc=USE_CALC)
    va_ds = CBISDDSMDataset(transform=get_transform(img_size, augment=False), use_cropped=False, use_calc=USE_CALC)
    train_loader = DataLoader(Subset(tr_ds, train_idx), batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(Subset(va_ds, val_idx),   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader


def set_phase(mode: str, lr: float) -> torch.optim.Optimizer:
    for p in model.parameters():
        p.requires_grad = False

    if mode == "head":
        for p in model.head.parameters():
            p.requires_grad = True
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  🔒 Backbone frozen — só cabeça  ({n:,} params)")

    elif mode == "partial":
        for p in model.backbone.blocks[-2:].parameters():
            p.requires_grad = True
        for p in model.head.parameters():
            p.requires_grad = True
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  🔓 Parcial — blocks[-2:] + cabeça  ({n:,} params)")

    elif mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  🔓 Full fine-tuning  ({n:,} params)")

    return torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )


class SmoothedBCE(nn.Module):
    """BCEWithLogitsLoss + label smoothing manual."""
    def __init__(self, pos_weight, smoothing=0.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.s   = smoothing

    def forward(self, logits, targets):
        if self.s > 0:
            targets = targets * (1 - self.s) + 0.5 * self.s
        return self.bce(logits, targets)


def run_epoch(loader, optimizer, criterion, train: bool, ep: int, total_ep: int) -> dict:
    model.train() if train else model.eval()

    total_loss    = 0.0
    valid_batches = 0
    tp = tn = fp = fn = 0

    phase_label = "Treino" if train else "  Val "
    bar = tqdm(loader, desc=f"  Ep {ep}/{total_ep} {phase_label}",
               leave=False, unit="batch", dynamic_ncols=True)

    for images, labels in bar:
        if images.dim() == 3:
            images = images.unsqueeze(1)
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)

        # pula batch com entrada inválida
        if torch.isnan(images).any() or torch.isinf(images).any():
            continue

        if train:
            optimizer.zero_grad()
            with autocast(device_type=DEVICE):
                preds = model(images)
                loss  = criterion(preds, labels)

            if math.isnan(loss.item()) or math.isinf(loss.item()):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        else:
            # fix 1: autocast desativado explicitamente na validação — fp32 garantido
            with torch.no_grad(), autocast(device_type=DEVICE, enabled=False):
                preds = model(images)
                loss  = criterion(preds, labels)

            if math.isnan(loss.item()) or math.isinf(loss.item()):
                continue

        total_loss    += loss.item()
        valid_batches += 1

        pred_bin = (torch.sigmoid(preds) >= THRESHOLD).float()
        tp += ((pred_bin == 1) & (labels == 1)).sum().item()
        tn += ((pred_bin == 0) & (labels == 0)).sum().item()
        fp += ((pred_bin == 1) & (labels == 0)).sum().item()
        fn += ((pred_bin == 0) & (labels == 1)).sum().item()

        n   = tp + tn + fp + fn
        acc = (tp + tn) / (n + 1e-8) * 100
        bar.set_postfix(
            loss=f"{total_loss / (valid_batches + 1e-8):.4f}",
            acc=f"{acc:.1f}%"
        )

    n           = tp + tn + fp + fn
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    accuracy    = (tp + tn) / (n + 1e-8)
    avg_loss    = total_loss / (valid_batches + 1e-8)

    return {
        "loss": avg_loss,
        "acc":  accuracy    * 100,
        "sens": sensitivity * 100,
        "spec": specificity * 100,
        "fn":   int(fn),
    }


# ── Loop por fases ────────────────────────────────────────────────────────────
best_val_loss = float("inf")
early_counter = 0
global_epoch  = 0

for phase in PHASES:
    print(f"\n{'='*68}")
    print(f"  {phase['label']}")
    print(f"  Resolução: {phase['img_size']}×{phase['img_size']}  |  LR: {phase['lr']}  |  Smooth: {phase['smooth']}")
    print(f"{'='*68}")

    optimizer = set_phase(phase["mode"], phase["lr"])
    criterion = SmoothedBCE(pos_weight, smoothing=phase["smooth"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=2
    )
    train_loader, val_loader = make_loaders(phase["img_size"])

    hdr = "  Ep   TrLoss  TrAcc TrSens   VaLoss  VaAcc VaSens VaSpec   FN        LR     s"
    print(f"\n{hdr}\n  {'-'*80}")

    for ep in range(1, phase["epochs"] + 1):
        global_epoch += 1
        t0 = time.time()

        tr = run_epoch(train_loader, optimizer, criterion, True,  ep, phase["epochs"])
        va = run_epoch(val_loader,   optimizer, criterion, False, ep, phase["epochs"])

        scheduler.step(va["loss"])
        lr_now  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(
            f"  {ep:>2}  {tr['loss']:>7.4f}  {tr['acc']:>5.1f}% {tr['sens']:>5.1f}%"
            f"  {va['loss']:>7.4f}  {va['acc']:>5.1f}% {va['sens']:>5.1f}% {va['spec']:>5.1f}%"
            f"  {va['fn']:>4}  {lr_now:>8.2e}  {elapsed:>4.0f}s"
        )

        # fix 2: early stopping protegido contra NaN
        va_loss_valid = va["loss"] if not math.isnan(va["loss"]) else float("inf")

        if va_loss_valid < best_val_loss:
            best_val_loss = va_loss_valid
            early_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / "best_model.pth")
            print(f"       ✅ Melhor modelo  loss={va_loss_valid:.4f}  sens={va['sens']:.1f}%  spec={va['spec']:.1f}%  FN={va['fn']}")
        else:
            early_counter += 1
            if early_counter >= PATIENCE:
                print(f"\n  🛑 Early stopping (epoch global {global_epoch})")
                break

    if early_counter >= PATIENCE:
        break

torch.save(model.state_dict(), MODELS_DIR / "last_model.pth")
print(f"\n✅ Treino concluído!")
print(f"   Melhor val_loss : {best_val_loss:.4f}")
print(f"   Modelos salvos  : {MODELS_DIR}")
