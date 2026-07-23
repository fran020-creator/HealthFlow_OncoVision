import torch
import numpy as np
import sys
sys.path.insert(0, '.')
from dataset import CBISDDSMDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

ds = CBISDDSMDataset(transform=transform)

patients = ds.df["patient_id"].unique()
rng = np.random.default_rng(42)
rng.shuffle(patients)
val_patients = set(patients[:int(len(patients) * 0.2)])
val_idx = ds.df[ds.df["patient_id"].isin(val_patients)].index.tolist()

print(f"Verificando {len(val_idx)} amostras do val set...")
bad = []
for i in val_idx:
    img, _ = ds[i]
    has_nan = torch.isnan(img).any().item()
    has_inf = torch.isinf(img).any().item()
    if has_nan or has_inf:
        bad.append(i)
        pid = ds.df.iloc[i]["patient_id"]
        print(f"  BAD idx={i}  patient={pid}  nan={has_nan}  inf={has_inf}  min={img.min():.3f}  max={img.max():.3f}")

print(f"\nTotal problemáticos: {len(bad)} de {len(val_idx)}")
