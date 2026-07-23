import pydicom
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Caminhos ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
patient_dir  = PROJECT_ROOT / "data/raw/cbis_ddsm/CBIS-DDSM/Mass-Training_P_00001_LEFT_CC_1"

dcm_files = sorted(patient_dir.rglob("*.dcm"))
if not dcm_files:
    raise FileNotFoundError(f"Nenhum .dcm encontrado em: {patient_dir}")

print(f"\n{'='*50}")
print(f"Arquivos encontrados: {len(dcm_files)}")
for f in dcm_files:
    print(f"  {f}")
print(f"{'='*50}\n")


# ── Upgrade 2 — Window Level (visão de radiologista) ─────────────────────────
def window_image(img, center, width):
    img_min = center - width // 2
    img_max = center + width // 2
    img = np.clip(img, img_min, img_max)
    img = (img - img_min) / (img_max - img_min)
    return img


# ── Upgrade 4 — Loop em todas as imagens do paciente ─────────────────────────
for file in dcm_files:
    dicom = pydicom.dcmread(file)

    # ── Upgrade 3 — Metadados clínicos ───────────────────────────────────────
    print(f"Arquivo  : {file.name}")
    print(f"Paciente : {getattr(dicom, 'PatientID',  'N/A')}")
    print(f"Modality : {getattr(dicom, 'Modality',   'N/A')}")
    print(f"View     : {getattr(dicom, 'ViewPosition','N/A')}")
    print(f"Rows     : {getattr(dicom, 'Rows',        'N/A')}")
    print(f"Columns  : {getattr(dicom, 'Columns',     'N/A')}")
    print()

    raw = dicom.pixel_array

    # ── Upgrade 2 — tenta Window Level; fallback para normalização simples ────
    if hasattr(dicom, "WindowCenter") and hasattr(dicom, "WindowWidth"):
        center = dicom.WindowCenter
        width  = dicom.WindowWidth
        if isinstance(center, pydicom.multival.MultiValue):
            center = center[0]
        if isinstance(width, pydicom.multival.MultiValue):
            width = width[0]
        image = window_image(raw.astype(float), float(center), float(width))
    else:
        # ── Upgrade 1 — normalização 0-1 (fallback) ──────────────────────────
        image = raw.astype(float)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="gray")
    plt.title(file.name)
    plt.axis("off")

plt.show()
