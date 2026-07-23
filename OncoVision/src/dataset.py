"""
dataset.py — OncoVision Dataset Loader
Suporta múltiplos CSVs (masses + calcifications) concatenados.
Retorna (imagem float32 normalizada, label binário) pronto para treino.
"""

import pydicom
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

# ── Caminhos base ─────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
CBIS_DDSM_DIR = PROJECT_ROOT / "data" / "raw" / "cbis_ddsm" / "CBIS-DDSM"

CSV_MASS = PROJECT_ROOT / "data" / "raw" / "mass_case_description_train_set.csv"
CSV_CALC = PROJECT_ROOT / "data" / "raw" / "calc_case_description_train_set.csv"

# ── Label binário ─────────────────────────────────────────────────────────────
LABEL_MAP = {
    "MALIGNANT":               1,
    "BENIGN":                  0,
    "BENIGN_WITHOUT_CALLBACK": 0,
}


# ── Utilitários de imagem ─────────────────────────────────────────────────────
def window_image(img: np.ndarray, center: float, width: float) -> np.ndarray:
    lo = center - width / 2
    hi = center + width / 2
    return (np.clip(img, lo, hi) - lo) / (hi - lo)


def normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(float)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def find_patient_dir(patient_folder: str) -> Path | None:
    exact = CBIS_DDSM_DIR / patient_folder
    if exact.is_dir():
        return exact
    for m in sorted(CBIS_DDSM_DIR.glob(f"{patient_folder}_*")):
        if m.is_dir():
            return m
    return None


def resolve_dcm_path(csv_file_path: str) -> Path:
    patient_folder = Path(csv_file_path).parts[0]
    patient_dir    = find_patient_dir(patient_folder)
    if patient_dir is None:
        raise FileNotFoundError(patient_folder)
    dcm_files = sorted(patient_dir.rglob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(patient_folder)
    return dcm_files[0]


def load_dicom_image(dcm_path: Path) -> np.ndarray:
    siblings   = sorted(dcm_path.parent.glob("*.dcm"))
    candidates = [dcm_path] + [f for f in siblings if f != dcm_path]
    for path in candidates:
        try:
            dicom = pydicom.dcmread(path, force=True)
            raw   = dicom.pixel_array.astype(float)
            break
        except Exception:
            pass
    else:
        raise RuntimeError("INVALID_DICOM")

    if hasattr(dicom, "WindowCenter") and hasattr(dicom, "WindowWidth"):
        center = dicom.WindowCenter
        width  = dicom.WindowWidth
        if isinstance(center, pydicom.multival.MultiValue):
            center = center[0]
        if isinstance(width, pydicom.multival.MultiValue):
            width = width[0]
        return window_image(raw, float(center), float(width)).astype(np.float32)
    return normalize(raw).astype(np.float32)


def pathology_to_label(pathology: str) -> int:
    return LABEL_MAP.get(pathology.strip().upper(), 0)


def load_and_merge_csvs(use_calc: bool = False) -> pd.DataFrame:
    """
    Carrega e concatena os CSVs de massa e (opcionalmente) calcificação.
    Normaliza nomes de colunas para garantir compatibilidade entre os dois.
    """
    df_mass = pd.read_csv(CSV_MASS)
    df_mass["lesion_type"] = "mass"

    frames = [df_mass]

    if use_calc and CSV_CALC.exists():
        df_calc = pd.read_csv(CSV_CALC)
        df_calc["lesion_type"] = "calc"

        # calc usa 'breast density' (sem underscore), mass usa 'breast_density'
        if "breast density" in df_calc.columns and "breast_density" not in df_calc.columns:
            df_calc = df_calc.rename(columns={"breast density": "breast_density"})

        frames.append(df_calc)

    df = pd.concat(frames, ignore_index=True)
    df = df[df["pathology"].isin(LABEL_MAP)].reset_index(drop=True)
    return df


# ── Dataset PyTorch ───────────────────────────────────────────────────────────
class CBISDDSMDataset(Dataset):
    """
    Dataset PyTorch para o CBIS-DDSM.

    Parâmetros
    ----------
    transform   : torchvision transforms opcionais
    use_cropped : usa imagem recortada (ROI) se True
    use_calc    : inclui calcificações além das massas (requer imagens no disco)
    """

    def __init__(self, transform=None, use_cropped: bool = False, use_calc: bool = False):
        self.transform   = transform
        self.use_cropped = use_cropped
        self.col         = "cropped image file path" if use_cropped else "image file path"

        df = load_and_merge_csvs(use_calc=use_calc)

        # filtra casos sem pasta no disco
        print("Verificando arquivos no disco...")
        valid_mask = df[self.col].apply(
            lambda p: find_patient_dir(Path(p).parts[0]) is not None
        )
        removed = (~valid_mask).sum()
        if removed:
            print(f"⚠️  {removed} caso(s) removido(s): pasta não encontrada no disco")
        df = df[valid_mask].reset_index(drop=True)

        self.df = df
        n_mass = (df["lesion_type"] == "mass").sum() if "lesion_type" in df.columns else len(df)
        n_calc = (df["lesion_type"] == "calc").sum() if "lesion_type" in df.columns else 0

        print(f"Dataset pronto: {len(df)} casos  (mass={n_mass}  calc={n_calc})")
        print(df["pathology"].value_counts().to_string())

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        while True:
            row   = self.df.iloc[idx]
            label = pathology_to_label(row["pathology"])
            try:
                dcm_path = resolve_dcm_path(row[self.col])
                image    = load_dicom_image(dcm_path)
                break
            except (RuntimeError, FileNotFoundError):
                idx = np.random.randint(0, len(self.df))

        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_info(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        return {
            "patient_id":   row["patient_id"],
            "lesion_type":  row.get("lesion_type", "mass"),
            "side":         row["left or right breast"],
            "view":         row["image view"],
            "pathology":    row["pathology"],
            "assessment":   row["assessment"],
            "subtlety":     row["subtlety"],
        }


# ── Teste rápido ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # testa só massas
    ds_mass = CBISDDSMDataset(transform=transform, use_calc=False)
    print(f"\nSó massas: {len(ds_mass)} casos")

    # testa com calcificações (se disponível)
    ds_both = CBISDDSMDataset(transform=transform, use_calc=True)
    print(f"Mass + Calc: {len(ds_both)} casos")
