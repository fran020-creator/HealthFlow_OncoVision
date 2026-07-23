import pydicom
import matplotlib.pyplot as plt
from pathlib import Path

# raiz do projeto (OncoVision/) — dois níveis acima de src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CBIS_DDSM_DIR = PROJECT_ROOT / "data/raw/cbis_ddsm/CBIS-DDSM"


def load_dicom(patient_folder: str) -> pydicom.Dataset:
    """
    Carrega o primeiro arquivo .dcm encontrado dentro da pasta do paciente.
    Usa rglob para navegar automaticamente pelas subpastas de data/série.
    """
    patient_path = CBIS_DDSM_DIR / patient_folder
    dcm_files = sorted(patient_path.rglob("*.dcm"))

    if not dcm_files:
        raise FileNotFoundError(f"Nenhum .dcm encontrado em: {patient_path}")

    print(f"Arquivo encontrado: {dcm_files[0]}")
    return pydicom.dcmread(dcm_files[0])


if __name__ == "__main__":
    ds = load_dicom("Mass-Training_P_00001_LEFT_CC_1")

    image = ds.pixel_array

    plt.imshow(image, cmap="gray")
    plt.title("Mamografia - OncoVision")
    plt.axis("off")
    plt.show()