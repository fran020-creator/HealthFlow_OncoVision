"""
load_labels.py — Inspeciona os CSVs clínicos do CBIS-DDSM
Execute: python load_labels.py
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

mass_csv = PROJECT_ROOT / "data/raw/mass_case_description_train_set.csv"
calc_csv = PROJECT_ROOT / "data/raw/calc_case_description_train_set.csv"


def inspect_csv(path: Path):
    df = pd.read_csv(path)

    print(f"\n{'='*60}")
    print(f"Arquivo : {path.name}")
    print(f"Linhas  : {len(df)}  |  Colunas: {len(df.columns)}")
    print(f"{'='*60}")

    print("\n── Colunas ──")
    for col in df.columns:
        print(f"  {col}")

    print("\n── Distribuição de patologia ──")
    print(df["pathology"].value_counts().to_string())

    print("\n── Primeiras linhas ──")
    cols_show = ["patient_id", "left or right breast", "image view",
                 "pathology", "mass shape", "mass margins", "assessment"]
    cols_show = [c for c in cols_show if c in df.columns]
    print(df[cols_show].head(5).to_string(index=False))

    return df


if __name__ == "__main__":
    df_mass = inspect_csv(mass_csv)

    if calc_csv.exists():
        df_calc = inspect_csv(calc_csv)
    else:
        print(f"\n⚠️  {calc_csv.name} não encontrado — pulando.")
