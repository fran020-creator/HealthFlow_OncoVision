<div align="center">

<img src="assets/banner.png" alt="OncoVision AI Banner" width="380" style="max-width: 100%; border-radius: 12px; margin-bottom: 12px;" />

# 🔬 OncoVision

**Detecção de câncer de mama por mamografia com deep learning clínico**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-EfficientNet--B4-00B4D8?style=flat-square)](https://github.com/huggingface/pytorch-image-models)
[![Dataset](https://img.shields.io/badge/Dataset-CBIS--DDSM-4CAF50?style=flat-square)](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## 📋 Visão Geral

O **OncoVision** é um sistema de classificação binária de mamografias (benigno × maligno) treinado sobre o dataset público **CBIS-DDSM**. O modelo é baseado em **EfficientNet-B4** com fine-tuning progressivo em três fases e foi projetado com métricas de qualidade clínica em mente — priorizando **sensitivity** (taxa de detecção de câncer) acima de accuracy.

> **Contexto clínico:** um falso negativo — câncer não detectado — é muito mais grave que um falso positivo. Por isso o modelo usa `pos_weight` elevado e threshold calibrado para maximizar sensitivity com specificity aceitável.

---

## ✨ Destaques

- 🏥 **Foco clínico real** — métricas de sensitivity, specificity e AUC-ROC
- 🧠 **EfficientNet-B4** pré-treinado no ImageNet, adaptado para imagens grayscale
- 📈 **Fine-tuning em 3 fases** — head → parcial → full, com resoluções crescentes (224 → 320 → 384)
- 🔬 **Entrada DICOM nativa** — lê os arquivos `.dcm` diretamente, com Window Level automático
- 🛡️ **Treinamento robusto** — AMP (mixed precision), gradient clipping, early stopping e proteção contra NaN
- 📊 **Análise de threshold** — gráfico publicável Sensitivity × Specificity por threshold

---

## 🗂️ Estrutura do Projeto

```
OncoVision/
│
├── data/
│   └── raw/
│       ├── cbis_ddsm/
│       │   └── CBIS-DDSM/          ← imagens .dcm dos pacientes
│       ├── mass_case_description_train_set.csv
│       └── calc_case_description_train_set.csv
│
├── models/
│   ├── best_model.pth              ← melhor checkpoint (val_loss)
│   └── last_model.pth              ← último epoch
│
├── notebooks/                      ← Jupyter notebooks (exploração)
│
├── outputs/
│   ├── image.png                   ← banner do projeto
│   ├── roc_curve.png
│   └── threshold_analysis.png
│
├── src/
│   ├── dataset.py                  ← Dataset PyTorch (DICOM + CSV)
│   ├── model.py                    ← EfficientNet-B4 adaptado
│   ├── train.py                    ← Loop de treino em 3 fases
│   ├── evaluate.py                 ← Métricas clínicas completas
│   ├── threshold_analysis.py       ← Análise e gráfico de threshold
│   ├── load_dicom.py               ← Utilitário de leitura DICOM
│   ├── load_labels.py              ← Inspeção dos CSVs clínicos
│   ├── check_val_nan.py            ← Verificação de NaN no val set
│   ├── test_cropped.py             ← Teste do dataset com ROI recortada
│   └── test_dicom.py               ← Visualização de arquivos DICOM
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalação

**Pré-requisitos:** Python 3.10+ e CUDA (recomendado)

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/OncoVision.git
cd OncoVision

# 2. Crie e ative o ambiente virtual
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 3. Instale as dependências
pip install -r requirements.txt
```

> Se sua GPU usa uma versão diferente de CUDA, consulte [pytorch.org/get-started](https://pytorch.org/get-started/locally/) para instalar o torch correto antes de rodar o passo 3.

---

## 📦 Dataset

O projeto usa o **[CBIS-DDSM](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)** (Curated Breast Imaging Subset of DDSM), um dataset de mamografias digitalizadas amplamente usado em pesquisa.

Após o download, organize assim:

```
data/raw/
├── cbis_ddsm/
│   └── CBIS-DDSM/          ← pastas dos pacientes com arquivos .dcm
├── mass_case_description_train_set.csv
└── calc_case_description_train_set.csv
```

Para inspecionar os CSVs e verificar o dataset:

```bash
python src/load_labels.py
python src/test_dicom.py
```

---

## 🚀 Uso

### Treino

```bash
python src/train.py
```

O treino roda automaticamente em **3 fases progressivas**:

| Fase | Epochs | Resolução | Modo | LR |
|------|--------|-----------|------|----|
| 1 | 5 | 224 × 224 | Só cabeça (backbone frozen) | 3e-4 |
| 2 | 10 | 320 × 320 | Parcial (blocks[-2:] + cabeça) | 3e-5 |
| 3 | 30 | 384 × 384 | Full fine-tuning | 1e-5 |

Os melhores pesos são salvos automaticamente em `models/best_model.pth`.

---

### Avaliação

```bash
python src/evaluate.py
```

Gera métricas clínicas completas e salva a curva ROC em `outputs/roc_curve.png`:

```
=============================================
  RESULTADOS CLÍNICOS — OncoVision
=============================================
  Accuracy    : 0.8123  (81.2%)
  AUC-ROC     : 0.8741
  Sensitivity : 0.9210  (92.1%)  ← detecção de câncer
  Specificity : 0.7034  (70.3%)
  Precision   : 0.7856  (78.6%)
  F1-Score    : 0.8479
=============================================
```

---

### Análise de Threshold

```bash
python src/threshold_analysis.py
```

Gera um gráfico publicável mostrando como sensitivity e specificity variam pelo threshold de decisão, e indica o **ponto ótimo clínico** (sensitivity ≥ 92% com maior specificity possível).

---

## 🧠 Arquitetura

```
Input: [B, 1, H, W]  ← grayscale, sem duplicar canais
           │
    EfficientNet-B4
    (pretrained, in_chans=1)
           │
    AdaptiveAvgPool2d(1)
           │
        Flatten
           │
       Dropout(0.3)
           │
       Linear(1792 → 1)
           │
Output: logit escalar [B, 1]
```

A saída é um **logit** — aplique `sigmoid` para obter a probabilidade de malignidade.

---

## 📊 Outputs Gerados

| Arquivo | Descrição |
|---|---|
| `outputs/roc_curve.png` | Curva ROC com AUC anotado |
| `outputs/threshold_analysis.png` | Sensitivity / Specificity / F1 × Threshold + ROC |
| `models/best_model.pth` | Pesos do melhor checkpoint |
| `models/last_model.pth` | Pesos do último epoch |

---

## 🔧 Configuração Rápida

As principais constantes de treino estão no topo de `src/train.py`:

```python
BATCH_SIZE = 8       # reduza para 4 se tiver pouca VRAM
THRESHOLD  = 0.55    # threshold de decisão clínico
USE_CALC   = True    # True = massa + calcificação | False = só massa
PATIENCE   = 7       # early stopping
```

---

## 📚 Referências

- **CBIS-DDSM Dataset** — [Kaggle](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) | [Artigo original (Lee et al., 2017)](https://www.nature.com/articles/sdata2017177)
- **EfficientNet** — [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- **timm** — [pytorch-image-models](https://github.com/huggingface/pytorch-image-models)

---

<div align="center">

Feito com 🔬 para pesquisa em saúde — **não é um dispositivo médico certificado**

</div>
