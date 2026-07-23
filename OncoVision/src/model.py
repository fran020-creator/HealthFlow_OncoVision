"""
model.py — OncoVision Neural Network
EfficientNet-B4 fine-tuned para classificação binária de mamografias.
Entrada: imagem grayscale [B, 1, H, W]
Saída  : logit escalar   [B, 1]
"""

import torch
import torch.nn as nn
import timm


class OncoVisionNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()

        # EfficientNet-B4 — melhor que ResNet18 em imagens médicas
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,          # remove cabeça original
            in_chans=1,             # grayscale direto — sem duplicar canais
        )

        # cabeça de classificação binária
        in_features = self.backbone.num_features   # 1792 no B4
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        return self.head(features)


if __name__ == "__main__":
    model = OncoVisionNet()
    dummy = torch.randn(2, 1, 224, 224)
    out   = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")
    print("✅ model.py OK — EfficientNet-B4")
