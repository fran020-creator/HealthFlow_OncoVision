import sys
sys.path.insert(0, ".")
from dataset import CBISDDSMDataset
from torchvision import transforms
import torch

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

ds = CBISDDSMDataset(transform=transform, use_cropped=True)
print()

for i in range(5):
    img, label = ds[i]
    info = ds.get_info(i)
    pathology = info["pathology"]
    print(f"  [{i}] shape={tuple(img.shape)}  label={label}  ({pathology})")

print()
print("Dataset cropped OK!")
