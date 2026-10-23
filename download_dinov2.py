"""
Download DINOv2-Small weights for offline Kaggle use.
Saves the state_dict to a .pth file that can be uploaded as a Kaggle dataset.
"""
import timm
import torch
from pathlib import Path

MODEL_NAME = 'vit_small_patch14_dinov2.lvd142m'
OUTPUT_DIR = Path('dinov2_weights')
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / 'vit_small_patch14_dinov2.pth'

print(f'Downloading {MODEL_NAME} via timm...')
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)

# Save just the state dict
torch.save(model.state_dict(), str(OUTPUT_PATH))

file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
num_params = sum(p.numel() for p in model.parameters())
feature_dim = model.num_features

print(f'Saved to: {OUTPUT_PATH}')
print(f'File size: {file_size_mb:.1f} MB')
print(f'Parameters: {num_params:,}')
print(f'Feature dim: {feature_dim}')
print(f'\nNext step: Upload the "{OUTPUT_DIR}" folder to Kaggle as a new dataset.')
