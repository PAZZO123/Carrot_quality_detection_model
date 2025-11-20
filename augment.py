import random
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
import os

augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop((224,224), scale=(0.8,1.0))
])

out_aug = Path("clean_data/augmented")
out_aug.mkdir(exist_ok=True)
n_aug_per_image = 20
df = pd.read_csv("clean_data/manifest_labelled.csv")  # <- use your actual file

for idx, row in df.iterrows():
    p = row['orig_path']  # <- match your column
    label = row['label']
    im = Image.open(p).convert('RGB')
    for i in range(n_aug_per_image):
        aug_img = augment(im)
        fname = f"{Path(p).stem}_aug{i}.jpg"
        save_dir = out_aug / label
        save_dir.mkdir(parents=True, exist_ok=True)
        aug_img.save(save_dir / fname, quality=90)

print(" Augmentation completed!")
