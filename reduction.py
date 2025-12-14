import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm
from pathlib import Path
import os

df = pd.read_csv("integrated_data/integrated_manifest.csv")
paths = df["orig_path"].tolist()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18(weights="IMAGENET1K_V1")
model = torch.nn.Sequential(*list(model.children())[:-1]).to(device).eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
features = []
valid_paths = []
for p in tqdm(paths, desc="Extracting features"):
    try:
        img = Image.open(p).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(x).cpu().numpy().reshape(-1)
        features.append(feat)
        valid_paths.append(p)
    except Exception as e:
        print(f" Skipped {p}: {e}")

features = np.vstack(features)
pca = PCA(n_components=50)
reduced = pca.fit_transform(features)
print("Explained variance ratio sum:", round(pca.explained_variance_ratio_.sum(), 3))
os.makedirs("reduced_data", exist_ok=True)
reduced_df = pd.DataFrame(reduced)
reduced_df["filepath"] = valid_paths
reduced_df.to_csv("reduced_data/embeddings_pca.csv", index=False)

print("Saved reduced embeddings to reduced_data/embeddings_pca.csv")
