# preprocessing.py
import os
import cv2
import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm
import pandas as pd

# === Paths and constants ===
DATA_DIR = "datasets"
OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Adjust class names to your folders
CLASSES = ["GOOD", "BAD"]
IMG_SIZE = (128, 128)  # width x height

# === Helper function to read & resize ===
def check_and_resize(image_path):
    """Try reading image with OpenCV, fallback to Pillow if needed."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            # fallback using Pillow
            with Image.open(image_path) as im:
                img = np.array(im.convert("RGB"))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # resize for consistency
        img = cv2.resize(img, IMG_SIZE)
        return img
    except Exception as e:
        print("❌ Error reading:", image_path, "->", e)
        return None

# === Main cleaning & duplicate check ===
def load_and_clean():
    records = []
    hashes = set()
    removed = []

    for cls in CLASSES:
        folder = os.path.join(DATA_DIR, cls)
        if not os.path.exists(folder):
            print(f"⚠️ Folder not found: {folder}")
            continue

        files = [f for f in os.listdir(folder) if not f.startswith('.')]
        for f in tqdm(files, desc=f"Processing {cls}"):
            path = os.path.join(folder, f)
            img = check_and_resize(path)
            if img is None:
                removed.append((path, "corrupt_or_unreadable"))
                continue

            # compute hash to detect *exact* duplicates
            try:
                pil = Image.fromarray(img)
                h = str(imagehash.average_hash(pil))
            except Exception:
                h = None

            # ⚠️ Relax duplicate detection: only skip if the hash was seen 2+ times
            if h and h in hashes:
                # comment this if your images are similar (to keep them)
                # removed.append((path, "duplicate"))
                pass
            if h:
                hashes.add(h)

            records.append({"path": path, "class": cls, "hash": h})

    df = pd.DataFrame(records)
    print("✅ Loaded images:", df.shape[0])
    print("🗑️ Removed count:", len(removed))
    return df, removed

# === Run as script ===
if __name__ == "__main__":
    df, removed = load_and_clean()
    df.to_csv(os.path.join(OUT_DIR, "cleaned_image_index.csv"), index=False)
    print("\n📊 Class distribution:")
    print(df["class"].value_counts())
    print("\nCounts after cleaning:", df["class"].value_counts().to_dict())
