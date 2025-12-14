#overview.py
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# -------------------------------
# STEP 1: Define root and classes
# -------------------------------
root = Path("clean_data/augmented")   # Your dataset folder with GOOD/ and BAD/ subfolders
classes = ["GOOD", "BAD"]

# -------------------------------
# STEP 2: Count images per class
# -------------------------------
counts = {c: len(list((root / c).glob("*.*"))) for c in classes}
print("📊 Image counts per class:", counts)

# -------------------------------
# STEP 3: Show 4 sample images per class
# -------------------------------
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, c in enumerate(classes):
    imgs = list((root / c).glob("*.*"))[:4]
    for j, p in enumerate(imgs):
        try:
            axes[i, j].imshow(Image.open(p))
        except Exception as e:
            print(f"⚠️ Error loading {p}: {e}")
        axes[i, j].axis("off")
        if j == 0:
            axes[i, j].set_title(c)
plt.suptitle("Sample GOOD vs BAD Carrots")
plt.tight_layout()
plt.show()

# -------------------------------
# STEP 4: Generate labels.csv
# -------------------------------
print("\n🏷️ Generating labels.csv...")

data = []
for label in classes:
    folder = root / label
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            data.append([fname, label])

labels_df = pd.DataFrame(data, columns=["filename", "label"])
labels_df.to_csv("labels.csv", index=False)

print(f"✅ labels.csv created successfully with {len(labels_df)} records!")

# -------------------------------
# Summary
# -------------------------------
print("\n🎯 DONE:")
print(f"- Labels saved to labels.csv")
print(f"- Counts: {counts}")
