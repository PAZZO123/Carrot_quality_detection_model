import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("clean_data/with_green_bins.csv")  # your main dataset
Path("figs").mkdir(exist_ok=True)

# Create 3x2 subplot
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
axes = axes.flatten()  # flatten for easier indexing

# -----------------------------
# 1️⃣ Class Distribution (GOOD/BAD)
# -----------------------------
df['label'].value_counts().plot(kind='bar', color=['green', 'red'], ax=axes[0])
axes[0].set_title("Class Distribution: GOOD vs BAD")
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Count")

# -----------------------------
# 2️⃣ Mean Green Histogram
# -----------------------------
df['mean_green'].hist(bins=30, color='lightgreen', edgecolor='black', ax=axes[1])
axes[1].set_title("Mean Green Intensity Distribution")
axes[1].set_xlabel("mean_green")
axes[1].set_ylabel("Count")

# -----------------------------
# 3️⃣ Mean Green Boxplot
# -----------------------------
axes[2].boxplot(df['mean_green'], patch_artist=True, boxprops=dict(facecolor='lightblue'))
axes[2].set_title("Mean Green Intensity Boxplot")
axes[2].set_ylabel("mean_green")

# -----------------------------
# 4️⃣ Green Bin Counts
# -----------------------------
df['green_bin'].value_counts().reindex(['low', 'med', 'high']).plot(kind='bar', color='orange', ax=axes[3])
axes[3].set_title("Green Bin Counts")
axes[3].set_xlabel("Green Bin")
axes[3].set_ylabel("Count")

# -----------------------------
# 5️⃣ Sample Images Grid
# -----------------------------
classes = ['GOOD', 'BAD']
sample_imgs = []
for c in classes:
    imgs = list((Path("datasets") / c).glob("*.*"))
    sample_imgs.extend(random.sample(imgs, min(2, len(imgs))))  # 2 per class for space

# Show the first image as example (overlay multiple images in one axes is not ideal)
if sample_imgs:
    im = Image.open(sample_imgs[0])
    axes[4].imshow(im)
axes[4].axis('off')
axes[4].set_title("Sample Images (GOOD/BAD)")

# -----------------------------
# 6️⃣ Duplicates Only
# -----------------------------
duplicates_file = Path("clean_data/duplicates.csv")

if duplicates_file.exists() and duplicates_file.stat().st_size > 0:
    duplicates = pd.read_csv(duplicates_file)
    axes[5].bar(['Duplicates'], [len(duplicates)], color='orange')
    axes[5].set_title("Data Cleaning: Duplicates")
    axes[5].set_ylabel("Count")
else:
    axes[5].text(0.5, 0.5, "No duplicates info", ha='center', va='center', fontsize=12)
    axes[5].set_axis_off()

plt.tight_layout()
plt.savefig("figs/all_plots_combined.png")
plt.show()

print("✅ All 6 plots combined in a single figure: figs/all_plots_combined.png")
