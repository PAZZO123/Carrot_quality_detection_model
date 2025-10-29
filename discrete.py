import numpy as np
import pandas as pd
from PIL import Image

# Use the correct CSV file from the previous step
df = pd.read_csv("clean_data/manifest_labelled.csv")

# Function to compute mean green channel
def mean_green(path):
    im = Image.open(path).convert('RGB')
    arr = np.array(im)
    return arr[:, :, 1].mean()  # Green channel

# Apply function
df['mean_green'] = df['orig_path'].apply(mean_green)

# Discretize into 3 bins: low, medium, high
df['green_bin'] = pd.qcut(df['mean_green'], q=3, labels=['low', 'med', 'high'])

# Save the new CSV
df[['orig_path','label','mean_green','green_bin']].to_csv("clean_data/with_green_bins.csv", index=False)

print("✅ Discretization complete! File saved as clean_data/with_green_bins.csv")
