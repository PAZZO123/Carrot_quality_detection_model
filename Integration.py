import os
import pandas as pd

# 1. Load the cleaned manifest from Data Cleaning step
manifest = pd.read_csv("clean_data/pre_manifest.csv")

# 2. Load labels (GOOD/BAD)
labels = pd.read_csv("labels.csv")

# 3. Extract just the filename from manifest paths
manifest["filename"] = manifest["orig_path"].apply(lambda x: os.path.basename(x))

# 4. Merge manifest and labels based on filename
df = manifest.merge(labels, on="filename", how="left")

# 5. Save the integrated dataset
os.makedirs("integrated_data", exist_ok=True)
df.to_csv("integrated_data/integrated_manifest.csv", index=False)

print(f"✅ Integration complete! {len(df)} records saved to integrated_data/integrated_manifest.csv")
