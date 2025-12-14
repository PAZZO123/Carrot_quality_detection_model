#integration.py
import os
import pandas as pd

manifest = pd.read_csv("clean_data/pre_manifest.csv")
labels = pd.read_csv("labels.csv")
manifest["filename"] = manifest["orig_path"].apply(lambda x: os.path.basename(x))
df = manifest.merge(labels, on="filename", how="left")

os.makedirs("integrated_data", exist_ok=True)
df.to_csv("integrated_data/integrated_manifest.csv", index=False)

print(f"Integration complete! {len(df)} records saved to integrated_data/integrated_manifest.csv")
