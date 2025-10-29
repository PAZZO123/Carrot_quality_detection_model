import pandas as pd

# Load cleaned manifest
df = pd.read_csv("clean_data/pre_manifest.csv")

# Encode labels: GOOD = 1, BAD = 0
df['label_enc'] = df['label'].map({'GOOD': 1, 'BAD': 0})

# Save the result
df.to_csv("clean_data/manifest_labelled.csv", index=False)

print("✅ Labels encoded. File saved as clean_data/manifest_labelled.csv")
