#cleaning.py
import hashlib, os, pandas as pd
from PIL import Image, UnidentifiedImageError
from pathlib import Path

root = Path("datasets")
out_root = Path("clean_data")
out_root.mkdir(exist_ok=True)
classes = ["GOOD","BAD"]

def sha256(path):
    import hashlib
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda: f.read(8192), b''):
            h.update(b)
    return h.hexdigest()

rows=[]
corrupted=[]
hash_map={}
duplicates=[]

for label in classes:
    for p in (root/label).glob("*.*"):
        try:
            with Image.open(p) as im:
                im.verify()
            h = sha256(p)
            if h in hash_map:
                duplicates.append((str(p), str(hash_map[h])))
            else:
                hash_map[h]=p
                rows.append({"orig_path": str(p), "label": label, "sha256": h})
        except (UnidentifiedImageError, OSError) as e:
            corrupted.append({"path": str(p), "err": str(e)})

pd.DataFrame(rows).to_csv("clean_data/pre_manifest.csv", index=False)
pd.DataFrame(corrupted).to_csv("clean_data/corrupted.csv", index=False)
pd.DataFrame(duplicates, columns=["file","duplicate_of"]).to_csv("clean_data/duplicates.csv", index=False)
print("Done. corrupted:", len(corrupted), "duplicates:", len(duplicates))
