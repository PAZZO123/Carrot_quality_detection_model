# ==========================================
#  CREATE TRAIN / VAL / TEST SPLIT
# ==========================================

import os
import shutil

import splitfolders

# 1️⃣ Delete old carrot_data folder if it exists
if os.path.exists("carrot_data1"):
    shutil.rmtree("carrot_data1")
    print("Old carrot_data folder removed.")

# 2️⃣ Split the dataset into train / val / test
# Ratio: 70% train, 15% val, 15% test
splitfolders.ratio(
    input="clean_data/augmented",   # Original dataset folder
    output="carrot_data1",           # New split folder
    seed=42,
    ratio=(0.7, 0.15, 0.15)
)

# 3️⃣ Verify folder structure
for split in ["train", "val", "test"]:
    split_path = os.path.join("carrot_data1", split)
    if os.path.exists(split_path):
        good_count = len(os.listdir(os.path.join(split_path, "GOOD")))
        bad_count = len(os.listdir(os.path.join(split_path, "BAD")))
        print(f"{split.upper()}: GOOD={good_count}, BAD={bad_count}")
    else:
        print(f"{split} folder not found!")

print("✅ Train / Val / Test split created successfully!")
