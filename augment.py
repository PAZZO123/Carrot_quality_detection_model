# augment.py
import os
import cv2
import numpy as np
from tqdm import tqdm
import random

SRC_DIR = "datasets"
AUG_DIR = "dataset_aug"
os.makedirs(AUG_DIR, exist_ok=True)
AUG_PER_IMAGE = 1  # number of augmented copies per image

def random_brightness(img, factor=0.15):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype('float32')
    hsv[:,:,2] = hsv[:,:,2] * (1 + (random.uniform(-factor, factor)))
    hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
    out = cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2BGR)
    return out

def augment_image(img):
    ops = []
    # flip
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    # rotate
    angle = random.uniform(-15, 15)
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    # brightness
    if random.random() < 0.7:
        img = random_brightness(img)
    return img

def run_augmentation():
    for cls in os.listdir(SRC_DIR):
        src_folder = os.path.join(SRC_DIR, cls)
        dest_folder = os.path.join(AUG_DIR, cls)
        os.makedirs(dest_folder, exist_ok=True)
        for fname in tqdm(os.listdir(src_folder), desc=f"Aug {cls}"):
            src_path = os.path.join(src_folder, fname)
            img = cv2.imread(src_path)
            if img is None:
                continue
            # copy original to augmented folder
            cv2.imwrite(os.path.join(dest_folder, fname), img)
            # produce augmentations
            base, ext = os.path.splitext(fname)
            for i in range(AUG_PER_IMAGE):
                aug = augment_image(img)
                out_name = f"{base}_aug{i}{ext}"
                cv2.imwrite(os.path.join(dest_folder, out_name), aug)

if __name__ == "__main__":
    run_augmentation()
    print("Augmentation complete. Check dataset_aug folder.")
