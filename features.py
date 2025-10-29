# features.py
import numpy as np
import cv2
import pandas as pd
from skimage.feature import local_binary_pattern
from tqdm import tqdm
import os

IMG_SIZE = (128, 128)

def color_histogram(img, bins=(8, 8, 8)):
    # img in RGB format
    hist = cv2.calcHist([img], [0,1,2], None, bins, [0,256,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def lbp_histogram(img_gray, P=8, R=1, bins=256):
    lbp = local_binary_pattern(img_gray, P, R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, bins+1), range=(0, bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_features(df_index_path="outputs/cleaned_image_index.csv"):
    df = pd.read_csv(df_index_path)
    X = []
    y = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        path = row['path']
        cls = row['class']
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        # feature 1: color histogram
        ch = color_histogram(img)
        # feature 2: LBP (texture)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lbp = lbp_histogram(gray, P=8, R=1, bins=26)  # using 26 bins for uniform LBP
        feats = np.concatenate([ch, lbp])
        X.append(feats)
        y.append(0 if cls == "GOOD" else 1)
    X = np.array(X)
    y = np.array(y)
    np.save("outputs/X_features.npy", X)
    np.save("outputs/y_labels.npy", y)
    print("Saved features shape:", X.shape, y.shape)
    return X, y

if __name__ == "__main__":
    extract_features()
