# discretize.py
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def compute_mean_intensity(df_index_path="outputs/cleaned_image_index.csv"):
    df = pd.read_csv(df_index_path)
    mean_vals = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        path = row['path']
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_vals.append(gray.mean())
    df['mean_intensity'] = mean_vals
    df.to_csv("outputs/with_intensity.csv", index=False)
    return df

def discretize(df_path="outputs/with_intensity.csv"):
    df = pd.read_csv(df_path)
    bins = [0, 80, 140, 255]  # example thresholds
    labels = ['Low', 'Medium', 'High']
    df['brightness_bin'] = pd.cut(df['mean_intensity'], bins=bins, labels=labels, include_lowest=True)
    df.to_csv("outputs/with_bins.csv", index=False)
    # plot distribution
    plt.figure(figsize=(6,4))
    df['brightness_bin'].value_counts().reindex(labels).plot(kind='bar')
    plt.title("Brightness bins count")
    plt.savefig("outputs/figures/brightness_bins.png")
    plt.close()
    return df

if __name__ == "__main__":
    df = compute_mean_intensity()
    df = discretize()
    print(df[['path','class','mean_intensity','brightness_bin']].head())
