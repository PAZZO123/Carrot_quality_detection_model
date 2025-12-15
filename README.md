# CarrotHub — AI-Driven Carrot Quality Classification

CarrotHub is a deep learning project that classifies carrot images into **GOOD** vs **BAD** quality using transfer learning.  
The project includes dataset cleaning, preprocessing, augmentation, training, and evaluation across multiple CNN backbones (e.g., EfficientNet, ResNet, VGG).

## Links (Dataset & Models)

- **Dataset + Trained Models (Google Drive):**  
 https://drive.google.com/drive/folders/1QLs4Q454vdGFSmO4XWwj1sCI5reT9Dh5?usp=sharing

- **(Optional) Kaggle source dataset:**  
  Add your Kaggle dataset URL here (the original dataset page you downloaded from).

## Project Highlights

- Binary classification: GOOD vs BAD carrots
- Transfer learning with pretrained ImageNet backbones via `torchvision`
- Strong performance with CPU-friendly architectures (e.g., EfficientNet-B0 / MobileNetV2)
- Designed for practical pack-house quality inspection use cases

## Repository Structure (suggested)


## Setup

### 1) Create environment


### 2) Install dependencies


If you don’t have a `requirements.txt` yet, typical dependencies are:
- `torch`, `torchvision`
- `numpy`, `pandas`
- `opencv-python` or `Pillow`
- `scikit-learn`
- `matplotlib`

## Data Preparation

### Option A — Use provided Drive package (recommended)

1. Open the Drive link:
   https://drive.google.com/drive/folders/1rcSUAdxkxFh4xgv_ctQaBYffNu1_Nihb?usp=sharing
2. Download:
   - the dataset folder(s)
   - the trained model weights/checkpoints
3. Place them into your repo, for example:
   - dataset → `data/`
   - weights → `models/checkpoints/`

### Option B — Download from Kaggle (if you prefer)

1. Download the dataset from Kaggle (add your Kaggle link above).
2. Extract into `data/raw/`
3. Run your cleaning/preprocessing pipeline (see next section).

## Training

> Update these commands to match your script names/arguments.


## Evaluation


Common metrics:
- Accuracy
- Precision / Recall
- F1-score
- Confusion matrix

## Inference (Predict on new image)


Output: `GOOD` or `BAD` (optionally with confidence score).

## Models Used

This project compares multiple CNN backbones using transfer learning (pretrained ImageNet weights via `torchvision`), such as:
- EfficientNet-B0
- MobileNetV2
- ResNet18
- VGG16
- DenseNet121

## Notes on Dataset Cleaning

The raw dataset required cleaning due to duplicates/near-duplicates, corrupted samples, mislabeled images, and low-quality images.  
After rigorous filtering, a smaller verified-clean subset was used, and augmentation was applied to expand training data.



## Authors

- MBABAZI Patrick Straton
- IRADUKUNDA Toussaint

University of Rwanda — College of Science and Technology




