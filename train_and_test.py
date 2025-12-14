# ===============================
# 🚀 CARROT CLASSIFIER – TRAINING WITH TEST SET
# ===============================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import splitfolders
import os

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
# ======================================================
# 1. DATA TRANSFORMS – STRONG TO REDUCE OVERFITTING
# ======================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ======================================================
# 2. DATA SPLITTING – TRAIN / VAL / TEST
# ======================================================
if not os.path.exists("carrot_data1/train"):
    splitfolders.ratio(
        "clean_data/augmented",
        output="carrot_data",
        seed=42,
        ratio=(0.7, 0.15, 0.15),  # Train / Val / Test
        group_prefix=None
    )
# ======================================================
# ✅ FORCE CREATE DATASET FOLDERS IF MISSING
# ======================================================

base_dir = "carrot_data1"

required_folders = [
    f"{base_dir}/train",
    f"{base_dir}/val",
    f"{base_dir}/test"
]

for folder in required_folders:
    os.makedirs(folder, exist_ok=True)

# If dataset not yet split, perform split
if not os.listdir(f"{base_dir}/train"):
    splitfolders.ratio(
        "clean_data/augmented",
        output=base_dir,
        seed=42,
        ratio=(0.7, 0.15, 0.15)
    )
# ======================================================
# 3. LOAD DATA
# ======================================================
train_data = datasets.ImageFolder("carrot_data1/train", transform=train_transform)
val_data   = datasets.ImageFolder("carrot_data1/val",   transform=val_test_transform)
test_data  = datasets.ImageFolder("carrot_data1/test",  transform=val_test_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=32)
test_loader  = DataLoader(test_data,  batch_size=32)

# ======================================================
# 4. BUILD MODEL – FREEZE LAYERS + DROPOUT
# ======================================================
model = models.resnet18(weights="IMAGENET1K_V1")

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Train only last layer
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(512, 2)  # 2 classes = GOOD / BAD
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)

# ======================================================
# 5. TRAINING WITH EARLY STOPPING
# ======================================================
epochs = 12
best_loss = float('inf')
patience = 3
trigger_times = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # ===== VALIDATION LOSS =====
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

    # ===== EARLY STOPPING =====
    if val_loss < best_loss:
        best_loss = val_loss
        trigger_times = 0
        torch.save(model.state_dict(), "models/carrot_classifier_best2.pth")
        print("📌 Model improved → Saved!")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("⛔ EARLY STOPPING TRIGGERED!")
            break

# ======================================================
# 6. FINAL VALIDATION ACCURACY
# ======================================================
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_accuracy = 100 * correct / total
print(f"\n🎯 VALIDATION ACCURACY: {val_accuracy:.2f}%")

# ======================================================
# 7. TEST METRICS: Precision, Recall, F1-score, Confusion Matrix ON UNSEEN DATA
# ======================================================
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Accuracy
test_accuracy = (all_preds == all_labels).mean() * 100

# Precision, Recall, F1
precision = precision_score(all_labels, all_preds, average='binary')
recall    = recall_score(all_labels, all_preds, average='binary')
f1        = f1_score(all_labels, all_preds, average='binary')

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

print("\n📊 TEST METRICS REPORT")
print("================================")
print(f"Accuracy  : {test_accuracy:.2f}%")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("\n📌 Confusion Matrix:")
print(cm)

# Detailed classification report (optional but powerful)
print("\n🔍 Full Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['BAD', 'GOOD']))

torch.save(model.state_dict(), "models/carrot_classifier_final.pth")
print("✅ Final model saved as carrot_classifier_final.pth")
