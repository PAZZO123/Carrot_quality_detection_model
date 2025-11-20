# ===============================
# 🚀 CARROT CLASSIFIER – TRAINING
# ===============================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import splitfolders
import os

# ======================================================
# 1. DATA TRANSFORMS – STRONG TO REDUCE OVERFITTING
# ======================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),              # NEW
    transforms.RandomRotation(30),                   # NEW
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),  # NEW
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ======================================================
# 2. DATA SPLITTING  (ONLY FIRST TIME)
# ======================================================
if not os.path.exists("carrot_data/train"):
    splitfolders.ratio(
        "clean_data/augmented",
        output="carrot_data",
        seed=42,
        ratio=(0.8, 0.2)
    )

# ======================================================
# 3. LOAD DATA
# ======================================================
train_data = datasets.ImageFolder("carrot_data/train", transform=train_transform)
val_data   = datasets.ImageFolder("carrot_data/val",   transform=val_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=32)


# ======================================================
# 4. BUILD MODEL – FREEZE LAYERS + DROPOUT
# ======================================================
model = models.resnet18(weights="IMAGENET1K_V1")

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Train only last layer
model.fc = nn.Sequential(
    nn.Dropout(0.5),    # Avoid overfitting
    nn.Linear(512, 2)   # 2 classes = GOOD / BAD
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
        torch.save(model.state_dict(), "carrot_classifier_best.pth")
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

accuracy = 100 * correct / total
print(f"\n🎯 FINAL VALIDATION ACCURACY: {accuracy:.2f}%")
print("✔ BEST MODEL SAVED AS carrot_classifier_best.pth")
