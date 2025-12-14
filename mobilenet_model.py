# ===============================
# 🚀 CARROT CLASSIFIER – TRAINING WITH MOBILE NET V2
# ===============================

import os

import numpy as np
import splitfolders
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

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
# 2. LOAD EXISTING DATASET
# ======================================================
base_dir = "carrot_data1"   # ⚠ Using your existing split

train_data = datasets.ImageFolder(f"{base_dir}/train", transform=train_transform)
val_data   = datasets.ImageFolder(f"{base_dir}/val",   transform=val_test_transform)
test_data  = datasets.ImageFolder(f"{base_dir}/test",  transform=val_test_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=32)
test_loader  = DataLoader(test_data,  batch_size=32)

print(f"✔ Train samples: {len(train_data)}")
print(f"✔ Val samples  : {len(val_data)}")
print(f"✔ Test samples : {len(test_data)}")

# ======================================================
# 3. BUILD MODEL – MOBILENETV2 + DROPOUT
# ======================================================
from torchvision.models import mobilenet_v2

model = mobilenet_v2(weights="IMAGENET1K_V1")   # Pretrained ImageNet model

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Train only classification head
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(1280, 2)    # 2 classes → BAD / GOOD
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)

# ======================================================
# 4. TRAINING WITH EARLY STOPPING
# ======================================================
epochs = 8
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

    # ===== VALIDATION =====
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

    if val_loss < best_loss:
        best_loss = val_loss
        trigger_times = 0
        torch.save(model.state_dict(), "models/carrot_mobilenet_best.pth")
        print("📌 Model improved → Saved!")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("⛔ EARLY STOPPING TRIGGERED!")
            break

# ======================================================
# 5. VALIDATION ACCURACY
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
# 6. TEST METRICS ON UNSEEN DATA
# ======================================================
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

test_accuracy = (all_preds == all_labels).mean() * 100
precision = precision_score(all_labels, all_preds, average='binary')
recall    = recall_score(all_labels, all_preds, average='binary')
f1        = f1_score(all_labels, all_preds, average='binary')
cm        = confusion_matrix(all_labels, all_preds)

print("\n📊 TEST METRICS REPORT")
print("================================")
print(f"Accuracy  : {test_accuracy:.2f}%")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print("\n📌 Confusion Matrix:")
print(cm)
print("\n🔍 Full Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['BAD', 'GOOD']))

torch.save(model.state_dict(), "models/carrot_mobilenet_final.pth")
print("✅ Final model saved as carrot_mobilenet_final.pth")

# ======================================================
# 7. EXPORT FOR MOBILE DEPLOYMENT (TorchScript)
# ======================================================
scripted = torch.jit.script(model)
scripted.save("models/carrot_mobilenet_mobile.pt")
print("📱 Mobile-ready model saved as carrot_mobilenet_mobile.pt")
