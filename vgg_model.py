# ======================================================
# 1. IMPORT LIBRARIES
# ======================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import vgg16, VGG16_Weights
from torch.utils.data import DataLoader
import os

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# ======================================================
# 2. DATA TRANSFORMS (STRONG AUGMENTATION)
# ======================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
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
# 3. LOAD DATASETS
# ======================================================
data_dir = "carrot_data1"

train_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
val_data = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_test_transform)
test_data = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=val_test_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"✔ Train samples: {len(train_data)}")
print(f"✔ Val samples  : {len(val_data)}")
print(f"✔ Test samples : {len(test_data)}")


# ======================================================
# 4. LOAD VGG16 MODEL (FINE-TUNED)
# ======================================================
model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

# 🔓 Unfreeze only LAST conv block (features.24–29)
for name, param in model.named_parameters():
    if "features.24" in name or "features.25" in name or \
       "features.26" in name or "features.27" in name or \
       "features.28" in name or "features.29" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Replace classifier for 2 classes
model.classifier[6] = nn.Linear(4096, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-4
)


# ======================================================
# 5. TRAINING + EARLY STOPPING
# ======================================================
num_epochs = 10
best_val_acc = 0
patience = 3
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    # VALIDATION
    model.eval()
    correct_val, total_val = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            correct_val += (outputs.argmax(1) == labels).sum().item()
            total_val += labels.size(0)

    val_acc = 100 * correct_val / total_val

    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {total_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # EARLY STOPPING
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "models/best_vgg16.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⛔ Early stopping triggered.")
            break


# ======================================================
# 6. TEST MODEL
# ======================================================
model.load_state_dict(torch.load("models/best_vgg16.pth"))
model.eval()

correct_test, total_test = 0, 0
y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

        correct_test += (preds == labels).sum().item()
        total_test += labels.size(0)

test_acc = 100 * correct_test / total_test
print(f"\n🔥 Test Accuracy: {test_acc:.2f}%")


# ======================================================
# 7. CONFUSION MATRIX + CLASSIFICATION REPORT
# ======================================================
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=train_data.classes,
            yticklabels=train_data.classes)
plt.title("Confusion Matrix - VGG16")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("figs/confusion_matrix_vgg16.png")
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=train_data.classes))


# ======================================================
# 8. SAVE FINAL MODEL
# ======================================================
torch.save(model.state_dict(), "models/vgg16_carrot_final.pth")
print("💾 Final model saved as 'vgg16_carrot_final.pth'")
