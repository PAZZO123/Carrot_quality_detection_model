# ======================================================
# 1. IMPORT LIBRARIES
# ======================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    resnet50, ResNet50_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
    vgg16, VGG16_Weights
)
from torch.utils.data import DataLoader
import os

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# ======================================================
# 2. DATA TRANSFORMS (STRONGER AUGMENTATION)
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
# 4. SELECT MODEL (EfficientNet / ResNet / MobileNet / VGG)
# ======================================================
MODEL_NAME = "efficientnet"   # change: efficientnet | resnet | mobilenet | vgg

num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(name):
    if name == "efficientnet":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Partial unfreeze
        for n, p in model.named_parameters():
            p.requires_grad = ("features.6" in n or "features.7" in n)
        # Replace classifier
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280, num_classes)
        )
        return model

    elif name == "resnet":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        for n, p in model.named_parameters():
            p.requires_grad = ("layer4" in n)  # last block only
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif name == "mobilenet":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        for n, p in model.named_parameters():
            p.requires_grad = ("features.17" in n or "features.16" in n)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    elif name == "vgg":
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        for n, p in model.named_parameters():
            p.requires_grad = ("features.28" in n or "features.29" in n)
        model.classifier[6] = nn.Linear(4096, num_classes)
        return model

    else:
        raise ValueError("Unknown model name!")

model = load_model(MODEL_NAME)
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

    # Validation
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
        torch.save(model.state_dict(), "models/Efficiency_best_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⛔ Early stopping triggered.")
            break


# ======================================================
# 6. TEST MODEL
# ======================================================
model.load_state_dict(torch.load("models/Efficiency_best_model.pth"))
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
# 7. CONFUSION MATRIX + REPORT
# ======================================================
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=train_data.classes,
            yticklabels=train_data.classes)
plt.title(f"Confusion Matrix - {MODEL_NAME}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(f"figs/confusion_matrix_{MODEL_NAME}.png")
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=train_data.classes))


# ======================================================
# 8. SAVE FINAL MODEL
# ======================================================
torch.save(model.state_dict(), f"models/{MODEL_NAME}_carrot_final.pth")
print(f"💾 Final model saved as '{MODEL_NAME}_carrot_final.pth'")
