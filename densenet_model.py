# ======================================================
# 1. IMPORT LIBRARIES
# ======================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import densenet121, DenseNet121_Weights
from torch.utils.data import DataLoader
import os

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# ======================================================
# 2. DATA TRANSFORMS
# ======================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
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
# 4. MODEL SETUP (Freeze all except classifier)
# ======================================================
num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False  # FULL freeze

in_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, num_classes)
)

model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4, weight_decay=1e-4)


# ======================================================
# 5. TRAINING
# ======================================================
num_epochs = 7
best_val_acc = 0

os.makedirs("models", exist_ok=True)
save_path = "models/densenet_best_final_model.pth"

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
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

    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {running_loss/len(train_loader):.4f} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)


# ======================================================
# 6. TESTING
# ======================================================
model.load_state_dict(torch.load(save_path))
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
# 7. CONFUSION MATRIX
# ======================================================
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=train_data.classes, yticklabels=train_data.classes)
plt.title("Confusion Matrix")
plt.savefig("figs/confusion_matrix.png")
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=train_data.classes))
