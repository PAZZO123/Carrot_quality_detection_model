import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_curve)
from torchvision import models
from torchvision.models import (DenseNet121_Weights, EfficientNet_B0_Weights,
                                MobileNet_V2_Weights, ResNet18_Weights,
                                VGG16_Weights, densenet121, efficientnet_b0,
                                mobilenet_v2, resnet18, vgg16)

# -----------------------------------------------------
# CREATE FIGS DIRECTORY
# -----------------------------------------------------
os.makedirs("figs", exist_ok=True)

from torch.utils.data import DataLoader
# -----------------------------------------------------
# LOAD TEST DATA
# -----------------------------------------------------
from torchvision import datasets, transforms

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

test_data = datasets.ImageFolder("carrot_data1/test", transform=test_transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------
# MODEL EVALUATION FUNCTION
# -----------------------------------------------------
def evaluate_model(model, loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds) * 100,
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
        "cm": confusion_matrix(all_labels, all_preds)
    }
    return metrics

# -----------------------------------------------------
# MODEL LOADERS
# -----------------------------------------------------
def load_densenet():
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(1024, 2))
    model.load_state_dict(torch.load("models/densenet_best_final_model.pth"))
    return model.to(device)

def load_mobilenet():
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280, 2))
    model.load_state_dict(torch.load("models/carrot_mobilenet_best.pth"))
    return model.to(device)

def load_efficientnet():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280, 2))
    model.load_state_dict(torch.load("models/Efficiency_best_model.pth"))
    return model.to(device)

def load_vgg():
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(4096, 2)
    model.load_state_dict(torch.load("models/best_vgg16.pth"))
    return model.to(device)

def load_resnet():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 2))
    model.load_state_dict(torch.load("models/carrot_classifier_best2.pth", map_location=device))
    return model.to(device)

# -----------------------------------------------------
# MODEL REGISTRY
# -----------------------------------------------------
model_loaders = {
    "DenseNet121": load_densenet,
    "MobileNetV2": load_mobilenet,
    "EfficientNet-B0": load_efficientnet,
    "VGG16": load_vgg,
    "ResNet18": load_resnet
}

# -----------------------------------------------------
# RUN EVALUATION
# -----------------------------------------------------
results = {}
models_list = list(model_loaders.keys())

for name, loader_fn in model_loaders.items():
    print(f"\n🔍 Evaluating {name}...")
    model = loader_fn()
    metrics = evaluate_model(model, test_loader)
    results[name] = metrics

    # ---- Save individual confusion matrix ----
    cm = metrics["cm"]
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=test_data.classes,
                yticklabels=test_data.classes)
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f"figs/cm_{name}.png")
    plt.close()

# -----------------------------------------------------
# SUMMARY METRICS
# -----------------------------------------------------
accuracies = [results[m]["accuracy"] for m in models_list]
precisions = [results[m]["precision"] for m in models_list]
recalls = [results[m]["recall"] for m in models_list]
f1s = [results[m]["f1"] for m in models_list]

print("\n====================== MODEL COMPARISON ======================")
print("{:<20} {:<10} {:<10} {:<10} {:<10}".format("Model", "Acc(%)", "Prec", "Recall", "F1"))
print("--------------------------------------------------------------")
for name, m in results.items():
    print("{:<20} {:<10.2f} {:<10.3f} {:<10.3f} {:<10.3f}".format(
        name, m["accuracy"], m["precision"], m["recall"], m["f1"]
    ))

# -----------------------------------------------------
# MODEL SIZE (MB) FOR COMPARISON
# -----------------------------------------------------
model_sizes = {
    "DenseNet121": 33.2,
    "MobileNetV2": 14.0,
    "EfficientNet-B0": 20.0,
    "VGG16": 528.0,
    "ResNet18": 44.7
}
sizes = [model_sizes[m] for m in models_list]

# -----------------------------------------------------
# MULTI-PANEL FIGURE FOR PRESENTATION
# -----------------------------------------------------
# -----------------------------------------------------
# ALL CONFUSION MATRICES IN ONE FIGURE
# -----------------------------------------------------
num_models = len(models_list)
cols = 3
rows = (num_models + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(18, 10))
axes = axes.flatten()

for idx, model_name in enumerate(models_list):
    cm = results[model_name]["cm"]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=test_data.classes,
        yticklabels=test_data.classes,
        ax=axes[idx]
    )
    axes[idx].set_title(model_name)
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Actual")

# Hide unused subplots if any
for i in range(idx + 1, len(axes)):
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("figs/all_confusion_matrices.png")
plt.close()

print("✅ All confusion matrices saved in one figure: figs/all_confusion_matrices.png")
# -----------------------------------------------------
# 4.8 ERROR ANALYSIS COMPARISON (FP & FN)
# -----------------------------------------------------
fps, fns = [], []

for model_name in models_list:
    cm = results[model_name]["cm"]
    tn, fp, fn, tp = cm.ravel()
    fps.append(fp)
    fns.append(fn)

x = np.arange(len(models_list))
width = 0.35

plt.figure(figsize=(14,6))
plt.bar(x - width/2, fps, width, label='False Positives (Bad → Good)')
plt.bar(x + width/2, fns, width, label='False Negatives (Good → Bad)')

plt.xticks(x, models_list, rotation=20)
plt.ylabel("Number of Errors")
plt.title("Comparison of Predicted Error Analysis for All Models")
plt.legend()
plt.grid(axis='y')

plt.tight_layout()
plt.savefig("figs/error_analysis_fp_fn_1.png")
plt.close()

print("✅ Figure 4.8 saved as figs/Error_analysis_fp_fn.png")


# -----------------------------------------------------
# ROC & Precision-Recall Curves (Separate Files)
# -----------------------------------------------------
for curve_type in ["ROC", "PR"]:
    plt.figure(figsize=(10,6))
    for name, loader_fn in model_loaders.items():
        model = loader_fn()
        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = nn.functional.softmax(outputs, dim=1)[:,1]
                all_scores.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())

        if curve_type=="ROC":
            fpr, tpr, _ = roc_curve(all_labels, all_scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={roc_auc:.2f})')
            plt.plot([0,1],[0,1], color='gray', linestyle='--')
            plt.title("ROC Curves")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
        else:
            precision, recall, _ = precision_recall_curve(all_labels, all_scores)
            ap = average_precision_score(all_labels, all_scores)
            plt.plot(recall, precision, lw=2, label=f'{name} (AP={ap:.2f})')
            plt.title("Precision–Recall Curves")
            plt.xlabel("Recall")
            plt.ylabel("Precision")

    plt.legend()
    plt.grid(True)
    plt.savefig(f"figs/{curve_type.lower()}_curves_all.png")
    plt.close()

print("\n🎉 All plots (bar plots, ROC, PR, model size) saved in figs/ folder successfully!")
