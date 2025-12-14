import tkinter as tk

import torch
import torch.nn as nn
from PIL import Image, ImageTk
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ===========================
# 1. DEVICE SETUP
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# 2. MODEL DEFINITION
# ===========================
model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(512, 2)  # 2 classes: GOOD / BAD
)
model.load_state_dict(torch.load("models/carrot_mobilenet_final.pth", map_location=device))
model = model.to(device)
model.eval()

# ===========================
# 3. IMAGE TRANSFORM
# ===========================
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

classes = ["BAD", "GOOD"]

# ===========================
# 4. LOAD TEST DATA
# ===========================
test_data = datasets.ImageFolder("carrot_data1/test")
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# ===========================
# 5. CREATE TKINTER WINDOW
# ===========================
root = tk.Tk()
root.title("Carrot Classifier – Test Images")

canvas = tk.Canvas(root)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# ===========================
# 6. FUNCTION TO PREDICT IMAGE
# ===========================
def predict_image(image_path, label_widget):
    img = Image.open(image_path).convert("RGB")
    input_img = val_test_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_img)
        _, pred = torch.max(output, 1)
    pred_class = classes[pred.item()]
    label_widget.config(text=f"Prediction: {pred_class}")

# ===========================
# 7. DISPLAY IMAGES
# ===========================
image_refs = []  # Keep references to avoid garbage collection

for path, label in test_data.imgs:  # test_data.imgs contains (path, class_index)
    img = Image.open(path).convert("RGB")
    img_resized = img.resize((150, 150))
    img_tk = ImageTk.PhotoImage(img_resized)

    frame = tk.Frame(scrollable_frame, pady=5)
    label_img = tk.Label(frame, image=img_tk)
    label_img.pack(side="left", padx=10)

    label_text = tk.Label(frame, text="Prediction: Not predicted", font=("Arial", 12))
    label_text.pack(side="left", padx=10)

    # Bind click event to predict this image
    label_img.bind("<Button-1>", lambda e, p=path, l=label_text: predict_image(p, l))

    frame.pack()
    image_refs.append(img_tk)

root.mainloop()
