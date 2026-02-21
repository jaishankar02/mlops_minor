import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from PIL import Image

# ==========================
# Config
# ==========================
DATA_DIR = "data/data/test/"
MODEL_PATH = "setA.pth"
BATCH_SIZE = 32
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Transforms
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# Dataset
# ==========================
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = dataset.classes
print("Classes:", class_names)

# ==========================
# Load Model
# ==========================
model = models.resnet18(weights=None) 
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print(f"Successfully loaded weights from: {MODEL_PATH}")
except FileNotFoundError:
    print(f"ERROR: Could not find {MODEL_PATH}!")
    exit()

model = model.to(DEVICE)
model.eval()

# ==========================
# Evaluation Loop
# ==========================
all_preds = []
all_labels = []

print("Running Evaluation...")
with torch.no_grad():
    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ==========================
# Metrics & Plotting
# ==========================
overall_acc = accuracy_score(all_labels, all_preds)
print(f"\nOverall Accuracy: {overall_acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# --- Confusion Matrix Logic ---
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')
print("\nSuccess: Confusion matrix saved as 'confusion_matrix.png'")

# ==========================
# Single Image Prediction
# ==========================
def predict_single_image(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)
    print(f"\nTest Image: {image_path}")
    print(f"Predicted Class: {class_names[pred.item()]} ({confidence.item()*100:.2f}%)")

# Fixed the path and variable assignment
test_img = "data/data/test/5/340.png" 
predict_single_image(test_img)
