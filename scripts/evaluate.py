import os
from ultralytics import YOLO
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Load trained model
model = YOLO('D:/Projects/LungCancerDetection/runs/train/yolov8s_lc25000/weights/best.pt')

# Test set paths
test_img_dir = 'D:/Projects/LungCancerDetection/data/images/test'
test_label_dir = 'D:/Projects/LungCancerDetection/data/labels/test'

# Initialize lists
y_true = []
y_pred = []

# Predict
for img_name in os.listdir(test_img_dir):
    img_path = os.path.join(test_img_dir, img_name)
    results = model.predict(img_path, conf=0.5)
    
    # Extract class prediction
    if results[0].boxes:
        pred_class = int(results[0].boxes.cls[0])
        y_pred.append(pred_class)
    else:
        y_pred.append(0)  # Default to cancer if no detection
    
    # Read ground truth
    label_path = os.path.join(test_label_dir, img_name.replace('.jpeg', '.txt'))
    with open(label_path, 'r') as f:
        true_class = int(f.read().split()[0])
        y_true.append(true_class)

# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
f1 = f1_score(y_true, y_pred, average='binary')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cancer', 'Non-Cancer'], yticklabels=['Cancer', 'Non-Cancer'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('D:/Projects/LungCancerDetection/confusion_matrix.png')
plt.show()