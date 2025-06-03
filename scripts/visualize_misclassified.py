import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
model_path = 'D:/Projects/LungCancerDetection/runs/detect/yolov8s_lc25000/weights/best.pt'
test_img_dir = 'D:/Projects/LungCancerDetection/data/images/test'
test_label_dir = 'D:/Projects/LungCancerDetection/data/labels/test'

# Load model
model = YOLO(model_path)

# Class names
class_names = ['Cancer', 'Non-Cancer']

# Collect misclassified images
misclassified = []

for img_name in os.listdir(test_img_dir):
    if not img_name.endswith('.jpeg'):
        continue
    img_path = os.path.join(test_img_dir, img_name)
    results = model.predict(img_path, conf=0.5)
    
    # Predicted class
    pred_class = int(results[0].boxes.cls[0]) if results[0].boxes else 0
    
    # Ground truth
    label_path = os.path.join(test_label_dir, img_name.replace('.jpeg', '.txt'))
    with open(label_path, 'r') as f:
        true_class = int(f.read().strip().split()[0])
    
    if pred_class != true_class:
        misclassified.append({
            'img_path': img_path,
            'true_class': true_class,
            'pred_class': pred_class
        })

# Visualize misclassified images
if not misclassified:
    print("No misclassified images found!")
else:
    print(f"Found {len(misclassified)} misclassified images.")
    for item in misclassified[:5]:  # Show up to 5
        img = cv2.imread(item['img_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"True: {class_names[item['true_class']]}, Pred: {class_names[item['pred_class']]}")
        plt.axis('off')
        plt.show()

# Save misclassified paths to a file
if misclassified:
    with open('D:/Projects/LungCancerDetection/misclassified.txt', 'w') as f:
        for item in misclassified:
            f.write(f"{item['img_path']}, True: {class_names[item['true_class']]}, Pred: {class_names[item['pred_class']]}\n")