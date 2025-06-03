import os
from ultralytics import YOLO
from pathlib import Path
import cv2
import json

# Configuration
model_path = 'D:/Projects/LungCancerDetection/runs/detect/yolov8s/lc25000_3epochs/weights/last.pt'  # Or use best.pt for 17-epoch
input_dir = 'D:/Projects/LungCancerDetection/test_images'  # Directory with new images
output_dir = 'D:/Projects/LungCancerDetection/predictions'
conf_threshold = 0.5
class_names = ['Cancer', 'Non-Cancer']

# Create output directory
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Load model
model = YOLO(model_path)

# Predict on new images
results_list = []
for img_name in os.listdir(input_dir):
    if not img_name.endswith(('.jpg', '.jpeg', '.png')):
        continue
    img_path = os.path.join(input_dir, img_name)
    results = model.predict(img_path, conf=conf_threshold)
    
    # Extract prediction
    pred_class = int(results[0].boxes.cls[0]) if results[0].boxes else 0
    pred_label = class_names[pred_class]
    conf = float(results[0].boxes.conf[0]) if results[0].boxes else 0.0
    
    # Save result
    results_list.append({
        'image': img_name,
        'prediction': pred_label,
        'confidence': conf
    })
    
    # Save image with label
    img = cv2.imread(img_path)
    cv2.putText(img, f'{pred_label} ({conf:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, img_name), img)

# Save predictions to JSON
with open(os.path.join(output_dir, 'predictions.json'), 'w') as f:
    json.dump(results_list, f, indent=4)

print(f"Predictions saved to {output_dir}")