# Lung Cancer Detection with YOLOv8

**Date:** June 2, 2025  
**Author:** [Your Name]

---

## Overview

This project uses the YOLOv8 model to detect lung cancer from histopathological images in the LC25000 dataset. The dataset contains 25,000 images (17,500 training, 3,750 validation, 3,750 test), split into two classes: Cancer and Non-Cancer. The primary model was trained for 2 epochs using YOLOv8s, achieving ~99.81% accuracy on the test set. A previous 17-epoch model achieved 100% accuracy and is included for reference.

---

## Dataset

- **Source:** LC25000 (Lung and Colon Cancer Histopathological Images)  
- **Classes:**  
  - Cancer (2,250 test images)  
  - Non-Cancer (1,500 test images)  
- **Image Size:** 640×640 pixels  
- **Splits:**  
  - Training: 17,500 images  
  - Validation: 3,750 images  
  - Test: 3,750 images  

---

## Model and Results

### 2-Epoch Model (YOLOv8s)

#### Training
- **Epochs:** 2  
- **Batch Size:** 4  
- **Image Size:** 640×640  
- **Model:** YOLOv8s (11.1M parameters)  

#### Test Set Metrics (3,750 images)
- **Accuracy:** 99.81%  
- **Precision:** 99.80%  
- **Recall:** 99.73%  
- **F1-Score:** 99.77%  
- **Misclassifications:** 7 images (~0.19% error rate)  

#### Outputs
- **Evaluation Report:** `D:\Projects\LungCancerDetection\evaluation_report_2epochs.txt`  
- **Metrics:** `D:\Projects\LungCancerDetection\metrics_2epochs.txt`  
- **Confusion Matrix:** `D:\Projects\LungCancerDetection\confusion_matrix_2epochs.png`  
- **Misclassified Images:** `D:\Projects\LungCancerDetection\misclassified_2epochs.txt`  
- **Model Weights:** `D:\Projects\LungCancerDetection\runs\detect\yolov8s_lc25000_2epochs\weights\last.pt`  

---

### 17-Epoch Model (YOLOv8s)

> **Note:** Trained for 17 epochs, achieving 100% accuracy, but considered over-optimized. Backed up for reference.

- **Training Metrics Plot:**  
  `D:\Projects\LungCancerDetection\backup_17epochs\runs\detect\yolov8s_lc25000\weights\best.pt`

---

## Predictions on New Images

- **Script:** `D:\Projects\LungCancerDetection\scripts\predict.py`  
- **Input Directory:** `D:\Projects\LungCancerDetection\test_images`  
- **Output:**  
  - Labeled images  
  - JSON file in `D:\Projects\LungCancerDetection\predictions`  

### Usage

```bash
cd D:\Projects\LungCancerDetection\scripts
python predict.py


Evaluation Report: D:\Projects\LungCancerDetection\evaluation_report_2epochs.txt

Metrics: D:\Projects\LungCancerDetection\metrics_2epochs.txt

Confusion Matrix: D:\Projects\LungCancerDetection\confusion_matrix_2epochs.png

Misclassified Images: D:\Projects\LungCancerDetection\misclassified_2epochs.txt

Predictions:

D:\Projects\LungCancerDetection\predictions\predictions.json and labeled images

Training Log (optional):
D:\Projects\LungCancerDetection\runs\detect\yolov8s_lc25000_2epochs\results.csv