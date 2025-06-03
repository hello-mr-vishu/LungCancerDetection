from ultralytics import YOLO

# Load model
model = YOLO('yolov8s.pt')

# Train
model.train(
    data='D:/Projects/LungCancerDetection/data/lc25000.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,  # Use GPU
    patience=10,  # Early stopping
    name='yolov8s_lc25000'
)

print("Training completed!")