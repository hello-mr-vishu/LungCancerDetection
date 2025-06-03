import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
raw_data_dir = Path('D:/Projects/LungCancerDetection/data/raw/lung_colon_image_set')
output_dir = Path('D:/Projects/LungCancerDetection/data')
img_size = 640

# Class mapping: cancer (0), non-cancer (1)
class_map = {
    'lung_aca': 0,
    'lung_scc': 0,
    'colon_aca': 0,
    'lung_n': 1,
    'colon_n': 1
}

def create_dirs():
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

def process_images():
    images = []
    labels = []
    
    # Check if base directory exists
    if not raw_data_dir.exists():
        print(f"Error: {raw_data_dir} does not exist.")
        exit(1)
    
    # Define subfolder groups
    subfolder_groups = {
        'lung_image_sets': ['lung_aca', 'lung_scc', 'lung_n'],
        'colon_image_sets': ['colon_aca', 'colon_n']
    }
    
    # Iterate through subfolder groups and their classes
    for group_name, class_names in subfolder_groups.items():
        group_dir = raw_data_dir / group_name
        if not group_dir.exists():
            print(f"Warning: {group_dir} does not exist. Skipping.")
            continue
        for class_name in class_names:
            class_dir = group_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist. Skipping.")
                continue
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.jpeg'):
                    images.append(str(class_dir / img_name))
                    labels.append(class_map[class_name])
    
    # Check if images were found
    if not images:
        print("Error: No valid images found in subfolders. Check dataset structure.")
        exit(1)
    
    # Split dataset
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        images, labels, test_size=0.3, stratify=labels, random_state=42
    )
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    # Process each split
    for split, img_list, label_list in [
        ('train', train_imgs, train_labels),
        ('val', val_imgs, val_labels),
        ('test', test_imgs, test_labels)
    ]:
        for img_path, label in zip(img_list, label_list):
            # Read and resize image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to read {img_path}")
                continue
            img = cv2.resize(img, (img_size, img_size))
            img_name = Path(img_path).name
            cv2.imwrite(str(output_dir / 'images' / split / img_name), img)
            
            # Create annotation: full-image bounding box
            with open(output_dir / 'labels' / split / img_name.replace('.jpeg', '.txt'), 'w') as f:
                f.write(f'{label} 0.5 0.5 1.0 1.0\n')

def create_yaml():
    yaml_content = f"""
train: {output_dir}/images/train
val: {output_dir}/images/val
test: {output_dir}/images/test
nc: 2
names: ['cancer', 'non-cancer']
"""
    with open(output_dir / 'lc25000.yaml', 'w') as f:
        f.write(yaml_content)

if __name__ == '__main__':
    create_dirs()
    process_images()
    create_yaml()
    print("Dataset prepared successfully!")
    print(f"Train images: {len(os.listdir(output_dir / 'images' / 'train'))}")
    print(f"Val images: {len(os.listdir(output_dir / 'images' / 'val'))}")
    print(f"Test images: {len(os.listdir(output_dir / 'images' / 'test'))}")