import os
from pathlib import Path
import zipfile
import shutil

# Define paths
project_dir = Path('D:/Projects/LungCancerDetection')
raw_data_dir = project_dir / 'data' / 'raw'

# Create directories
raw_data_dir.mkdir(parents=True, exist_ok=True)

# Download dataset
os.system(f'kaggle datasets download -d andrewmvd/lung-and-colon-cancer-histopathological-images -p "{raw_data_dir}"')

# Unzip dataset
zip_path = raw_data_dir / 'lung-and-colon-cancer-histopathological-images.zip'
extract_path = raw_data_dir / 'lung_colon_image_set'
if zip_path.exists():
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_data_dir)
    # Fix nesting if present
    nested_dir = raw_data_dir / 'lung_colon_image_set' / 'lung_colon_image_set'
    if nested_dir.exists():
        for item in nested_dir.iterdir():
            shutil.move(str(item), str(extract_path))
        shutil.rmtree(nested_dir)
    print("Dataset unzipped successfully!")
else:
    print(f"Error: {zip_path} not found.")
    exit(1)

# Verify structure
if extract_path.exists():
    print("Directory contents:", os.listdir(extract_path))
else:
    print(f"Error: {extract_path} not found.")