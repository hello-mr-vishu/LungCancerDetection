from pathlib import Path
import os

# Path to unzipped dataset
dataset_dir = Path('D:/Projects/LungCancerDetection/data/raw/lung_colon_image_set')

# List contents recursively
def list_dir(path, level=0):
    print('  ' * level + f'{path.name}/')
    for item in path.iterdir():
        if item.is_dir():
            list_dir(item, level + 1)
        else:
            print('  ' * (level + 1) + item.name)

list_dir(dataset_dir)