import os
from pathlib import Path
from PIL import Image
import numpy as np

# Define input and output directories
input_base = Path(r'D:\Msc project\skin lesion\images')
output_base = Path(r'D:\Msc project\skin lesion\resized_224_norm')

# Subfolders to process
parts = ['imgs_part_1', 'imgs_part_2', 'imgs_part_3']

# Target size
TARGET_SIZE = (224, 224)

for part in parts:
    src_dir = input_base / part
    dst_dir = output_base / part
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Loop over all PNG files in the source directory
    for img_path in src_dir.glob('*.png'):
        # Open image
        with Image.open(img_path) as img:
            # Resize with bicubic interpolation
            img_resized = img.resize(TARGET_SIZE, resample=Image.BICUBIC)

        # Convert to NumPy array (H×W×C) and to float32
        arr = np.asarray(img_resized).astype(np.float32)

        # Min–max normalize to [0,1]
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max > arr_min:
            arr_norm = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr_norm = np.zeros_like(arr)

        # Scale back to [0,255] and convert to uint8
        arr_uint8 = (arr_norm * 255).round().astype(np.uint8)

        # Convert back to PIL Image and save
        img_norm = Image.fromarray(arr_uint8)
        img_norm.save(dst_dir / img_path.name)

    print(f"Processed {part}: saved resized & normalized images to {dst_dir}")
