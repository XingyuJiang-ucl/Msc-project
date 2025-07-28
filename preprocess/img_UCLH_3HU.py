import os
import json
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
from PIL import Image

# Paths
csv_path      = Path(r'D:\Msc project\UCLH-Cyst\data\csv\outcomes.csv')
input_dir     = Path(r'D:\Msc project\UCLH-Cyst\data\all_scans\all_scans')
pred_dir      = Path(r'D:\Msc project\UCLH-Cyst\data\all_scans\all_predictions\umamba_predictions')
output_base   = Path(r'D:\Msc project\UCLH-Cyst\data\preprocessed')

# Read the CSV and extract the first-column prefixes
df = pd.read_csv(csv_path, header=0)
prefixes = df.iloc[:, 0].astype(str).tolist()  # e.g. "UCLH-Cyst_0001"

# Split into train/test/val by index
train_prefixes = prefixes[:70]
test_prefixes  = prefixes[70:80]
val_prefixes   = prefixes[80:90]

splits = {
    'train': train_prefixes,
    'test':  test_prefixes,
    'val':   val_prefixes,
}

# Target size for 2D slices
TARGET_SIZE = (224, 224)

def window_hu(volume: np.ndarray,
              hu_min: float,
              hu_max: float) -> np.ndarray:
    """
    Apply HU windowing (clipping) to the CT volume.
    - volume: 3D array of HU values
    - hu_min: lower bound
    - hu_max: upper bound
    Returns clipped volume.
    """
    return np.clip(volume, hu_min, hu_max)

# Window settings
WINDOWS_3CH = [
    (-135, 215),   # channel 0
    (0, 1000),     # channel 1
    (-650,850)  # channel 2
]

for split, prefixes in splits.items():
    split_dir = output_base / split
    split_dir.mkdir(parents=True, exist_ok=True)

    image_data = []

    for prefix in prefixes:
        # Derive numeric case_id from the prefix
        case_num = int(prefix.split('_')[-1])
        case_folder = f'case_{case_num:05d}'
        out_case_dir = split_dir / case_folder
        out_case_dir.mkdir(exist_ok=True)

        # Record for JSON
        image_data.append({
            'case_id': case_num,
            'folder':  case_folder
        })

        # Find matching image .nii.gz file
        nii_files = list(input_dir.glob(f"{prefix}*.nii.gz"))
        if not nii_files:
            print(f"[{split}] Warning: no image NIfTI found for prefix {prefix}")
            continue
        nii_path = nii_files[0]

        # Find corresponding segmentation .nii.gz file
        seg_path = pred_dir / f"{prefix}.nii.gz"
        if not seg_path.exists():
            print(f"[{split}] Warning: no segmentation found for prefix {prefix}, skipping case")
            continue

        # Load image volume (no resampling)
        img = nib.load(str(nii_path))
        vol = img.get_fdata()
        if vol.ndim == 4:
            vol = vol[..., 0]

        # Load segmentation volume (no resampling)
        seg_img = nib.load(str(seg_path))
        seg_vol = seg_img.get_fdata().astype(np.int32)
        if seg_vol.ndim == 4:
            seg_vol = seg_vol[..., 0]

        # Ensure shapes match
        if seg_vol.shape != vol.shape:
            print(f"[{split}] Warning: image and segmentation shapes differ for {prefix}. Skipping.")
            continue

        # Slice along the axial (z) axis
        num_slices = vol.shape[2]
        saved_count = 0

        for z in range(num_slices):
            seg_slice = seg_vol[:, :, z]
            # Check if this slice contains the label value 29
            if not np.any(seg_slice == 29):
                continue

            # Create a 3-channel array for this slice
            h, w = vol.shape[0], vol.shape[1]
            three_ch = np.zeros((h, w, 3), dtype=np.uint8)

            # For each window, apply windowing, normalize, and fill a channel
            for ch_idx, (hu_min, hu_max) in enumerate(WINDOWS_3CH):
                # Apply HU window and extract 2D
                vol_windowed = window_hu(vol, hu_min, hu_max)
                slice_2d = vol_windowed[:, :, z]

                # Normalize this 2D slice to [0, 255]
                mn, mx = slice_2d.min(), slice_2d.max()
                if mx > mn:
                    norm = (slice_2d - mn) / (mx - mn)
                else:
                    norm = np.zeros_like(slice_2d)
                slice_uint8 = (norm * 255).astype(np.uint8)

                # Assign to the appropriate channel
                three_ch[:, :, ch_idx] = slice_uint8

            # Convert to PIL Image (mode='RGB')
            img_pil = Image.fromarray(three_ch)

            # Resize to 224×224
            img_resized = img_pil.resize(TARGET_SIZE, resample=Image.BICUBIC)

            # Rotate 90°
            img_rotated = img_resized.rotate(90, expand=True)

            # Save PNG with incremental slice index
            out_path = out_case_dir / f"slice_{saved_count:03d}.png"
            img_rotated.save(out_path)
            saved_count += 1

        if saved_count == 0:
            print(f"[{split}] Info: no slices with label 29 found for {prefix}")
        else:
            print(f"[{split}] processed {case_folder}: saved {saved_count} slices")

    # Write JSON metadata
    json_path = split_dir / f"image_data_{split}.json"
    with open(json_path, 'w') as jf:
        json.dump(image_data, jf, indent=4)
    print(f"Wrote metadata for {split} to {json_path}")
