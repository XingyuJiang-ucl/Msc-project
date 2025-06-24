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

TARGET_SIZE = (224, 224)

for split, prefixes in splits.items():
    split_dir = output_base / split
    split_dir.mkdir(parents=True, exist_ok=True)

    image_data = []

    for prefix in prefixes:
        # derive numeric case_id from the prefix
        # prefix like "UCLH-Cyst_0001"
        case_num = int(prefix.split('_')[-1])
        case_folder = f'case_{case_num:05d}'
        out_case_dir = split_dir / case_folder
        out_case_dir.mkdir(exist_ok=True)

        # record for JSON
        image_data.append({
            'case_id': case_num,
            'folder':  case_folder
        })

        # find the matching image .nii.gz file
        nii_files = list(input_dir.glob(f"{prefix}*.nii.gz"))
        if not nii_files:
            print(f"[{split}] Warning: no image NIfTI found for prefix {prefix}")
            continue
        nii_path = nii_files[0]

        # find the matching segmentation .nii.gz file (prefix without "_0000")
        seg_path = pred_dir / f"{prefix}.nii.gz"
        if not seg_path.exists():
            print(f"[{split}] Warning: no segmentation found for prefix {prefix}, skipping case")
            continue

        # load image volume
        img = nib.load(str(nii_path))
        vol = img.get_fdata()
        if vol.ndim == 4:
            vol = vol[..., 0]

        # load segmentation volume
        seg_img = nib.load(str(seg_path))
        seg_vol = seg_img.get_fdata().astype(np.int32)

        # make sure dimensions match
        if seg_vol.shape != vol.shape:
            print(f"[{split}] Warning: image and segmentation shapes differ for {prefix}. Skipping.")
            continue

        # slice along axial (z) axis
        num_slices = vol.shape[2]
        saved_count = 0

        for z in range(num_slices):
            seg_slice = seg_vol[:, :, z]
            # check if this slice contains value 29
            if not np.any(seg_slice == 29):
                continue

            # extract the corresponding image slice
            slice_2d = vol[:, :, z]

            # min-max normalize to [0,255]
            mn, mx = slice_2d.min(), slice_2d.max()
            if mx > mn:
                norm = (slice_2d - mn) / (mx - mn)
            else:
                norm = np.zeros_like(slice_2d)
            slice_uint8 = (norm * 255).astype(np.uint8)

            # resize to 224×224
            img_pil = Image.fromarray(slice_uint8)
            img_resized = img_pil.resize(TARGET_SIZE, resample=Image.BICUBIC)

            # rotate 90° if desired (example shown below; remove if not needed)
            img_rotated = img_resized.rotate(90, expand=True)

            # save PNG using incremental slice index
            out_path = out_case_dir / f"slice_{saved_count:03d}.png"
            img_rotated.save(out_path)
            saved_count += 1

        if saved_count == 0:
            print(f"[{split}] Info: no slices with label 29 found for {prefix}")
        else:
            print(f"[{split}] processed {case_folder}: saved {saved_count} slices")

    # write JSON metadata
    json_path = split_dir / f"image_data_{split}.json"
    with open(json_path, 'w') as jf:
        json.dump(image_data, jf, indent=4)
    print(f"Wrote metadata for {split} to {json_path}")
