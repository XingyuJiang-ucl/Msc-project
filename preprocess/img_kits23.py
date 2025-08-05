import os
import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import zoom


input_base  = r'D:\Msc project\kits23\dataset'
output_base = r'D:\Msc project\code\2D Slices 224 Tumor'

TARGET_SIZE    = (224, 224)
TARGET_SPACING = (1.5, 1.5, 1.5)

for i in range(589):
    case_name     = f'case_{i:05d}'
    case_input_dir= os.path.join(input_base, case_name)
    img_path      = os.path.join(case_input_dir, 'imaging.nii.gz')
    seg_path      = os.path.join(case_input_dir, 'segmentation.nii.gz')

    # Determine split based on case index
    if i <= 410:
        split = 'train'
    elif i <= 499:
        split = 'val'
    else:
        split = 'test'

    split_dir      = os.path.join(output_base, split)
    case_output_dir= os.path.join(split_dir, case_name)
    os.makedirs(case_output_dir, exist_ok=True)

    if not os.path.exists(img_path) or not os.path.exists(seg_path):
        print(f"[{split}] Missing files for {case_name}, skipping")
        continue

    img     = nib.load(img_path)
    seg     = nib.load(seg_path)
    vol     = img.get_fdata()
    seg_vol = seg.get_fdata().astype(np.int32)

    # Get original spacing
    sx, sy, sz = img.header.get_zooms()[:3]

    # Compute resampling factors
    zf_x = sx / TARGET_SPACING[0]
    zf_y = sy / TARGET_SPACING[1]
    zf_z = sz / TARGET_SPACING[2]

    # Resample volumes
    vol_rs = zoom(vol,     (zf_x, zf_y, zf_z), order=1)
    seg_rs = zoom(seg_vol, (zf_x, zf_y, zf_z), order=0)

    # Number of axial slices after resampling
    num_slices = vol_rs.shape[0]

    saved = 0
    for z in range(num_slices):
        # only keep slices where segmentation==2 (tumor)
        if not np.any(seg_rs[z, :, :] == 2):
            continue

        slice_data = vol_rs[z, :, :]
        mn, mx    = slice_data.min(), slice_data.max()
        if mx > mn:
            slice_norm = (slice_data - mn) / (mx - mn)
        else:
            slice_norm = np.zeros_like(slice_data)
        slice_uint8 = (slice_norm * 255).astype(np.uint8)

        img_pil       = Image.fromarray(slice_uint8).resize(TARGET_SIZE, resample=Image.BICUBIC)
        slice_fname   = f'slice_{saved:03d}.png'
        slice_outpath = os.path.join(case_output_dir, slice_fname)
        img_pil.save(slice_outpath)
        saved += 1

    print(f"[{split}] {case_name}: found {saved} tumor slices out of {num_slices}")

