import os
import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import zoom


input_base = r'D:\Msc project\kits23\dataset'
output_base = r'D:\Msc project\code\2D Slices 224 Tumor 3HU'

TARGET_SIZE = (224, 224)
TARGET_SPACING = (1.5, 1.5, 1.5)

# Window settings for three channels
WINDOWS_3CH = [
    (-135, 215),   # channel 0
    (0,   1000),   # channel 1
    (-650, 850)    # channel 2
]

for i in range(589):
    case_name = f'case_{i:05d}'
    case_input_dir = os.path.join(input_base, case_name)
    img_path = os.path.join(case_input_dir, 'imaging.nii.gz')
    seg_path = os.path.join(case_input_dir, 'segmentation.nii.gz')

    # Determine split based on case index
    if i <= 410:
        split = 'train'
    elif i <= 499:
        split = 'val'
    else:
        split = 'test'

    # Prepare output directory for this split
    split_dir = os.path.join(output_base, split)
    case_output_dir = os.path.join(split_dir, case_name)
    os.makedirs(case_output_dir, exist_ok=True)

    if not os.path.exists(img_path) or not os.path.exists(seg_path):
        print(f"[{split}] Missing files for {case_name}, skipping")
        continue

    img = nib.load(img_path)
    seg = nib.load(seg_path)
    vol = img.get_fdata()
    seg_vol = seg.get_fdata().astype(np.int32)

    # Get original spacing and compute zoom factors
    sx, sy, sz = img.header.get_zooms()[:3]
    zf_x = sx / TARGET_SPACING[0]
    zf_y = sy / TARGET_SPACING[1]
    zf_z = sz / TARGET_SPACING[2]

    # Resample volumes
    vol_rs = zoom(vol, (zf_x, zf_y, zf_z), order=1)
    seg_rs = zoom(seg_vol, (zf_x, zf_y, zf_z), order=0)

    num_slices = vol_rs.shape[0]

    saved = 0
    for z in range(num_slices):
        # only keep slices where segmentation==2 (tumor)
        if not np.any(seg_rs[z, :, :] == 2):
            continue

        slice_2d = vol_rs[z, :, :]
        h, w = slice_2d.shape

        # build a 3-channel image
        three_ch = np.zeros((h, w, 3), dtype=np.uint8)
        for ch_idx, (hu_min, hu_max) in enumerate(WINDOWS_3CH):
            win = np.clip(slice_2d, hu_min, hu_max)
            mn, mx = win.min(), win.max()
            if mx > mn:
                norm = (win - mn) / (mx - mn)
            else:
                norm = np.zeros_like(win)
            three_ch[:, :, ch_idx] = (norm * 255).astype(np.uint8)

        # convert to PIL and resize
        img_pil = Image.fromarray(three_ch)
        img_resized = img_pil.resize(TARGET_SIZE, resample=Image.BICUBIC)

        out_fname = os.path.join(case_output_dir, f'slice_{saved:03d}.png')
        img_resized.save(out_fname)
        saved += 1

    print(f"[{split}] {case_name}: saved {saved}/{num_slices} tumor slices")

