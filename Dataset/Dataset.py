import json
import os
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class ImageDataset_1(Dataset):
    def __init__(self, image_json_file, image_root, transform=None):
        """
            For CT scans dataset
           Initialization:
           - image_json_file: JSON file containing information for each case
           - image_root: root directory path for all images.
           - transform: optional preprocessing operations to apply to each image.

           We load the JSON, then iterate through each case’s folder to build
           a list `self.samples`, where each entry is a dict containing:
               - "image_path": full path to the image file
               - "case_id": identifier of the case this image belongs to
           """
        with open(image_json_file, 'r') as f:
            self.image_data = json.load(f)
        self.image_root = image_root
        self.transform = transform

        # Build a list of sample dicts for every image across all cases
        self.samples = []
        for entry in self.image_data:
            case_id = entry["case_id"]
            folder = entry["folder"]
            folder_path = os.path.join(self.image_root, folder)
            image_files = sorted(os.listdir(folder_path))
            for img_file in image_files:
                image_path = os.path.join(folder_path, img_file)
                self.samples.append({
                    "image_path": image_path,
                    "case_id": case_id
                })

    def __getitem__(self, index):
        """
        Retrieve the sample at `index` from self.samples:
        - Open the image
        - Convert to single-channel ('L') for 1-channel input
          (or comment out for 3-channel input)
        - Apply `self.transform` if provided
        Returns:
            (image_tensor, case_id)
        """
        sample = self.samples[index]
        # use this for 1 channel
        image = Image.open(sample["image_path"]).convert("L")
        # # use this for 3 channel
        # image = Image.open(sample["image_path"])
        if self.transform:
            image = self.transform(image)
        return image, sample["case_id"]

    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return len(self.samples)



class ImageDataset_2(Dataset):
    """
    Dataset2 (for skin lesion dataset):
    - Accepts `image_root` as a pathlib.Path or string.
    - Automatically walks all subdirectories to collect valid image files.
    - Returns (image_tensor, case_id), where case_id is the image filename.
    """
    VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    def __init__(self, image_root, transform=None):
        """
        Args:
            image_root (str or Path): Root directory containing images.
            transform: Optional torchvision.transforms to apply.
        """
        self.image_root = Path(image_root)
        self.transform = transform
        self.samples = []

        # Walk through image_root (including subdirectories)
        # and collect all files with valid image extensions
        for path in sorted(self.image_root.rglob('*')):
            if path.is_file() and path.suffix.lower() in self.VALID_EXTENSIONS:
                self.samples.append({
                    "image_path": path,
                    "case_id": path.name
                })

        if not self.samples:
            raise RuntimeError(f"No valid images found in {self.image_root}")

    def __getitem__(self, index):
        """
        Return a single sample:
        - Open the image (convert Path to str for PIL)
        - Convert to single-channel ('L')
        - Apply transform if given
        Returns:
            (image_tensor, case_id)
        """
        sample = self.samples[index]
        img = Image.open(str(sample["image_path"])).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, sample["case_id"]

    def __len__(self):
        """
        Return the total number of images collected.
        """
        return len(self.samples)