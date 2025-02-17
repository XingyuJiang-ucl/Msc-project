import json
import os
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset1(Dataset):
    """
    This image dataset for calculate tabular data embedding in advanced
    """
    def __init__(self, ann_file, image_root, transform=None):
        self.ann = json.load(open(ann_file, 'r'))
        self.image_root = image_root
        self.transform = transform
        # Here we assume each annotation has a 'table_id'
        # For a one-to-many relationship, multiple images can share the same table_id

    def __getitem__(self, index):
        item = self.ann[index]
        image_path = os.path.join(self.image_root, item['image'])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Return both image and the associated table identifier
        table_id = item['table_id']
        return image, table_id

    def __len__(self):
        return len(self.ann)
