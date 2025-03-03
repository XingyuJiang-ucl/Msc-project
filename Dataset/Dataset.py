import json
import os
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset_1(Dataset):
    def __init__(self, image_json_file, image_root, transform=None):
        """
        初始化方法：
        - image_json_file：包含每个 case 信息的 JSON 文件（例如，每条记录包含 "case_id" 和 "folder"）。
        - image_root：图像的根目录路径，例如 "D:\\MSc project\\kits23\\2D Slices"。
        - transform：图像预处理操作，如 Resize、ToTensor 等。

        这里，我们先加载 JSON 文件，然后遍历每个 case 文件夹，
        对每个 case 中的所有图片，构造一个样本列表 self.samples，每个样本包含：
            - image_path：图片的完整路径。
            - case_id：该图片所属的 case 的编号。
        """
        with open(image_json_file, 'r') as f:
            self.image_data = json.load(f)
        self.image_root = image_root
        self.transform = transform

        # 构建包含所有图片信息的列表
        self.samples = []
        for entry in self.image_data:
            case_id = entry["case_id"]
            folder = entry["folder"]
            folder_path = os.path.join(self.image_root, folder)
            # 列出该文件夹中的所有图片（假设文件夹中只包含图片文件）
            image_files = sorted(os.listdir(folder_path))
            for img_file in image_files:
                image_path = os.path.join(folder_path, img_file)
                self.samples.append({
                    "image_path": image_path,
                    "case_id": case_id
                })

    def __getitem__(self, index):
        """
        根据索引 index，从 self.samples 中读取对应图片的路径和 case_id，
        打开图片、转换为 RGB 格式，并应用预处理操作（如果有）。
        返回：(image, case_id)
        """
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, sample["case_id"]

    def __len__(self):
        """
        返回数据集中所有图片的总数量，即 self.samples 列表的长度。
        """
        return len(self.samples)
